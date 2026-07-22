"""
Convert a VGGT per-view export into a LitePT (ScanNet-schema) scene directory so a
trained InsSeg model can run on VGGT-generated geometry.

This mirrors the RGB-D preprocessing in
`datasets/preprocessing/v1_0_1/preprocess_v1_0_1.py` (fuse views -> optional crop ->
voxel-downsample -> camera-oriented PCA normals), but ingests VGGT outputs instead of
HDF5 depth, and writes NO labels (inference is GT-free). It reuses that file's
`compute_normals` and `voxel_first_hit` verbatim so the per-point feature distribution
(`feat = concat(color/255, normal)`) matches training.

------------------------------------------------------------------------------------
EXPECTED INPUT  (one .npz per scene; this is exactly the VGGT `predictions` dict)
------------------------------------------------------------------------------------
Required:
  images        (S,3,H,W) or (S,H,W,3)   RGB; float in [0,1] or uint8 [0,255]
  extrinsic     (S,3,4) or (S,4,4)        camera extrinsic. VGGT convention is
                                          world->cam (w2c, OpenCV). Pass
                                          --extrinsic-c2w if yours is cam->world.
  one source of 3D points, first found wins (override with --points-key):
    world_points / world_points_from_depth / points / pts3d / xyz
                                          (S,H,W,3) world-frame point map
    -- OR -- depth / depth_map (S,H,W[,1]) PLUS intrinsic (S,3,3): unprojected here.

Optional:
  <conf>        (S,H,W)                   per-pixel confidence; key auto-detected from
                                          world_points_conf / depth_conf / conf /
                                          confidence / point_conf (override --conf-key).
                                          Filtered by --conf-threshold (absolute) and/or
                                          --conf-percentile (within the kept region).
  filtered_valid_mask / valid_mask
                (S,H,W) bool              VGGT's per-view validity. --valid-mask selects
                                          which one is AND-ed into the kept region:
                                          filtered (default) -> filtered_valid_mask,
                                          valid -> valid_mask, none -> finite-only
                                          (legacy; reproduces pre-mask scenes). Missing
                                          key -> warn + fall back to none.

Validity per view = isfinite(point) & stored_mask & (conf >= conf_thr). The confidence
floor is evaluated *within* the masked region, so --conf-percentile stacks on top of
the stored mask.

Units MUST be meters (the model bakes in a 2 mm voxel grid and 5 mm cluster radius).
The output is written to <output_root>/<split>/<scene_name>/ as
coord.npy (float32 [N,3]), color.npy (uint8 [N,3]), normal.npy (float32 [N,3]).

Example:
  /home/fai/miniconda3/envs/litept/bin/python tools/vggt_to_scene.py \
      --input /path/to/vggt_scene0.npz \
      --output-root data/scannet-v1.1.1-2of3-vggt --split test --scene scene_vggt0 \
      --valid-mask filtered
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_preprocess_helpers():
    """Import compute_normals / voxel_first_hit from the v1_0_1 preprocessing script.

    That directory is not a package (no __init__.py), so load it by file path.
    """
    path = os.path.join(
        _REPO_ROOT, "datasets", "preprocessing", "v1_0_1", "preprocess_v1_0_1.py"
    )
    spec = importlib.util.spec_from_file_location("preprocess_v1_0_1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_normals, mod.voxel_first_hit


# default VisionSCO load_fn.py location (the same preprocessing VGGT runs at recon time)
_VISIONSCO_LOAD_FN = "/home/fai/workspace/jhp/VisionSCO/models/vggt/vggt/utils/load_fn.py"


def _load_visionsco_load_fn(path=None):
    """Import VisionSCO's preprocess_view / load_cameras_from_json by file path.

    Used to transform D-FINE-seg 2D masks (raw 640x480 image space) through the EXACT
    same chain VGGT applied to the images (undistort -> rotate -> center_pp -> resize),
    so the masks land on the VGGT point grid (e.g. 392x518) pixel-for-pixel.
    """
    path = path or _VISIONSCO_LOAD_FN
    if not os.path.isfile(path):
        raise SystemExit(f"VisionSCO load_fn not found at {path} (override --load-fn)")
    spec = importlib.util.spec_from_file_location("visionsco_load_fn", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _decode_rle(seg):
    """Decode a COCO segmentation (uncompressed list-RLE or compressed) to (H,W) uint8.

    The D-FINE-seg masks store uncompressed, column-major (Fortran) RLE
    ({"counts": [int,...], "size": [H, W]}); decode it without pycocotools so this
    works regardless of the env. Falls back to pycocotools for byte/str counts.
    """
    h, w = int(seg["size"][0]), int(seg["size"][1])
    counts = seg["counts"]
    if isinstance(counts, (list, tuple)):
        flat = np.zeros(h * w, dtype=np.uint8)
        idx = 0
        val = 0
        for c in counts:
            if val:
                flat[idx : idx + c] = 1
            idx += c
            val ^= 1
        return flat.reshape((w, h)).T  # column-major -> (H, W)
    import pycocotools.mask as mask_util  # compressed RLE
    return mask_util.decode(seg).astype(np.uint8)


def _view_token(file_name):
    """Map a COCO image file_name to its camera view token (e.g. 'TB').

    Handles 'TB_L.png', 'cam-TB_L.jpg', and timestamped names like
    'TB_L_2025-12-09_10_40_00.353196.png'. The view token is the LEADING
    '_'-delimited field (an optional 'prefix-' is stripped); the side tag (_L/_R)
    and any trailing timestamp fields are ignored.
    """
    base = os.path.splitext(os.path.basename(file_name))[0]
    first = base.split("_")[0]          # leading field, e.g. 'TB' or 'cam-TB'
    if "-" in first:                    # strip an optional 'prefix-' before the token
        first = first.split("-")[-1]
    return first


def _load_coco_per_instance(coco_path, input_order_list, original_size):
    """Load D-FINE-seg COCO masks as PER-INSTANCE (not unioned) masks per view.

    Returns a list of length len(input_order_list); entry i is (S_i, H0, W0) uint8
    (0/255) holding one channel per annotation of that view (category_id != 0). The
    view order follows ``input_order_list`` so it matches the VGGT world_points axis.
    """
    import json

    W0, H0 = int(original_size[0]), int(original_size[1])
    with open(coco_path, "r") as f:
        coco = json.load(f)
    token_to_id = {_view_token(img["file_name"]): img["id"] for img in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    per_view = []
    for view in input_order_list:
        chans = []
        img_id = token_to_id.get(str(view))
        if img_id is not None:
            for ann in anns_by_img.get(img_id, []):
                if ann.get("category_id", 0) == 0:
                    continue
                m = _decode_rle(ann["segmentation"])
                if m.shape != (H0, W0):
                    m = m[:H0, :W0]
                chans.append((m > 0).astype(np.uint8) * 255)
        per_view.append(
            np.stack(chans, axis=0) if chans else np.zeros((0, H0, W0), dtype=np.uint8)
        )
    return per_view


def _prepare_aligned_masks(coco_path, camera_path, input_order_list, hw, args):
    """Load D-FINE-seg COCO masks and align them to the VGGT (H, W) point grid.

    Shared by the single-view (origin-pixel) and multi-view (reprojection) labelers.
    Returns (aligned, areas, offsets):
      aligned[s] : (S_i, H, W) bool   -- per-instance masks of view s on the grid
      areas[s]   : (S_i,) int64       -- aligned pixel area (smallest wins on overlap)
      offsets    : (n_views+1,) int64 -- per-view base so each (view, instance) gets a
                                         globally-unique id (offsets[s] + local).
    """
    H, W = int(hw[0]), int(hw[1])
    lf = _load_visionsco_load_fn(args.load_fn)
    per_view_raw = _load_coco_per_instance(
        coco_path, input_order_list, args.mask_original_size
    )

    # offset so each (view, local-instance) gets a unique global id
    offsets = np.cumsum([0] + [m.shape[0] for m in per_view_raw]).astype(np.int64)
    n_views = len(input_order_list)
    aligned = []  # per view: (S_i, H, W) bool aligned to world_points
    areas = []    # per view: (S_i,) aligned area (px), for overlap tie-break

    if getattr(args, "mask_pre_aligned", False):
        # The masks are ALREADY on the VGGT world_points grid -- produced by
        # D-FINE-seg/scripts/align_coco_to_world_points.py, which bakes in the full
        # preprocess_view pipeline (undistort -> rotate -> center_pp -> crop-resize) and
        # re-encodes the result at (H, W). Re-running preprocess_view here would
        # DOUBLE-apply those steps (esp. the 180 deg rotation on TF/TR/TL/RW/LW and the
        # center_pp padding -> measured IoU vs the aligned masks drops to ~0.2-0.5), so
        # index the decoded masks directly. No camera.json is needed on this path.
        for s in range(n_views):
            om = (per_view_raw[s] > 0).astype(bool)  # (S_i, H0, W0) on the aligned grid
            if om.shape[0] == 0:
                aligned.append(np.zeros((0, H, W), dtype=bool))
                areas.append(np.zeros(0, dtype=np.int64))
                continue
            if om.shape[1:] != (H, W):
                raise SystemExit(
                    f"--mask-pre-aligned: view {input_order_list[s]} mask grid "
                    f"{tuple(om.shape[1:])} != world_points grid {(H, W)}; pass "
                    f"--mask-original-size W,H matching the aligned masks (e.g. 518,392)"
                )
            aligned.append(om)
            areas.append(om.reshape(om.shape[0], -1).sum(1).astype(np.int64))
    else:
        cam = lf.load_cameras_from_json(
            camera_path,
            input_order_list=[str(v) for v in input_order_list],
            original_size=tuple(args.mask_original_size),
        )
        dummy_img = np.zeros(
            (int(args.mask_original_size[1]), int(args.mask_original_size[0]), 3),
            dtype=np.uint8,
        )
        for s in range(n_views):
            obj = per_view_raw[s]
            res = lf.preprocess_view(
                img_np=dummy_img.copy(),
                K=cam["intrinsics"][s],
                extrinsic=cam["extrinsics"][s],
                cam_view=str(input_order_list[s]),
                object_masks=obj if obj.shape[0] > 0 else None,
                dist_coeffs=cam["dist_coeffs_list"][s] if args.mask_undistort else None,
                original_size=tuple(args.mask_original_size),
                do_undistort=args.mask_undistort,
                center_pp=args.mask_center_pp,
                target_size=(H, W),
                resize_mode=args.mask_resize_mode,
            )
            om = res["object_masks"]
            if om is None or om.shape[0] == 0:
                aligned.append(np.zeros((0, H, W), dtype=bool))
                areas.append(np.zeros(0, dtype=np.int64))
            else:
                aligned.append(om.astype(bool))
                areas.append(om.reshape(om.shape[0], -1).sum(1).astype(np.int64))

    return aligned, areas, offsets


def _view_label_image(aligned_s, areas_s, offset_s, H, W):
    """Per-pixel global mask id for one view (smallest-area instance wins), -1 if none.

    aligned_s (S_i, H, W) bool, areas_s (S_i,), offset_s int. Painting by best (smallest)
    area is order-independent and reproduces the argmin-area tie-break used per-point.
    """
    lab = np.full((H, W), -1, dtype=np.int32)
    if aligned_s.shape[0] == 0:
        return lab
    best_area = np.full((H, W), np.iinfo(np.int64).max, dtype=np.int64)
    for c in range(aligned_s.shape[0]):
        take = aligned_s[c] & (int(areas_s[c]) < best_area)
        lab[take] = int(offset_s + c)
        best_area[take] = int(areas_s[c])
    return lab


def _compute_point_mask_labels(
    coco_path, camera_path, input_order_list, cam_idx, pix_idx, hw, args
):
    """Per-point 2D-instance labels via the ORIGIN (view, pixel) of each fused point.

    A point fused from view s at pixel (h, w) takes the aligned instance covering (h, w)
    (smallest-area wins on overlap); -1 if none. Returns (mask_instance[N], mask_view[N]).
    """
    H, W = int(hw[0]), int(hw[1])
    aligned, areas, offsets = _prepare_aligned_masks(
        coco_path, camera_path, input_order_list, hw, args
    )

    n = cam_idx.shape[0]
    mask_instance = np.full(n, -1, dtype=np.int32)
    mask_view = cam_idx.astype(np.int32)
    hh = (pix_idx // W).astype(np.int64)
    ww = (pix_idx % W).astype(np.int64)
    for i in range(n):
        s = int(cam_idx[i])
        om = aligned[s]
        if om.shape[0] == 0:
            continue
        h_i, w_i = int(hh[i]), int(ww[i])
        if not (0 <= h_i < H and 0 <= w_i < W):
            continue
        hits = np.nonzero(om[:, h_i, w_i])[0]
        if hits.size == 0:
            continue
        local = hits[int(np.argmin(areas[s][hits]))] if hits.size > 1 else hits[0]
        mask_instance[i] = int(offsets[s] + local)
    return mask_instance, mask_view


def _compute_point_mask_labels_allviews(
    store, aligned, areas, offsets, coord, input_order_list, hw, args
):
    """Per-point 2D-mask labels in EVERY view, with z-buffer occlusion.

    Each fused point is reprojected into all S views using the aligned camera
    (intrinsic_aligned + extrinsic_pnp, both in the VGGT world frame, written by
    tools/align_vggt_cameras.py). A point is "visible" in view s iff it projects in
    front of the camera, inside the image, and is NOT behind the recorded front surface
    -- world_points[s] gives the per-pixel surface depth (the z-buffer); a point whose
    camera-z exceeds that surface depth by more than --mask-occlusion-tol is occluded.
    Where the surface is non-finite (no recorded point) there is no occluder, so the
    point is treated as visible. A visible point takes the aligned instance covering its
    landing pixel (smallest-area wins); -1 otherwise.

    Returns mask_per_view (N, S) int32: column s is the globally-unique mask id (or -1).
    Mask ids are unique per (view, instance), so two points fall in DIFFERENT masks of
    the SAME view iff some column holds two distinct non-negative values -- exactly the
    cannot-link signal _split_proposals_by_mask_multiview consumes.
    """
    H, W = int(hw[0]), int(hw[1])
    n_views = len(input_order_list)

    if "intrinsic_aligned" not in store or "extrinsic_pnp" not in store:
        raise SystemExit(
            "--mask-all-views needs 'intrinsic_aligned' and 'extrinsic_pnp' in the VGGT "
            "npz (the aligned camera used to reproject the cloud); run "
            "tools/align_vggt_cameras.py on these scenes first"
        )
    K_all = np.asarray(store["intrinsic_aligned"], dtype=np.float64)  # (S,3,3)
    ext_all = np.asarray(store["extrinsic_pnp"], dtype=np.float64)    # (S,3,4)
    _, world_points = _first_present(store, POINTS_KEYS, args.points_key)
    if world_points is None:
        raise SystemExit(
            "--mask-all-views needs a world-point map (one of "
            f"{POINTS_KEYS}) in the npz for the occlusion z-buffer"
        )
    world_points = np.asarray(world_points, dtype=np.float64)         # (S,H,W,3)
    if K_all.shape[0] < n_views or ext_all.shape[0] < n_views or world_points.shape[0] < n_views:
        raise SystemExit(
            f"camera/point arrays have fewer than {n_views} views "
            f"(K={K_all.shape}, ext={ext_all.shape}, wp={world_points.shape})"
        )

    P = np.asarray(coord, dtype=np.float64)  # (N,3)
    N = P.shape[0]
    mask_per_view = np.full((N, n_views), -1, dtype=np.int32)
    tol = float(args.mask_occlusion_tol)

    for s in range(n_views):
        K = K_all[s]
        R = ext_all[s][:, :3]            # (3,3) world->cam rotation
        t = ext_all[s][:, 3]             # (3,) world->cam translation
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # front-surface depth buffer (camera-z of world_points[s]); NaN -> no occluder
        surf_z = (world_points[s] @ R[2, :]) + t[2]            # (H,W)

        # reproject all fused points into this view
        Pc = P @ R.T + t                                       # (N,3)
        z = Pc[:, 2]
        front = z > 1e-6
        zc = np.where(front, z, 1.0)
        u = fx * Pc[:, 0] / zc + cx
        v = fy * Pc[:, 1] / zc + cy
        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        in_b = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not in_b.any():
            continue

        idx = np.nonzero(in_b)[0]
        su = surf_z[vi[idx], ui[idx]]
        # occluded: surface recorded (finite) AND point meaningfully behind it
        occluded = np.isfinite(su) & (z[idx] > su + tol)
        vis = idx[~occluded]
        if vis.size == 0:
            continue

        lab_img = _view_label_image(aligned[s], areas[s], int(offsets[s]), H, W)
        mask_per_view[vis, s] = lab_img[vi[vis], ui[vis]]  # -1 where no mask covers

    return mask_per_view


POINTS_KEYS = ["world_points", "world_points_from_depth", "points", "pts3d", "xyz"]
IMAGE_KEYS = ["images", "image", "rgb", "rgbs"]
CONF_KEYS = ["world_points_conf", "depth_conf", "conf", "confidence", "point_conf"]
DEPTH_KEYS = ["depth", "depth_map", "depths"]
# stored per-view validity masks, keyed by the --valid-mask choice
VALID_MASK_KEYS = {"filtered": "filtered_valid_mask", "valid": "valid_mask", "none": None}
EXTRINSIC_KEYS = ["extrinsic", "extrinsics", "world2cam", "w2c", "cam2world", "c2w"]
INTRINSIC_KEYS = ["intrinsic", "intrinsics", "K"]


def _first_present(store, keys, override=None):
    if override is not None:
        if override not in store:
            raise KeyError(f"requested key {override!r} not in input (have {list(store)})")
        return override, store[override]
    for k in keys:
        if k in store:
            return k, store[k]
    return None, None


def _to_SHW3_images(images):
    """Normalize image array to (S,H,W,3) uint8 in [0,255]."""
    arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError(f"images must be 4-D, got shape {arr.shape}")
    # (S,3,H,W) -> (S,H,W,3)
    if arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.shape[-1] != 3:
        raise ValueError(f"could not resolve channel axis for images shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        # heuristic: values in [0,1] -> scale to [0,255]
        if np.nanmax(arr) <= 1.0 + 1e-3:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def _camera_origins(extrinsic, c2w):
    """Return (S,3) camera centers in world coords from (S,3,4) or (S,4,4) extrinsics."""
    E = np.asarray(extrinsic, dtype=np.float64)
    if E.ndim != 3 or E.shape[1] not in (3, 4) or E.shape[2] != 4:
        raise ValueError(f"extrinsic must be (S,3,4) or (S,4,4), got {E.shape}")
    R = E[:, :3, :3]
    t = E[:, :3, 3]
    if c2w:
        return t.astype(np.float32)  # cam->world: translation IS the camera center
    # world->cam: C = -R^T t
    return np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), -t).astype(np.float32)


def _unproject_depth(depth, K, extrinsic, c2w, depth_max):
    """Unproject one view's depth (H,W) to world coords (H,W,3); invalid -> NaN."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    x = (us - cx) / fx * depth
    y = (vs - cy) / fy * depth
    P_cam = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)  # (H,W,4)
    E = np.eye(4, dtype=np.float64)
    E[:3, :4] = extrinsic[:3, :4]
    Tc2w = E if c2w else np.linalg.inv(E)
    P_world = P_cam @ Tc2w.T
    out = P_world[..., :3].astype(np.float32)
    invalid = ~np.isfinite(depth) | (depth <= 0.0) | (depth >= depth_max)
    out[invalid] = np.nan
    return out


def build_scene(store, args):
    # --- resolve arrays -------------------------------------------------------
    _, images = _first_present(store, IMAGE_KEYS, args.image_key)
    if images is None:
        raise KeyError(f"no image array found (looked for {IMAGE_KEYS})")
    images = _to_SHW3_images(images)
    S, H, W, _ = images.shape

    ekey, extrinsic = _first_present(store, EXTRINSIC_KEYS, args.extrinsic_key)
    if extrinsic is None:
        raise KeyError(f"no extrinsic array found (looked for {EXTRINSIC_KEYS})")
    c2w = args.extrinsic_c2w or (ekey in ("cam2world", "c2w"))
    cam_origins = _camera_origins(extrinsic, c2w)

    pkey, pts = _first_present(store, POINTS_KEYS, args.points_key)
    if pts is not None:
        world_points = np.asarray(pts, dtype=np.float32)
        if world_points.shape[:3] != (S, H, W) or world_points.shape[-1] != 3:
            raise ValueError(
                f"point map {pkey!r} shape {world_points.shape} != (S,H,W,3)=({S},{H},{W},3)"
            )
    else:
        dkey, depth = _first_present(store, DEPTH_KEYS, args.depth_key)
        if depth is None:
            raise KeyError(
                f"no point map ({POINTS_KEYS}) and no depth ({DEPTH_KEYS}) in input"
            )
        _, K = _first_present(store, INTRINSIC_KEYS, args.intrinsic_key)
        if K is None:
            raise KeyError("depth supplied but no intrinsic (S,3,3) found for unprojection")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        K = np.asarray(K, dtype=np.float64)
        world_points = np.stack(
            [_unproject_depth(depth[s], K[s], np.asarray(extrinsic[s], np.float64),
                              c2w, args.depth_max) for s in range(S)],
            axis=0,
        )

    ckey, conf = _first_present(store, CONF_KEYS, args.conf_key)
    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)
        if conf.shape[:3] != (S, H, W):
            conf = conf.reshape(S, H, W)
    else:
        conf = np.ones((S, H, W), dtype=np.float32)

    # --- resolve the stored VGGT validity mask (default: filtered_valid_mask) -
    mask_key = args.valid_mask_key or VALID_MASK_KEYS[args.valid_mask]
    stored_mask = None
    if mask_key is not None:
        if mask_key in store:
            stored_mask = np.asarray(store[mask_key]).astype(bool)
            if stored_mask.shape[:3] != (S, H, W):
                stored_mask = stored_mask.reshape(S, H, W)
            print(f"[mask] applying stored {mask_key!r} "
                  f"({100.0 * stored_mask.mean():.1f}% of pixels kept)")
        else:
            print(f"[mask] WARN requested mask {mask_key!r} not in input "
                  f"(have {list(store)}); falling back to finite-only validity")

    # --- per-view valid masking + fuse ---------------------------------------
    # finite + (optional) stored mask define the kept region; the confidence
    # floor (absolute / percentile) is then applied *within* that region so a
    # higher --conf-percentile measures against the masked pixels, not all finite.
    conf_thr = args.conf_threshold
    if args.conf_percentile > 0:
        region = np.isfinite(world_points).all(axis=-1)
        if stored_mask is not None:
            region = region & stored_mask
        if region.any():
            conf_thr = max(conf_thr, float(np.percentile(conf[region], args.conf_percentile)))
    coords, colors, cam_idxs, pix_idxs = [], [], [], []
    for s in range(S):
        valid = np.isfinite(world_points[s]).all(axis=-1) & (conf[s] >= conf_thr)
        if stored_mask is not None:
            valid &= stored_mask[s]
        if not valid.any():
            continue
        coords.append(world_points[s][valid])
        colors.append(images[s][valid])
        cam_idxs.append(np.full(int(valid.sum()), s, dtype=np.int32))
        # row-major flat pixel index (h*W + w) of each kept pixel; aligns 1:1 with
        # world_points[s][valid], so we can recover the (h, w) origin of every point.
        pix_idxs.append(np.flatnonzero(valid).astype(np.int64))
    if not coords:
        raise RuntimeError("no valid points after confidence/finite masking")
    coord = np.concatenate(coords, axis=0).astype(np.float32)
    color = np.concatenate(colors, axis=0).astype(np.uint8)
    cam_idx = np.concatenate(cam_idxs, axis=0).astype(np.int32)
    pix_idx = np.concatenate(pix_idxs, axis=0).astype(np.int64)

    # --- optional working-volume AABB crop (centered on point-cloud median) ---
    if args.crop_half > 0:
        center = np.median(coord, axis=0)
        h = args.crop_half
        keep = np.all((coord >= center - h) & (coord <= center + h), axis=1)
        if not keep.any():
            raise RuntimeError("crop removed all points; increase --crop-half")
        coord, color, cam_idx, pix_idx = (
            coord[keep], color[keep], cam_idx[keep], pix_idx[keep]
        )

    return coord, color, cam_idx, cam_origins, pix_idx, (H, W)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="VGGT export .npz (or .npy dict)")
    ap.add_argument("--output-root", default="data/scannet-v1.1.1-2of3-vggt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--scene", required=True, help="output scene folder name")
    ap.add_argument("--voxel-size", type=float, default=0.002,
                    help="fuse/downsample voxel (m); match dataset build (2 mm).")
    ap.add_argument("--normal-k", type=int, default=16, help="k-NN for PCA normals")
    ap.add_argument("--valid-mask", choices=["filtered", "valid", "none"],
                    default="filtered",
                    help="which stored VGGT mask to AND into per-view validity: "
                         "filtered=filtered_valid_mask (default), valid=valid_mask, "
                         "none=finite-only (legacy). Falls back to none with a warning "
                         "if the requested mask key is absent.")
    ap.add_argument("--valid-mask-key", default=None,
                    help="explicit mask array key override (else from --valid-mask)")
    ap.add_argument("--conf-threshold", type=float, default=0.0,
                    help="absolute confidence floor; points below are dropped")
    ap.add_argument("--conf-percentile", type=float, default=0.0,
                    help="percentile floor (0-100) computed within the kept region "
                         "(finite & stored mask); 0 disables")
    ap.add_argument("--depth-max", type=float, default=10.0,
                    help="max depth (m) when unprojecting from a depth map")
    ap.add_argument("--crop-half", type=float, default=0.0,
                    help="if >0, keep an AABB of half-size H (m) around the median point")
    ap.add_argument("--extrinsic-c2w", action="store_true",
                    help="treat extrinsics as cam->world (default: world->cam / VGGT)")
    # explicit key overrides (else auto-detected)
    ap.add_argument("--points-key", default=None)
    ap.add_argument("--depth-key", default=None)
    ap.add_argument("--image-key", default=None)
    ap.add_argument("--conf-key", default=None)
    ap.add_argument("--extrinsic-key", default=None)
    ap.add_argument("--intrinsic-key", default=None)
    # --- optional per-point 2D-mask labels (for mask-constrained clustering) ---
    # When --coco (+ --camera) are given, each point's origin (view, pixel) is mapped to
    # the D-FINE-seg 2D instance covering it, written as mask_instance.npy / mask_view.npy.
    ap.add_argument("--coco", default=None,
                    help="D-FINE-seg coco_annotations.json; enables mask_instance.npy output")
    ap.add_argument("--camera", default=None,
                    help="stereo camera.json (K/dist/R/t) matching --coco; required with --coco")
    ap.add_argument("--load-fn", default=_VISIONSCO_LOAD_FN,
                    help="path to VisionSCO load_fn.py (preprocess_view/load_cameras_from_json)")
    ap.add_argument("--mask-original-size", default="640,480",
                    help="raw mask image size 'W,H' (default 640,480)")
    ap.add_argument("--mask-resize-mode", default="padding", choices=["padding", "crop"],
                    help="resize mode used at VGGT recon time (must match)")
    ap.add_argument("--no-mask-undistort", dest="mask_undistort", action="store_false",
                    help="skip mask undistortion (default: undistort, matching VGGT)")
    ap.add_argument("--no-mask-center-pp", dest="mask_center_pp", action="store_false",
                    help="skip mask center-pp (default: center_pp on, matching --apply_center_pp)")
    ap.add_argument("--mask-pre-aligned", action="store_true",
                    help="the --coco masks are ALREADY on the VGGT world_points grid (e.g. from "
                         "D-FINE-seg/scripts/align_coco_to_world_points.py, which bakes in "
                         "undistort->rotate->center_pp->crop). Index them directly: skip "
                         "preprocess_view (no re-undistort/rotate/center_pp/resize) and skip "
                         "camera.json. Set --mask-original-size to the aligned grid (e.g. 518,392).")
    ap.add_argument("--mask-all-views", action="store_true",
                    help="reproject EVERY fused point into ALL views (z-buffer occlusion via "
                         "world_points) and write mask_per_view.npy (N,S): the 2D mask each point "
                         "lands in per view, -1 if not visible/covered. Needs intrinsic_aligned + "
                         "extrinsic_pnp in the npz (run tools/align_vggt_cameras.py first). The "
                         "legacy mask_instance/mask_view (source-view column) are still written.")
    ap.add_argument("--mask-occlusion-tol", type=float, default=0.005,
                    help="z-buffer depth tolerance (m) for --mask-all-views: a reprojected point "
                         "is occluded only if its camera-z exceeds the recorded front surface by "
                         "more than this (default 5 mm).")
    ap.set_defaults(mask_undistort=True, mask_center_pp=True)
    args = ap.parse_args()
    args.mask_original_size = [int(x) for x in str(args.mask_original_size).split(",")]
    if args.coco and not args.camera and not args.mask_pre_aligned:
        ap.error("--coco requires --camera (camera.json with K/dist for mask preprocessing) "
                 "unless --mask-pre-aligned is set")

    compute_normals, _voxel_first_hit = _load_preprocess_helpers()

    raw = np.load(args.input, allow_pickle=True)
    if isinstance(raw, np.ndarray):  # .npy holding a 0-d object dict
        store = raw.item()
    else:  # NpzFile
        store = {k: raw[k] for k in raw.files}
    print(f"[load] {args.input} keys={sorted(store)}")

    coord, color, cam_idx, cam_origins, pix_idx, hw = build_scene(store, args)
    print(f"[fuse] {coord.shape[0]:,} valid points from {cam_origins.shape[0]} views; "
          f"extent(m)={(coord.max(0) - coord.min(0)).round(3).tolist()}")

    # voxel downsample (reuses the training preprocessing fn). We carry pix_idx through
    # the spare cat_raw slot so the per-point origin pixel survives the downsample
    # exactly like cam_idx (no edit to the shared preprocess_v1_0_1.voxel_first_hit).
    dummy = np.zeros(coord.shape[0], dtype=np.int64)
    coord, color, pix_idx, _a, cam_idx = _voxel_first_hit(
        coord, color, pix_idx, dummy, cam_idx, args.voxel_size
    )
    print(f"[voxel] -> {coord.shape[0]:,} points @ {args.voxel_size*1000:.1f} mm")

    normal = compute_normals(coord, cam_origins, cam_idx, k=args.normal_k)

    out_dir = os.path.join(args.output_root, args.split, args.scene)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "coord.npy"), coord.astype(np.float32))
    np.save(os.path.join(out_dir, "color.npy"), color.astype(np.uint8))
    np.save(os.path.join(out_dir, "normal.npy"), normal.astype(np.float32))
    print(f"[write] {out_dir}  (coord/color/normal; no labels)")
    print(f"        coord {coord.shape} {coord.dtype} | color {color.shape} {color.dtype} "
          f"| normal {normal.shape} {normal.dtype}")
    print(f"        normal |n| mean={np.linalg.norm(normal, axis=1).mean():.3f} "
          f"(should be ~1.0)")

    # --- optional per-point 2D-mask labels -----------------------------------
    if args.coco:
        order = store.get("input_order_list")
        if order is None:
            raise SystemExit("--coco given but the VGGT npz has no 'input_order_list' "
                             "(needed to map cam_idx -> camera/view); cannot label masks")
        input_order_list = [str(v) for v in list(order)]
        if args.mask_all_views:
            # Reproject every fused point into ALL views with z-buffer occlusion.
            aligned, areas, offsets = _prepare_aligned_masks(
                args.coco, args.camera, input_order_list, hw, args
            )
            mask_per_view = _compute_point_mask_labels_allviews(
                store, aligned, areas, offsets, coord, input_order_list, hw, args
            )
            np.save(os.path.join(out_dir, "mask_per_view.npy"), mask_per_view.astype(np.int32))
            # legacy single-view labels = each point's OWN origin-view column (so the
            # old single-view constraint can A/B against the multi-view one).
            rows = np.arange(mask_per_view.shape[0])
            mask_view = cam_idx.astype(np.int32)
            mask_instance = mask_per_view[rows, cam_idx.astype(np.int64)].astype(np.int32)
            np.save(os.path.join(out_dir, "mask_instance.npy"), mask_instance)
            np.save(os.path.join(out_dir, "mask_view.npy"), mask_view)
            labeled_any = int((mask_per_view >= 0).any(axis=1).sum())
            views_per_pt = (mask_per_view >= 0).sum(axis=1)
            n_inst = int(len(np.unique(mask_per_view[mask_per_view >= 0])))
            n_pts = mask_per_view.shape[0]
            print(f"[mask] wrote mask_per_view.npy (N={n_pts:,}, S={mask_per_view.shape[1]}) "
                  f"+ legacy mask_instance/mask_view  "
                  f"({labeled_any:,}/{n_pts:,} pts labeled in >=1 view = "
                  f"{100.0 * labeled_any / max(n_pts, 1):.1f}%, "
                  f"mean {views_per_pt.mean():.2f} labeled views/pt, "
                  f"{n_inst} distinct 2D instances across {len(input_order_list)} views)")
        else:
            mask_instance, mask_view = _compute_point_mask_labels(
                args.coco, args.camera, input_order_list, cam_idx, pix_idx, hw, args
            )
            np.save(os.path.join(out_dir, "mask_instance.npy"), mask_instance.astype(np.int32))
            np.save(os.path.join(out_dir, "mask_view.npy"), mask_view.astype(np.int32))
            labeled = int((mask_instance >= 0).sum())
            n_inst = int(len(np.unique(mask_instance[mask_instance >= 0])))
            print(f"[mask] wrote mask_instance.npy / mask_view.npy  "
                  f"({labeled:,}/{mask_instance.shape[0]:,} pts labeled = "
                  f"{100.0 * labeled / max(mask_instance.shape[0], 1):.1f}%, "
                  f"{n_inst} distinct 2D instances across {len(input_order_list)} views)")


if __name__ == "__main__":
    main()
