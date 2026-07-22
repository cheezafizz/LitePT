"""
Apply EACH inference-time clustering config to EACH already-labeled val_sample
scene, writing one predicted-instance GLB (+ viser .npz) per (scene x config).

This is the "cluster on the GIVEN labels" path: the per-point SEMANTIC class comes
from the 2D semantic logits baked into each scene's data_seg.npz (`sem_seg`,
(S,H,W,7) over background/machine/board/tray/paper/object/oob), NOT from LitePT's
semantic head. The embed checkpoint still supplies the per-point CENTROID OFFSET and
the INSTANCE EMBEDDING (so touching objects separate and the embed* split works),
and the D-FINE-seg 2D instance masks (from <aligned_root>/<scene>/coco_annotations.json,
pre-aligned to the VGGT grid) drive the embed-maskclust* cannot-link / filter.

No model change is needed: model._cluster() derives the semantic class from
argmax(softmax(logit_pred)), so we synthesize a one-hot logit_pred from the given
per-point class. segment_ignore_index=(-1,0,1) then ignores background+machine.

The backbone runs ONCE per scene; all clustering configs are swept on the cached
per-point predictions (same pattern as tools/reeval_insseg_cluster_sweep.py). It
composes existing helpers from tools/vggt_to_scene.py, tools/infer_insseg.py and
engines/hooks/insseg_viz.py rather than re-implementing fuse / cluster / viz.

Run with the litept env, fp32 (bf16 breaks the spconv autotuner in eval):
  /home/fai/miniconda3/envs/litept/bin/python tools/cluster_labeled_scenes.py
  # optional: --scenes bemealbakeryyedang/007098 niseko/023349
  #           --configs geometric embed-maskclust-strong-filter
  #           --zero-offset            (A/B: pure-geometric on the given labels)
"""
import argparse
import glob
import os
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
for _p in (_REPO_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pointops  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.env import set_seed  # noqa: E402
from datasets.utils import collate_fn  # noqa: E402
from models import build_model  # noqa: E402

# Reused as-is (no edits): scene fusion + 2D-mask labeling, the inference transform /
# cluster-config applier, and the GLB/npz writers.
import vggt_to_scene as v2s  # noqa: E402
import infer_insseg as ifs  # noqa: E402
from engines.hooks.insseg_viz import (  # noqa: E402
    assign_instance_ids,
    colorize_predicted_instances,
    instance_palette,
    save_pointcloud_glb,
    voxel_downsample_indices,
)

# The given-label scheme (data_seg.npz `sem_seg` channel order). NOT the model's
# floor/.../object names -- we replace the semantic head with these labels.
LABEL_NAMES = ["background", "machine", "board", "tray", "paper", "object", "oob"]

_DEFAULT_EXP = os.path.join(
    _REPO_ROOT, "exp", "scannet-v1.1.1-2of3", "insseg-litept-small-v1m2-2of3-embed"
)
_DEFAULT_ROOT = "/home/fai/workspace/jhp/dataset/val_sample_output"
_DEFAULT_ALIGNED = "/home/fai/workspace/jhp/dataset/val_sample_output_aligned"
_DEFAULT_CLUSTERING_DIR = os.path.join(
    _REPO_ROOT, "configs", "scannet-v1.1.1-2of3", "clustering"
)


def fuse_args(voxel_size, conf_percentile=0.0, conf_threshold=0.0,
              valid_mask="filtered"):
    """A single SimpleNamespace carrying every attribute the reused vggt_to_scene
    helpers read (build_scene + the mask labelers). Matches the vggt_to_scene
    defaults for the VGGT + pre-aligned-coco path. conf_percentile/conf_threshold
    add a confidence floor at fusion time (build_scene drops points below it, within
    the finite & valid_mask region); valid_mask selects the stored VGGT mask."""
    return SimpleNamespace(
        # build_scene
        image_key=None, extrinsic_key=None, points_key=None, depth_key=None,
        intrinsic_key=None, conf_key=None, extrinsic_c2w=False,
        conf_threshold=conf_threshold, conf_percentile=conf_percentile, depth_max=10.0,
        valid_mask=valid_mask, valid_mask_key=None, crop_half=0.0,
        voxel_size=voxel_size,
        # mask labelers (coco already on the VGGT grid -> pre-aligned, no camera.json)
        mask_pre_aligned=True, mask_original_size=[518, 392],
        mask_resize_mode="padding", mask_undistort=True, mask_center_pp=True,
        mask_occlusion_tol=0.005, load_fn=v2s._VISIONSCO_LOAD_FN,
    )


def prepare_pre_aligned_masks(coco_path, input_order_list, hw):
    """Pre-aligned variant of vggt_to_scene._prepare_aligned_masks.

    The aligned coco masks are ALREADY on the VGGT (H, W) grid, so index them
    directly -- no preprocess_view, no camera.json. Inlined here to avoid
    vggt_to_scene's unconditional VisionSCO load_fn import (which needs cv2, absent
    in the litept env) on the pre-aligned path where it is never used.

    Returns (aligned, areas, offsets) exactly like the upstream helper.
    """
    H, W = int(hw[0]), int(hw[1])
    per_view_raw = v2s._load_coco_per_instance(coco_path, input_order_list, [W, H])
    offsets = np.cumsum([0] + [m.shape[0] for m in per_view_raw]).astype(np.int64)
    aligned, areas = [], []
    for raw in per_view_raw:
        om = (raw > 0).astype(bool)  # (S_i, H0, W0)
        if om.shape[0] == 0:
            aligned.append(np.zeros((0, H, W), dtype=bool))
            areas.append(np.zeros(0, dtype=np.int64))
            continue
        if om.shape[1:] != (H, W):
            raise SystemExit(
                f"pre-aligned mask grid {tuple(om.shape[1:])} != world_points grid "
                f"{(H, W)}")
        aligned.append(om)
        areas.append(om.reshape(om.shape[0], -1).sum(1).astype(np.int64))
    return aligned, areas, offsets


def discover_scenes(root):
    """All <group>/<seq> dirs under root that contain a data_seg.npz, sorted."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "data_seg.npz"))):
        seq_dir = os.path.dirname(p)
        out.append(os.path.relpath(seq_dir, root))
    return out


def load_clustering_configs(clustering_dir, only=None):
    """Return [(name, path)] for clustering/*.py, optionally filtered to `only`."""
    paths = sorted(glob.glob(os.path.join(clustering_dir, "*.py")))
    cfgs = [(os.path.splitext(os.path.basename(p))[0], p) for p in paths]
    if only:
        want = set(only)
        cfgs = [(n, p) for (n, p) in cfgs if n in want]
        missing = want - {n for (n, _) in cfgs}
        if missing:
            raise SystemExit(f"--configs not found in {clustering_dir}: {sorted(missing)}")
    return cfgs


def build_labeled_scene(seq_dir, aligned_seq_dir, fa):
    """Fuse a data_seg.npz scene into a labeled LitePT cloud.

    Returns dict(coord, color, normal, segment, mask_per_view, mask_instance,
    mask_view) at the 2 mm voxel resolution. segment is the per-point class from
    argmax(sem_seg); mask_* are the D-FINE-seg 2D instance labels (or None if the
    aligned coco is absent). All arrays are index-aligned to coord.
    """
    store = dict(np.load(os.path.join(seq_dir, "data_seg.npz"), allow_pickle=True))
    if "sem_seg" not in store:
        raise SystemExit(f"{seq_dir}/data_seg.npz has no 'sem_seg'")

    # Fuse views -> point cloud (reuses the exact training validity/conf selection).
    coord, color, cam_idx, cam_origins, pix_idx, (H, W) = v2s.build_scene(store, fa)

    # Per-point GIVEN semantic class: argmax over the 7 sem_seg channels at each
    # point's origin (view, pixel). build_scene exposes both (cam_idx + flat pix_idx).
    sem_cls = np.asarray(store["sem_seg"]).argmax(-1).astype(np.int32)  # (S,H,W)
    hh = (pix_idx // W).astype(np.int64)
    ww = (pix_idx % W).astype(np.int64)
    seg_pt = sem_cls[cam_idx.astype(np.int64), hh, ww].astype(np.int32)

    # Voxel-downsample at 2 mm, carrying seg_pt + pix_idx + cam_idx through the three
    # spare slots of voxel_first_hit (same trick as vggt_to_scene.main carries pix_idx).
    compute_normals, voxel_first_hit = v2s._load_preprocess_helpers()
    coord, color, seg_pt, pix_idx, cam_idx = voxel_first_hit(
        coord, color, seg_pt, pix_idx, cam_idx, fa.voxel_size
    )
    normal = compute_normals(coord, cam_origins, cam_idx, k=16)

    out = dict(
        coord=coord.astype(np.float32), color=color.astype(np.uint8),
        normal=normal.astype(np.float32), segment=seg_pt.astype(np.int64),
        mask_per_view=None, mask_instance=None, mask_view=None,
    )

    # D-FINE-seg 2D instance masks (for embed-maskclust*). The aligned coco is on the
    # VGGT grid (pre-aligned). Multi-view reprojection needs intrinsic_aligned +
    # extrinsic_pnp, which live in the sibling data.npz, not data_seg.npz.
    coco_path = os.path.join(aligned_seq_dir, "coco_annotations.json")
    if not os.path.isfile(coco_path):
        print(f"  [mask] no aligned coco at {coco_path}; maskclust configs degrade")
        return out

    cam_npz = os.path.join(seq_dir, "data.npz")
    cams = np.load(cam_npz, allow_pickle=True) if os.path.isfile(cam_npz) else {}
    if "intrinsic_aligned" in store and "extrinsic_pnp" in store:
        store2 = store
    elif "intrinsic_aligned" in cams and "extrinsic_pnp" in cams:
        store2 = dict(store)
        store2["intrinsic_aligned"] = cams["intrinsic_aligned"]
        store2["extrinsic_pnp"] = cams["extrinsic_pnp"]
    else:
        print(f"  [mask] no aligned camera (intrinsic_aligned/extrinsic_pnp) for "
              f"{seq_dir}; maskclust configs degrade")
        return out

    order = store.get("input_order_list")
    if order is None:
        print(f"  [mask] no input_order_list; maskclust configs degrade")
        return out
    input_order_list = [str(v) for v in list(order)]

    aligned, areas, offsets = prepare_pre_aligned_masks(
        coco_path, input_order_list, (H, W)
    )
    mask_per_view = v2s._compute_point_mask_labels_allviews(
        store2, aligned, areas, offsets, coord, input_order_list, (H, W), fa
    )  # (N, S) int32
    rows = np.arange(mask_per_view.shape[0])
    out["mask_per_view"] = mask_per_view.astype(np.int64)
    out["mask_view"] = cam_idx.astype(np.int64)
    out["mask_instance"] = mask_per_view[rows, cam_idx.astype(np.int64)].astype(np.int64)
    n_lab = int((mask_per_view >= 0).any(1).sum())
    n_inst = int(len(np.unique(mask_per_view[mask_per_view >= 0])))
    print(f"  [mask] {n_lab:,}/{mask_per_view.shape[0]:,} pts in >=1 of "
          f"{mask_per_view.shape[1]} views, {n_inst} distinct 2D instances")
    return out


def run_backbone(model, scene, transform, zero_offset, semantic_source="labels"):
    """Run the backbone once. Returns the grid-resolution tensors needed by every
    clustering config: coord/offset, network bias (offset) + embedding, the semantic
    logits, and the mask labels remapped to the grid. Also returns origin_coord +
    grid->origin index for remapping masks back out.

    semantic_source selects what drives _cluster's argmax semantics:
      "labels" -> one-hot of the GIVEN per-point class (data_seg.npz sem_seg).
      "model"  -> the embed model's own seg_head prediction (GT-free inference)."""
    data_dict = dict(coord=scene["coord"].copy(), color=scene["color"].copy(),
                     normal=scene["normal"].copy(), name="scene")
    input_dict = collate_fn([transform(data_dict)])
    for k in list(input_dict.keys()):
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = input_dict[k].cuda(non_blocking=True)

    # For each GRID point, the nearest ORIGIN (fused 2mm) point -> remap labels in.
    g2o, _ = pointops.knn_query(
        1, input_dict["origin_coord"].float(), input_dict["origin_offset"].int(),
        input_dict["coord"].float(), input_dict["offset"].int())
    g2o = g2o.cpu().flatten().long()

    n_grid = int(input_dict["coord"].shape[0])
    with torch.no_grad():
        feat = model._backbone_feat(input_dict)
        bias = model.bias_head(feat)
        if zero_offset:
            bias = torch.zeros_like(bias)
        embedding = model.embedding_head(feat) if getattr(
            model, "embedding_head", None) is not None else None
        model_logit = model.seg_head(feat)  # (n_grid, n_cls) predicted semantics

    if semantic_source == "model":
        # Full GT-free inference: cluster on the model's OWN predicted semantics.
        logit = model_logit
    else:
        # Given per-point class -> grid resolution -> one-hot logit (so _cluster's
        # argmax(softmax(logit)) == the given label; score ~1).
        seg_grid = torch.from_numpy(scene["segment"])[g2o].to(feat.device)  # (n_grid,)
        n_cls = model.seg_head.out_features
        logit = torch.full((n_grid, n_cls), -10.0, device=feat.device, dtype=feat.dtype)
        logit.scatter_(1, seg_grid.clamp(min=0).long().unsqueeze(1), 20.0)

    def _remap_np(arr):  # (N_origin[,S]) numpy -> grid resolution tensor
        return torch.from_numpy(arr)[g2o] if arr is not None else None

    return dict(
        input_dict=input_dict, g2o=g2o,
        coord=input_dict["coord"], offset=input_dict["offset"],
        bias=bias, embedding=embedding, logit=logit,
        # grid-resolution semantic class actually used by _cluster (argmax of the logit):
        # the given label for the labels arm, the model's prediction for the model arm.
        seg_grid=logit.argmax(1).detach().cpu(),
        point_mask_views=_remap_np(scene["mask_per_view"]),
        point_mask=_remap_np(scene["mask_instance"]),
        point_view=_remap_np(scene["mask_view"]),
    )


def cluster_one(model, cached):
    """Run model._cluster for the model's CURRENT clustering attributes."""
    pmv = cached["point_mask_views"]
    pm = cached["point_mask"]
    pv = cached["point_view"]
    dev = cached["coord"].device
    with torch.no_grad():
        out = model._cluster(
            cached["coord"], cached["bias"], cached["logit"], cached["offset"],
            embedding=cached["embedding"] if getattr(model, "instance_embedding", False) else None,
            point_view=pv.to(dev) if pv is not None else None,
            point_mask=pm.to(dev) if pm is not None else None,
            point_mask_views=pmv.to(dev) if pmv is not None else None,
        )
    return out


def write_outputs(out, cached, scene, out_glb, out_npz, scene_name, config_name,
                  class_names, filter_on=False, origin_has_label=None):
    """Remap grid-resolution proposals back to the full fused cloud and write a
    predicted-instance GLB + structured npz (mirrors tools/infer_insseg.py).

    `class_names` names the pred_classes scheme (the given LABEL_NAMES for the labels
    arm, the model's cfg.data.names for the model arm). The GIVEN per-point semantic
    label (`scene["segment"]`, always in the LABEL_NAMES scheme) is also baked in as
    `segment`/`segment_names` for the viewer's "Semantic (labels)" view.

    For a cluster_mask_filter config (filter_on) the unmasked points are dropped
    from the OUTPUT cloud too -- only masked foreground remains -- so the filtered
    viz is visibly distinct from its non-filtered sibling (matches infer_insseg)."""
    input_dict = cached["input_dict"]
    origin_coord = input_dict["origin_coord"]
    o2g, _ = pointops.knn_query(
        1, input_dict["coord"].float(), input_dict["offset"].int(),
        origin_coord.float(), input_dict["origin_offset"].int())
    o2g = o2g.cpu().flatten().long()
    origin_coord_np = origin_coord.cpu().numpy()

    pred_masks = out["pred_masks"]
    pred_scores = out["pred_scores"]
    pred_classes = out["pred_classes"]
    n_pred = int(pred_masks.shape[0])
    rgb_full = np.clip(scene["color"], 0, 255).astype(np.uint8)
    # per-point semantic class ACTUALLY used for clustering, remapped grid -> origin
    # (given labels for the labels arm, model prediction for the model arm).
    seg_full = cached["seg_grid"][o2g].numpy().astype(np.int32)
    masks_origin = (pred_masks[:, o2g].bool().numpy() if n_pred > 0
                    else np.zeros((0, origin_coord_np.shape[0]), dtype=bool))

    # Drop unmasked points from the output for a mask-filter config.
    if filter_on and origin_has_label is not None and \
            origin_has_label.shape[0] == origin_coord_np.shape[0]:
        keep = origin_has_label
        origin_coord_np = origin_coord_np[keep]
        masks_origin = masks_origin[:, keep]
        rgb_full = rgb_full[keep] if rgb_full.shape[0] == keep.shape[0] else rgb_full
        seg_full = seg_full[keep] if seg_full.shape[0] == keep.shape[0] else seg_full

    n_pts = origin_coord_np.shape[0]
    pred_scores_np = pred_scores.numpy() if n_pred > 0 else np.zeros(0, np.float32)
    pred_classes_np = (pred_classes.numpy() if n_pred > 0
                       else np.zeros(0, np.int32)).astype(np.int32)

    colors = colorize_predicted_instances(n_pts, masks_origin, pred_scores_np)
    save_pointcloud_glb(out_glb, origin_coord_np, colors)

    inst_id = assign_instance_ids(n_pts, masks_origin, pred_scores_np)
    inst_color = instance_palette(n_pred)
    rgb = rgb_full if rgb_full.shape[0] == n_pts else np.full((n_pts, 3), 160, np.uint8)
    segment = seg_full if seg_full.shape[0] == n_pts else np.full((n_pts,), -1, np.int32)
    np.savez_compressed(
        out_npz, coord=origin_coord_np.astype(np.float32), rgb=rgb,
        inst_id=inst_id, inst_class=pred_classes_np,
        inst_score=pred_scores_np.astype(np.float32), inst_color=inst_color,
        class_names=np.asarray([str(x) for x in class_names]),
        segment=segment, segment_names=np.asarray([str(x) for x in class_names]),
        scene=scene_name, model=config_name,
    )
    cn = list(class_names)
    by_cls = {cn[c] if 0 <= c < len(cn) else str(c): int((pred_classes_np == c).sum())
              for c in np.unique(pred_classes_np)} if n_pred else {}
    return n_pred, by_cls


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=_DEFAULT_ROOT)
    ap.add_argument("--aligned-root", default=_DEFAULT_ALIGNED)
    ap.add_argument("--config-file", default=os.path.join(_DEFAULT_EXP, "config.py"))
    ap.add_argument("--weight", default=os.path.join(_DEFAULT_EXP, "model", "model_best.pth"))
    ap.add_argument("--clustering-dir", default=_DEFAULT_CLUSTERING_DIR)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="subset of '<group>/<seq>' (default: all under --root)")
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset of clustering config names (default: all *.py)")
    ap.add_argument("--zero-offset", action="store_true",
                    help="A/B: zero the network centroid offsets (pure-geometric "
                         "grouping on the given labels)")
    ap.add_argument("--semantic-source", choices=["labels", "model"], default="labels",
                    help="semantics driving clustering: 'labels' = the given sem_seg "
                         "(one-hot), 'model' = the embed model's own seg_head prediction")
    ap.add_argument("--out-suffix", default="",
                    help="appended to each output filename + npz 'model' field "
                         "(e.g. '-modelsem' to sit beside the labels arm in the viewer)")
    ap.add_argument("--out-subdir", default="insseg",
                    help="per-scene output subdir (default 'insseg'); use a distinct "
                         "name (e.g. 'insseg_conf30') to keep a thresholded run separate")
    ap.add_argument("--conf-percentile", type=float, default=0.0,
                    help="confidence percentile floor (0-100) at fusion, computed within "
                         "the finite & valid_mask region; 0 disables (e.g. 30 drops the "
                         "lowest-confidence ~30%% of points)")
    ap.add_argument("--conf-threshold", type=float, default=0.0,
                    help="absolute confidence floor at fusion; points below are dropped")
    ap.add_argument("--valid-mask", choices=["filtered", "valid", "none"],
                    default="filtered",
                    help="stored VGGT validity mask AND-ed into per-view validity "
                         "(default filtered = filtered_valid_mask)")
    ap.add_argument("--rgb-voxel", type=float, default=0.005,
                    help="voxel (m) for the per-scene RGB reference GLB (<=0 = full-res)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    model = build_model(cfg.model)
    model = ifs.load_weight(model, args.weight)
    model = model.cuda().eval()  # fp32; bf16 breaks the spconv autotuner in eval
    if getattr(model, "embedding_head", None) is None:
        raise SystemExit("checkpoint has no embedding head; use the embed exp "
                         "(insseg-litept-small-v1m2-2of3-embed) for the embed* configs")
    base_voxel = float(model.voxel_size)
    transform = ifs.build_infer_transform(grid_size=base_voxel)
    fa = fuse_args(base_voxel, conf_percentile=args.conf_percentile,
                   conf_threshold=args.conf_threshold, valid_mask=args.valid_mask)

    # Snapshot the clustering attributes so each config starts from the SAME baseline.
    # apply_cluster_config only sets keys a config mentions, so without this reset a
    # toggle from one config (e.g. cluster_mask_filter) would leak into the next.
    cluster_baseline = {k: getattr(model, k) for k in ifs.CLUSTER_KEYS
                        if hasattr(model, k)}

    # pred_classes scheme: the model arm returns the model's classes; the labels arm
    # returns the given sem_seg classes. (segment/segment_names are always LABEL_NAMES.)
    class_names = (list(cfg.data.names) if args.semantic_source == "model"
                   else LABEL_NAMES)

    scenes = args.scenes or discover_scenes(args.root)
    configs = load_clustering_configs(args.clustering_dir, args.configs)
    print(f"[run] {len(scenes)} scenes x {len(configs)} configs "
          f"(semantic_source={args.semantic_source} suffix={args.out_suffix!r} "
          f"zero_offset={args.zero_offset} | valid_mask={args.valid_mask} "
          f"conf_pct={args.conf_percentile} conf_thr={args.conf_threshold} "
          f"-> {args.out_subdir}/)")
    print(f"[run] configs: {[n for n, _ in configs]}")

    n_ok = n_fail = 0
    for rel in scenes:
        seq_dir = os.path.join(args.root, rel)
        aligned_seq_dir = os.path.join(args.aligned_root, rel)
        scene_name = rel.replace("/", "__")
        out_root = os.path.join(seq_dir, args.out_subdir)
        os.makedirs(out_root, exist_ok=True)
        print(f"\n=== {rel} ===")
        try:
            scene = build_labeled_scene(seq_dir, aligned_seq_dir, fa)
            uc, cc = np.unique(scene["segment"], return_counts=True)
            print(f"  [fuse] {scene['coord'].shape[0]:,} pts @ {base_voxel*1000:.0f}mm | "
                  f"classes " + ", ".join(
                      f"{LABEL_NAMES[c]}={n}" for c, n in zip(uc, cc)))

            # RGB reference GLB (once per scene).
            rgb_glb = os.path.join(out_root, "_rgb.glb")
            kept = (voxel_downsample_indices(scene["coord"], args.rgb_voxel)
                    if args.rgb_voxel > 0 else np.arange(scene["coord"].shape[0]))
            save_pointcloud_glb(rgb_glb, scene["coord"][kept],
                                scene["color"][kept].astype(np.uint8))

            cached = run_backbone(model, scene, transform, args.zero_offset,
                                  semantic_source=args.semantic_source)

            # origin-resolution "point falls in >=1 2D mask" (for the filter output).
            mpv = scene["mask_per_view"]
            origin_has_label = ((mpv >= 0).any(1) if mpv is not None else None)

            for name, path in configs:
                for k, v in cluster_baseline.items():
                    setattr(model, k, v)
                ifs.apply_cluster_config(model, path)
                out = cluster_one(model, cached)
                tag = f"{name}{args.out_suffix}"
                out_glb = os.path.join(out_root, f"{tag}.glb")
                out_npz = os.path.join(out_root, f"{tag}.npz")
                n_pred, by_cls = write_outputs(
                    out, cached, scene, out_glb, out_npz, scene_name, tag, class_names,
                    filter_on=bool(getattr(model, "cluster_mask_filter", False)),
                    origin_has_label=origin_has_label)
                print(f"  [{tag:40s}] {n_pred:3d} proposals  {by_cls}")
            n_ok += 1
        except Exception as e:  # noqa: BLE001 -- keep the batch going
            n_fail += 1
            print(f"  [ERROR] {rel}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\n[done] {n_ok} scenes ok, {n_fail} failed; "
          f"outputs under <scene>/insseg/<config>.glb (+ .npz)")


if __name__ == "__main__":
    main()
