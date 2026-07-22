"""
Preprocess /home/fai/workspace/jhp/dataset/v1.0.1 -> LitePT ScanNet schema.

For each sequence under <dataset_root>/<scene>/<seq>/:
  - load 14 RGB-D views (PNG + HDF5) and camera_metadata.json (w2c extrinsics)
  - unproject each view to world coords, fuse, crop to the WORKING_VOLUME_HALF
    cube centred on origin, voxel-downsample (5 mm default)
  - resolve per-pixel labels to (semantic, instance):
      semantic comes from the HDF5 instance_segmap (which actually stores
      category_id), then mapped via category names to a fixed semantic table
      (floor/machine/board/tray/paper/table/object)
      instance comes from rasterizing per-annotation COCO RLEs and merging
      duplicates across views; every non-stuff class (i.e. outside
      STUFF_CLASS_IDS) gets its own per-scene sequential instance id
  - compute per-point normals via PCA on k-NN, oriented toward the source camera
  - write coord.npy / color.npy / normal.npy / segment20.npy / instance.npy
    to <output_root>/<train|val|test>/scene<SS>_<NNNN>/

Splits: random 80/10/10 by sequence with seed=42.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=DeprecationWarning)


VIEW_NAMES = [
    "TB_left", "TB_right",
    "TF_left", "TF_right",
    "TL_left", "TL_right",
    "TR_left", "TR_right",
    "TC_left", "TC_right",
    "LW_left", "LW_right",
    "RW_left", "RW_right",
]

# COCO category name (lowercased) -> semantic class index
SEMANTIC_NAME_TO_ID = {
    "floor": 0,
    "machine": 1,
    "board": 2,
    "tray": 3,
    "paper": 4,
    "table": 5,
}
OBJECT_CLASS = 6        # catch-all for object-instance assets
# Semantic indices treated as "stuff" — no instance ids assigned. Must stay
# aligned with `segment_ignore_index` in the training config.
STUFF_CLASS_IDS = (0, 1)
IGNORE_INDEX = -1
DEPTH_MAX = 5.0         # safety clip; observed max ~2.5m
# Working-volume AABB: drop world-coord points outside [-HALF, +HALF] on any axis.
WORKING_VOLUME_HALF = 2.0


def load_camera_metadata(path):
    """Return dict view -> (K (3,3) float64, T_w2c (4,4) float64, cam_origin (3,) float32)."""
    with open(path) as f:
        cm = json.load(f)
    out = {}
    for v in VIEW_NAMES:
        entry = cm[v][0]
        K = np.asarray(entry["intrinsics"], dtype=np.float64)
        Tw2c = np.asarray(entry["extrinsics"], dtype=np.float64)
        Tc2w = np.linalg.inv(Tw2c)
        out[v] = (K, Tw2c, Tc2w[:3, 3].astype(np.float32))
    return out


def decode_uncompressed_rle(counts, h, w):
    """COCO uncompressed RLE -> (H, W) uint8 mask. counts is a list of run lengths
    alternating 0/1 starting with 0, indexed column-major (Fortran order)."""
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        flat[pos:pos + c] = val
        pos += c
        val ^= 1
    return flat.reshape((w, h)).T


def build_ann_id_map(annotations_for_image, h, w):
    """Rasterize all annotations for one image into an (H, W) int64 array of
    annotation ids. -1 marks pixels with no annotation. Larger-area annotations
    are painted first so smaller (likely-on-top) masks overwrite them."""
    out = np.full((h, w), -1, dtype=np.int64)
    for ann in sorted(annotations_for_image, key=lambda a: -int(a["area"])):
        seg = ann["segmentation"]
        sh, sw = int(seg["size"][0]), int(seg["size"][1])
        if sh != h or sw != w:
            continue
        m = decode_uncompressed_rle(seg["counts"], sh, sw).astype(bool)
        out[m] = int(ann["id"])
    return out


def unproject_view(rgb, depth, cat_map, ann_map, K, Tw2c):
    """Unproject one valid-pixel set to world. Returns
    (coord, color, cat_raw, ann_raw) or None."""
    valid = np.isfinite(depth) & (depth > 0.0) & (depth < DEPTH_MAX)
    if not valid.any():
        return None
    ys, xs = np.nonzero(valid)
    d = depth[ys, xs].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (xs.astype(np.float64) - cx) / fx * d
    y_cam = (ys.astype(np.float64) - cy) / fy * d
    P_cam = np.stack([x_cam, y_cam, d, np.ones_like(d)], axis=1)  # (P, 4)
    Tc2w = np.linalg.inv(Tw2c)
    P_world = (Tc2w @ P_cam.T).T[:, :3]
    color = rgb[ys, xs]                       # (P, 3) uint8
    cat = cat_map[ys, xs]                     # (P,)   int64
    ann = ann_map[ys, xs]                     # (P,)   int64
    return (
        P_world.astype(np.float32),
        color.astype(np.uint8),
        cat.astype(np.int64),
        ann.astype(np.int64),
    )


def merge_anns_via_voxel_overlap(coord, cat_raw, ann_raw, cam_idx, voxel_size,
                                 *, overlap_threshold=0.3):
    """Cross-view canonicalization: annotation ids whose unprojected points share
    voxels and the same HDF5 category are merged via union-find, so the same 3D
    object viewed from 14 cameras collapses to one id. Two safeguards prevent
    distinct-but-adjacent same-class instances from being fused:
      A. view-disjoint: ann_ids that ever appear together in one camera view
         cannot be duplicates (build_ann_id_map gives one ann_id per pixel).
      B. overlap fraction: shared voxels / min(voxels) must exceed
         overlap_threshold; boundary-touch noise produces tiny fractions,
         true multi-view duplicates are near 1.0."""
    if ann_raw.size == 0:
        return ann_raw.copy()
    obj_mask = ann_raw >= 0
    if obj_mask.sum() < 2:
        return ann_raw.copy()

    keys = np.floor(coord / voxel_size).astype(np.int64)
    p1, p2, p3 = np.int64(73856093), np.int64(19349663), np.int64(83492791)
    h_all = (keys[:, 0] * p1) ^ (keys[:, 1] * p2) ^ (keys[:, 2] * p3)

    h_v = h_all[obj_mask]
    ann_v = ann_raw[obj_mask]
    cat_v = cat_raw[obj_mask]
    cam_v = cam_idx[obj_mask]

    ann_unique, ann_inv = np.unique(ann_v, return_inverse=True)

    # per-ann set of camera views it appears in
    sort_ann = np.argsort(ann_inv, kind="stable")
    inv_s = ann_inv[sort_ann]
    cam_s = cam_v[sort_ann]
    diff_a = np.concatenate(([True], inv_s[1:] != inv_s[:-1]))
    a_starts = np.nonzero(diff_a)[0]
    a_ends = np.concatenate((a_starts[1:], [inv_s.size]))
    ann_views = [None] * ann_unique.size
    for s, e in zip(a_starts, a_ends):
        ann_views[int(inv_s[s])] = set(int(c) for c in np.unique(cam_s[s:e]))

    # per-ann count of unique voxels containing it
    pair = np.stack([ann_inv, h_v], axis=1)
    uniq_pair = np.unique(pair, axis=0)
    voxel_count = np.bincount(uniq_pair[:, 0], minlength=ann_unique.size)

    # pairwise shared-voxel count, restricted to same voxel + same cat
    order = np.argsort(h_v, kind="stable")
    h_so = h_v[order]
    cat_so = cat_v[order]
    inv_so = ann_inv[order]
    diff = np.concatenate(([True], h_so[1:] != h_so[:-1]))
    starts = np.nonzero(diff)[0]
    ends = np.concatenate((starts[1:], [h_so.size]))

    cooccur = {}
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        cats_here = cat_so[s:e]
        invs_here = inv_so[s:e]
        for cat in np.unique(cats_here):
            invs_cat = np.unique(invs_here[cats_here == cat])
            if invs_cat.size < 2:
                continue
            invs_cat = invs_cat.tolist()
            for i in range(len(invs_cat)):
                ai = invs_cat[i]
                for j in range(i + 1, len(invs_cat)):
                    bj = invs_cat[j]
                    key = (ai, bj) if ai < bj else (bj, ai)
                    cooccur[key] = cooccur.get(key, 0) + 1

    parent = {int(a): int(a) for a in ann_unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (i, j), shared in cooccur.items():
        if ann_views[i] & ann_views[j]:
            continue
        denom = min(voxel_count[i], voxel_count[j])
        if denom == 0 or shared / denom < overlap_threshold:
            continue
        union(int(ann_unique[i]), int(ann_unique[j]))

    unique_orig = np.array(sorted(parent.keys()), dtype=np.int64)
    canonical_vals = np.array([find(int(a)) for a in unique_orig], dtype=np.int64)
    canonical = np.copy(ann_raw)
    valid = ann_raw >= 0
    if valid.any():
        idx = np.searchsorted(unique_orig, ann_raw[valid])
        canonical[valid] = canonical_vals[idx]
    return canonical


def voxel_first_hit(coord, color, cat_raw, ann_raw, cam_idx, voxel_size):
    """Keep one point per voxel (first one to appear). Returns downsampled arrays."""
    keys = np.floor(coord / voxel_size).astype(np.int64)
    p1, p2, p3 = np.int64(73856093), np.int64(19349663), np.int64(83492791)
    h = (keys[:, 0] * p1) ^ (keys[:, 1] * p2) ^ (keys[:, 2] * p3)
    order = np.argsort(h, kind="stable")
    h_sorted = h[order]
    diff = np.concatenate(([True], h_sorted[1:] != h_sorted[:-1]))
    first_in_sorted = np.nonzero(diff)[0]
    keep = order[first_in_sorted]
    return (
        coord[keep],
        color[keep],
        cat_raw[keep],
        ann_raw[keep],
        cam_idx[keep],
    )


def compute_normals(coord, cam_origins, cam_idx, k=16):
    """PCA normals on k-NN, oriented toward the source camera."""
    n = coord.shape[0]
    k = min(k, n)
    tree = cKDTree(coord)
    _, nn_idx = tree.query(coord, k=k, workers=1)
    if k == 1:
        return np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
    pts = coord[nn_idx]                                      # (N, k, 3)
    centered = pts - pts.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / float(k)
    # eigh returns eigenvalues ascending; col 0 is smallest -> normal direction
    _, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]
    cams = cam_origins[cam_idx]
    view_dir = cams - coord
    flip = np.einsum("ni,ni->n", normals, view_dir) < 0
    normals[flip] *= -1.0
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return (normals / nrm).astype(np.float32)


def map_labels(cat_raw, ann_raw, name_by_cat):
    """Resolve per-point (cat_raw, canonical ann_raw) -> (segment, instance).

    Semantic comes from the HDF5 category-id segmap (cat_raw).
    Instance comes from the canonical annotation ids in ann_raw (post cross-view
    merging): each unique canonical id whose semantic falls outside
    STUFF_CLASS_IDS gets a sequential per-scene instance index starting at 0.
    Points whose semantic is stuff (floor/machine) or IGNORE_INDEX always get
    IGNORE_INDEX. Same canonical ann_id maps to a single semantic class by
    construction (merge_anns_via_voxel_overlap only unions within a category),
    so keying on ann_raw alone is safe.
    """
    sem_arr = np.full(cat_raw.shape, IGNORE_INDEX, dtype=np.int32)
    for raw in np.unique(cat_raw):
        raw_int = int(raw)
        name = name_by_cat.get(raw_int)
        if name is None:
            sem = IGNORE_INDEX
        else:
            sem = SEMANTIC_NAME_TO_ID.get(name.strip().lower(), OBJECT_CLASS)
        sem_arr[cat_raw == raw_int] = sem

    inst_arr = np.full(ann_raw.shape, IGNORE_INDEX, dtype=np.int32)
    thing_mask = (
        (~np.isin(sem_arr, STUFF_CLASS_IDS))
        & (sem_arr != IGNORE_INDEX)
        & (ann_raw >= 0)
    )
    if thing_mask.any():
        _, inverse = np.unique(ann_raw[thing_mask], return_inverse=True)
        inst_arr[thing_mask] = inverse.astype(np.int32)
    return sem_arr, inst_arr


def process_sequence(task):
    seq_dir, scene_name, split_out_dir, voxel_size, force_instance, crop_min, crop_max = task
    out_dir = os.path.join(split_out_dir, scene_name)
    inst_path = os.path.join(out_dir, "instance.npy")
    if os.path.isfile(inst_path) and not force_instance:
        return scene_name, "skipped (exists)"
    try:
        cam_md = load_camera_metadata(os.path.join(seq_dir, "camera_metadata.json"))
        with open(os.path.join(seq_dir, "coco_annotations.json")) as f:
            coco = json.load(f)
        name_by_cat = {int(c["id"]): str(c["name"]) for c in coco["categories"]}

        view_by_image_id = {
            int(img["id"]): os.path.splitext(img["file_name"])[0]
            for img in coco["images"]
        }
        image_id_by_view = {v: i for i, v in view_by_image_id.items()}
        anns_by_image = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        global_origins = np.stack([cam_md[v][2] for v in VIEW_NAMES], axis=0)  # (14, 3)

        coords, colors, cats, anns, cam_idxs = [], [], [], [], []
        for ci, view in enumerate(VIEW_NAMES):
            img_path = os.path.join(seq_dir, "images", f"{view}.png")
            dep_path = os.path.join(seq_dir, "depth", f"{view}.hdf5")
            if not (os.path.isfile(img_path) and os.path.isfile(dep_path)):
                continue
            K, Tw2c, _ = cam_md[view]
            rgb = np.array(Image.open(img_path).convert("RGB"))
            import h5py  # lazy: only the HDF5 depth path needs it (keeps compute_normals
            # / voxel_first_hit importable in envs without h5py, e.g. the mask build)
            with h5py.File(dep_path, "r") as h:
                depth = np.asarray(h["depth"][...], dtype=np.float32)
                cat_map = np.asarray(h["instance_segmap"][...], dtype=np.int64)
            H, W = depth.shape
            image_id = image_id_by_view.get(view)
            if image_id is None:
                ann_map = np.full((H, W), -1, dtype=np.int64)
            else:
                ann_map = build_ann_id_map(anns_by_image.get(image_id, []), H, W)
            res = unproject_view(rgb, depth, cat_map, ann_map, K, Tw2c)
            if res is None:
                continue
            P, C, CAT, ANN = res
            coords.append(P)
            colors.append(C)
            cats.append(CAT)
            anns.append(ANN)
            cam_idxs.append(np.full(P.shape[0], ci, dtype=np.int32))

        if not coords:
            return scene_name, "FAILED (no valid views)"

        coord = np.concatenate(coords, axis=0)
        color = np.concatenate(colors, axis=0)
        cat_raw = np.concatenate(cats, axis=0)
        ann_raw = np.concatenate(anns, axis=0)
        cam_idx = np.concatenate(cam_idxs, axis=0)

        # Restrict to the working-volume AABB [crop_min, crop_max] (defaults to a
        # cube of side 2 * WORKING_VOLUME_HALF centred on origin). Far-field points
        # add no instance signal and would inflate downstream merge / downsample /
        # normal cost.
        in_box = np.all(
            (coord >= crop_min) & (coord <= crop_max),
            axis=1,
        )
        if not in_box.any():
            return scene_name, "FAILED (no points inside working volume)"
        coord = coord[in_box]
        color = color[in_box]
        cat_raw = cat_raw[in_box]
        ann_raw = ann_raw[in_box]
        cam_idx = cam_idx[in_box]

        ann_raw = merge_anns_via_voxel_overlap(coord, cat_raw, ann_raw, cam_idx, voxel_size)

        coord_d, color_d, cat_raw_d, ann_raw_d, cam_idx_d = voxel_first_hit(
            coord, color, cat_raw, ann_raw, cam_idx, voxel_size
        )

        segment, instance = map_labels(cat_raw_d, ann_raw_d, name_by_cat)

        os.makedirs(out_dir, exist_ok=True)
        if force_instance:
            # Refresh only instance.npy; keep the other arrays bit-exact.
            np.save(inst_path, instance.astype(np.int32))
            return scene_name, f"ok-instance ({coord_d.shape[0]} pts)"

        normals = compute_normals(coord_d, global_origins, cam_idx_d, k=16)
        np.save(os.path.join(out_dir, "coord.npy"), coord_d.astype(np.float32))
        np.save(os.path.join(out_dir, "color.npy"), color_d.astype(np.uint8))
        np.save(os.path.join(out_dir, "normal.npy"), normals.astype(np.float32))
        np.save(os.path.join(out_dir, "segment20.npy"), segment.astype(np.int32))
        np.save(inst_path, instance.astype(np.int32))
        return scene_name, f"ok ({coord_d.shape[0]} pts)"
    except Exception as e:
        return scene_name, f"FAILED ({type(e).__name__}: {e})"


def enumerate_sequences(dataset_root):
    seqs = []
    for scene_dir in sorted(glob.glob(os.path.join(dataset_root, "*"))):
        if not os.path.isdir(scene_dir):
            continue
        scene_name = os.path.basename(scene_dir)
        if not scene_name.isdigit():
            continue
        for seq in sorted(glob.glob(os.path.join(scene_dir, "*"))):
            if not os.path.isdir(seq):
                continue
            seq_id = os.path.basename(seq)
            if not seq_id.isdigit():
                continue
            ok = (
                os.path.isfile(os.path.join(seq, "camera_metadata.json"))
                and os.path.isfile(os.path.join(seq, "coco_annotations.json"))
                and os.path.isdir(os.path.join(seq, "depth"))
                and os.path.isdir(os.path.join(seq, "images"))
            )
            if ok:
                seqs.append((seq, int(scene_name), int(seq_id)))
    return seqs


def assign_splits(sequences, seed=42, frac=(0.8, 0.1, 0.1)):
    n = len(sequences)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * frac[0]))
    n_val = int(round(n * frac[1]))
    split_for = np.empty(n, dtype=object)
    split_for[perm[:n_train]] = "train"
    split_for[perm[n_train:n_train + n_val]] = "val"
    split_for[perm[n_train + n_val:]] = "test"
    return [(sd, scn, seq, split_for[i]) for i, (sd, scn, seq) in enumerate(sequences)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--crop_min", default=None,
        help="Comma-separated 'x,y,z' min corner of the working-volume AABB (metres). "
             "Default None => symmetric [-WORKING_VOLUME_HALF, +WORKING_VOLUME_HALF] cube.")
    parser.add_argument(
        "--crop_max", default=None,
        help="Comma-separated 'x,y,z' max corner of the working-volume AABB (metres). "
             "Default None => symmetric [-WORKING_VOLUME_HALF, +WORKING_VOLUME_HALF] cube.")
    parser.add_argument("--limit", type=int, default=0,
                        help="If >0, only process the first N sequences (post-split).")
    parser.add_argument("--force-instance", action="store_true",
                        help="Recompute and overwrite instance.npy in-place even if "
                             "it already exists. Leaves coord/color/normal/segment20 "
                             "bit-exact.")
    parser.add_argument("--scene-filter", default="",
                        help="Comma-separated 'scene_id:seq_id' pairs (e.g. '70:1001') "
                             "to limit processing to specific sequences. Empty = all.")
    args = parser.parse_args()

    def _parse_corner(s, default):
        if s is None:
            return default
        vals = [float(t) for t in s.split(",")]
        if len(vals) != 3:
            raise ValueError(f"expected 'x,y,z', got {s!r}")
        return np.asarray(vals, dtype=np.float64)

    crop_min = _parse_corner(args.crop_min, np.full(3, -WORKING_VOLUME_HALF))
    crop_max = _parse_corner(args.crop_max, np.full(3, WORKING_VOLUME_HALF))
    if np.any(crop_min >= crop_max):
        raise ValueError(f"crop_min {crop_min} must be < crop_max {crop_max} on every axis")

    print(f"Enumerating sequences under {args.dataset_root} ...")
    sequences = enumerate_sequences(args.dataset_root)
    print(f"  found {len(sequences)} sequences")
    swsp = assign_splits(sequences, seed=args.seed)
    n_train = sum(1 for _, _, _, s in swsp if s == "train")
    n_val = sum(1 for _, _, _, s in swsp if s == "val")
    n_test = sum(1 for _, _, _, s in swsp if s == "test")
    print(f"  splits: train={n_train} val={n_val} test={n_test}")

    if args.scene_filter:
        wanted = set()
        for tok in args.scene_filter.split(","):
            tok = tok.strip()
            if not tok:
                continue
            scn_s, seq_s = tok.split(":")
            wanted.add((int(scn_s), int(seq_s)))
        swsp = [t for t in swsp if (t[1], t[2]) in wanted]
        print(f"  --scene-filter set: processing only {len(swsp)} sequences")

    if args.limit > 0:
        swsp = swsp[: args.limit]
        print(f"  --limit set: processing only {len(swsp)} sequences")

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.output_root, split), exist_ok=True)

    tasks = []
    for sd, scn, seq, split in swsp:
        scene_name = f"scene{scn:02d}_{seq:04d}"
        tasks.append((
            sd, scene_name, os.path.join(args.output_root, split),
            args.voxel_size, args.force_instance, crop_min, crop_max,
        ))

    print(f"Processing {len(tasks)} sequences with {args.num_workers} workers, "
          f"voxel={args.voxel_size} m, crop=[{crop_min.tolist()}, {crop_max.tolist()}] m ...")
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        for i, (name, status) in enumerate(pool.map(process_sequence, tasks), 1):
            if status.startswith("FAILED"):
                n_fail += 1
                print(f"  [{i}/{len(tasks)}] {name}: {status}")
            else:
                n_ok += 1
                if i % 50 == 0:
                    print(f"  [{i}/{len(tasks)}] {name}: {status}")
    print(f"Done. ok={n_ok}, fail={n_fail}")


if __name__ == "__main__":
    main()
