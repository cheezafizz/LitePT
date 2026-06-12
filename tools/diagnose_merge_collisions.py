"""Read-only diagnostic for `merge_anns_via_voxel_overlap`.

Walks one or more <scene>/<seq> directories under a v1.0.1 dataset root,
replicates the pre-merge half of `process_sequence`, runs the *current*
merger, and reports:

  * provably-wrong merges: any merge group containing two ann_ids that
    co-occur in the same camera view (per-image rasterization in
    `build_ann_id_map` assigns at most one ann_id per pixel, so this
    cannot be a true multi-view duplicate);
  * shared-voxel overlap fraction across merged pairs, to size a
    cutoff for the planned fix.

Usage:
  python tools/diagnose_merge_collisions.py \
    --dataset-root /path/to/v1.0.1 \
    --scenes 70 71 --limit-seqs 2
"""

import argparse
import glob
import importlib.util
import os
import sys

import numpy as np


def _load_preprocess_module():
    here = os.path.dirname(os.path.abspath(__file__))
    mod_path = os.path.join(
        here, os.pardir, "datasets", "preprocessing", "v1_0_1", "preprocess_v1_0_1.py"
    )
    spec = importlib.util.spec_from_file_location("preprocess_v1_0_1", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pp = _load_preprocess_module()


def build_cloud(seq_dir):
    """Return (coord, cat_raw, ann_raw, cam_idx) or None."""
    cam_md = pp.load_camera_metadata(os.path.join(seq_dir, "camera_metadata.json"))
    import json

    with open(os.path.join(seq_dir, "coco_annotations.json")) as f:
        coco = json.load(f)
    view_by_image_id = {
        int(img["id"]): os.path.splitext(img["file_name"])[0] for img in coco["images"]
    }
    image_id_by_view = {v: i for i, v in view_by_image_id.items()}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    import h5py
    from PIL import Image

    coords, cats, anns, cam_idxs = [], [], [], []
    for ci, view in enumerate(pp.VIEW_NAMES):
        img_path = os.path.join(seq_dir, "images", f"{view}.png")
        dep_path = os.path.join(seq_dir, "depth", f"{view}.hdf5")
        if not (os.path.isfile(img_path) and os.path.isfile(dep_path)):
            continue
        K, Tw2c, _ = cam_md[view]
        rgb = np.array(Image.open(img_path).convert("RGB"))
        with h5py.File(dep_path, "r") as h:
            depth = np.asarray(h["depth"][...], dtype=np.float32)
            cat_map = np.asarray(h["instance_segmap"][...], dtype=np.int64)
        H, W = depth.shape
        image_id = image_id_by_view.get(view)
        if image_id is None:
            ann_map = np.full((H, W), -1, dtype=np.int64)
        else:
            ann_map = pp.build_ann_id_map(anns_by_image.get(image_id, []), H, W)
        res = pp.unproject_view(rgb, depth, cat_map, ann_map, K, Tw2c)
        if res is None:
            continue
        P, _, CAT, ANN = res
        coords.append(P)
        cats.append(CAT)
        anns.append(ANN)
        cam_idxs.append(np.full(P.shape[0], ci, dtype=np.int32))

    if not coords:
        return None
    return (
        np.concatenate(coords, axis=0),
        np.concatenate(cats, axis=0),
        np.concatenate(anns, axis=0),
        np.concatenate(cam_idxs, axis=0),
    )


def voxel_hash(coord, voxel_size):
    keys = np.floor(coord / voxel_size).astype(np.int64)
    p1, p2, p3 = np.int64(73856093), np.int64(19349663), np.int64(83492791)
    return (keys[:, 0] * p1) ^ (keys[:, 1] * p2) ^ (keys[:, 2] * p3)


def diagnose(seq_dir, voxel_size, sample_pairs=10):
    cloud = build_cloud(seq_dir)
    if cloud is None:
        return {"error": "no valid views"}
    coord, cat_raw, ann_raw, cam_idx = cloud

    pre_ann = ann_raw.copy()
    canonical = pp.merge_anns_via_voxel_overlap(
        coord, cat_raw, ann_raw, cam_idx, voxel_size
    )

    valid = pre_ann >= 0
    pre_v = pre_ann[valid]
    can_v = canonical[valid]
    cam_v = cam_idx[valid]
    h_v = voxel_hash(coord[valid], voxel_size)

    pre_unique = np.unique(pre_v)
    n_ann = pre_unique.size

    # canonical -> set of original ann_ids
    groups = {}
    for orig, cano in zip(pre_v, can_v):
        groups.setdefault(int(cano), set()).add(int(orig))

    multi = {c: g for c, g in groups.items() if len(g) > 1}

    # per-ann camera-view set
    ann_to_views = {}
    for a, c in zip(pre_v, cam_v):
        ann_to_views.setdefault(int(a), set()).add(int(c))

    # detect provably-wrong groups
    wrong_groups = []
    for cano, members in multi.items():
        members = sorted(members)
        bad_pair = None
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if ann_to_views[members[i]] & ann_to_views[members[j]]:
                    bad_pair = (members[i], members[j])
                    break
            if bad_pair:
                break
        if bad_pair:
            wrong_groups.append((cano, members, bad_pair))

    # per-ann voxel-count, and per-merged-pair shared-voxel count
    pair_h = np.stack([pre_v, h_v], axis=1)
    uniq_pair = np.unique(pair_h, axis=0)
    voxel_count = {}
    for a in uniq_pair[:, 0]:
        voxel_count[int(a)] = voxel_count.get(int(a), 0) + 1

    # shared voxels per merged pair (only same canonical group, same voxel)
    fractions = []
    for cano, members in multi.items():
        member_set = set(members)
        in_group = np.isin(pre_v, list(member_set))
        if in_group.sum() == 0:
            continue
        sub_ann = pre_v[in_group]
        sub_h = h_v[in_group]
        order = np.argsort(sub_h, kind="stable")
        sa = sub_ann[order]
        sh = sub_h[order]
        diff = np.concatenate(([True], sh[1:] != sh[:-1]))
        starts = np.nonzero(diff)[0]
        ends = np.concatenate((starts[1:], [sh.size]))
        cooccur = {}
        for s, e in zip(starts, ends):
            us = np.unique(sa[s:e])
            for i in range(us.size):
                for j in range(i + 1, us.size):
                    key = (int(us[i]), int(us[j]))
                    cooccur[key] = cooccur.get(key, 0) + 1
        for (a, b), shared in cooccur.items():
            denom = min(voxel_count[a], voxel_count[b])
            if denom > 0:
                fractions.append(shared / denom)

    report = {
        "scene": os.path.relpath(seq_dir),
        "n_valid_anns": int(n_ann),
        "n_merge_groups": int(len(groups)),
        "n_multi_ann_groups": int(len(multi)),
        "n_provably_wrong_groups": int(len(wrong_groups)),
        "wrong_sample": wrong_groups[:sample_pairs],
        "overlap_pctiles": (
            [float(x) for x in np.percentile(fractions, [10, 50, 90])]
            if fractions
            else None
        ),
        "overlap_min_max": (
            (float(min(fractions)), float(max(fractions))) if fractions else None
        ),
        "n_merged_pairs": len(fractions),
    }
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Scene IDs to inspect (e.g. 70 71). Default: first 3.")
    ap.add_argument("--limit-seqs", type=int, default=1,
                    help="Sequences per scene to inspect.")
    ap.add_argument("--voxel-size", type=float, default=0.005)
    args = ap.parse_args()

    if args.scenes:
        scene_dirs = [os.path.join(args.dataset_root, s) for s in args.scenes]
    else:
        scene_dirs = sorted(
            d for d in glob.glob(os.path.join(args.dataset_root, "*"))
            if os.path.isdir(d) and os.path.basename(d).isdigit()
        )[:3]

    for sd in scene_dirs:
        if not os.path.isdir(sd):
            print(f"[skip] {sd} (not a dir)")
            continue
        seqs = sorted(
            s for s in glob.glob(os.path.join(sd, "*"))
            if os.path.isdir(s) and os.path.basename(s).isdigit()
        )[: args.limit_seqs]
        for seq_dir in seqs:
            try:
                rep = diagnose(seq_dir, args.voxel_size)
            except Exception as e:
                print(f"scene {seq_dir}  FAILED ({type(e).__name__}: {e})")
                continue
            print(f"scene {rep['scene']}")
            print(f"  ann_ids: {rep['n_valid_anns']} (valid), "
                  f"merge_groups: {rep['n_merge_groups']}, "
                  f"multi-ann groups: {rep['n_multi_ann_groups']}")
            print(f"  provably-wrong merges (share-view test): "
                  f"{rep['n_provably_wrong_groups']} groups")
            for cano, members, bad in rep["wrong_sample"]:
                print(f"    canonical={cano}  members={members}  "
                      f"share-view pair={bad}")
            if rep["overlap_pctiles"]:
                p10, p50, p90 = rep["overlap_pctiles"]
                lo, hi = rep["overlap_min_max"]
                print(f"  overlap fractions across {rep['n_merged_pairs']} merged "
                      f"pairs: min={lo:.3f} p10={p10:.3f} p50={p50:.3f} "
                      f"p90={p90:.3f} max={hi:.3f}")
            else:
                print("  overlap fractions: (no merged pairs)")
            print()


if __name__ == "__main__":
    main()
