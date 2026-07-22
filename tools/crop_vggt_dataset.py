"""
Crop a VGGT InsSeg dataset (coord/color/normal scenes) to a fixed world-frame AABB.

The InsSeg model trains on `data/scannet-v1.1.1-2of3`, whose GT scenes were clamped
to the working-volume AABB x[-0.4, 0.6], y[-0.5, 0.7], z[-0.2, 0.7] (see
`logs/reprocess_2mm.sh`). VGGT scenes built by `tools/vggt_to_scene.py` are NOT cropped
to that box and carry ~26-36% far-field points the model never saw. This applies the
identical AABB so the inference inputs match the trained extent.

This is a pure post-hoc point mask: it reuses the exact AABB predicate from
`datasets/preprocessing/v1_0_1/preprocess_v1_0_1.py` (crop step) and keeps the stored
normals (recomputing oriented normals would need per-view camera origins, which the
built scenes don't persist; dropped points are >0.5 m away, far outside any kept point's
16-NN, so the stored normals stay valid). Re-running is idempotent.

Example:
  /home/fai/miniconda3/envs/litept/bin/python tools/crop_vggt_dataset.py \
      --data-root data/scannet-v1.1.1-2of3-vggt --splits test \
      --crop-min -0.4,-0.5,-0.2 --crop-max 0.6,0.7,0.7
"""
import argparse
import glob
import os

import numpy as np

ARRAYS = ("coord.npy", "color.npy", "normal.npy")


def _parse_corner(s):
    """Parse a comma-separated 'x,y,z' string into a (3,) float64 array."""
    vals = [float(t) for t in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"expected 'x,y,z', got {s!r}")
    return np.asarray(vals, dtype=np.float64)


def _atomic_save(path, arr):
    """np.save to a sibling .tmp then os.replace -> crash-safe in-place overwrite."""
    tmp = path + ".tmp"
    np.save(tmp, arr)
    # np.save appends .npy to a path without that suffix; account for it.
    written = tmp if tmp.endswith(".npy") else tmp + ".npy"
    os.replace(written, path)


def crop_scene(scene_dir, out_dir, crop_min, crop_max, dry_run):
    coord = np.load(os.path.join(scene_dir, "coord.npy"))
    n = coord.shape[0]
    mask = np.all((coord >= crop_min) & (coord <= crop_max), axis=1)
    kept = int(mask.sum())
    pct = 100.0 * kept / n if n else 0.0
    name = os.path.basename(os.path.normpath(scene_dir))
    if kept == 0:
        print(f"  WARN {name}: 0/{n} points in AABB -> left untouched")
        return n, n  # report as unchanged
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        for fname in ARRAYS:
            arr = np.load(os.path.join(scene_dir, fname))
            _atomic_save(os.path.join(out_dir, fname), arr[mask])
    print(f"  {name}: {n} -> {kept} ({pct:.1f}% kept, {n - kept} dropped)")
    return n, kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/scannet-v1.1.1-2of3-vggt")
    ap.add_argument("--splits", default="test",
                    help="comma-separated split subdirs to process (default: test)")
    ap.add_argument("--crop-min", default="-0.4,-0.5,-0.2",
                    help="comma-separated 'x,y,z' min corner (m)")
    ap.add_argument("--crop-max", default="0.6,0.7,0.7",
                    help="comma-separated 'x,y,z' max corner (m)")
    ap.add_argument("--output-root", default=None,
                    help="output dataset root; default = --data-root (in-place overwrite)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print kept counts, write nothing")
    args = ap.parse_args()

    crop_min = _parse_corner(args.crop_min)
    crop_max = _parse_corner(args.crop_max)
    if np.any(crop_min >= crop_max):
        raise ValueError(f"crop_min {crop_min} must be < crop_max {crop_max} on every axis")
    out_root = args.output_root or args.data_root
    in_place = os.path.abspath(out_root) == os.path.abspath(args.data_root)

    print(f"crop AABB min={crop_min.tolist()} max={crop_max.tolist()} "
          f"{'(DRY-RUN) ' if args.dry_run else ''}"
          f"{'in-place' if in_place else f'-> {out_root}'}")

    tot_n = tot_kept = n_scenes = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        split_dir = os.path.join(args.data_root, split)
        if not os.path.isdir(split_dir):
            print(f"split '{split}': not found, skipping")
            continue
        scenes = sorted(
            d for d in glob.glob(os.path.join(split_dir, "*"))
            if os.path.isfile(os.path.join(d, "coord.npy"))
        )
        print(f"split '{split}': {len(scenes)} scenes")
        for scene_dir in scenes:
            out_dir = scene_dir if in_place else os.path.join(
                out_root, split, os.path.basename(os.path.normpath(scene_dir)))
            n, kept = crop_scene(scene_dir, out_dir, crop_min, crop_max, args.dry_run)
            tot_n += n
            tot_kept += kept
            n_scenes += 1

    if tot_n:
        print(f"Done. {n_scenes} scenes: {tot_n} -> {tot_kept} points "
              f"({100.0 * tot_kept / tot_n:.1f}% kept overall)")


if __name__ == "__main__":
    main()
