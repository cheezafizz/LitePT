"""Count per-class GT instance occurrences in the val split.

The training run logs show Class_0..3 (board, tray, paper, table) all returning
NaN for AP. This script disambiguates the two possible causes:
  (a) those classes simply have no GT instances in val -> NaN is expected, or
  (b) they exist in val but the model never predicts them -> a real failure that
      warrants class-weighted CE or focal loss.

Run from the repo root:
    python tools/diagnose_val_classes.py
    python tools/diagnose_val_classes.py --data-root data/scannet --split val
"""

import argparse
import os
import glob
from collections import Counter

import numpy as np


CLASS_NAMES = ["floor", "machine", "board", "tray", "paper", "table", "object"]
SEGMENT_IGNORE_INDEX = (-1, 0, 1)
INSTANCE_IGNORE_INDEX = -1


def count_split(data_root: str, split: str):
    split_dir = os.path.join(data_root, split)
    scenes = sorted(
        d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)
    )
    if not scenes:
        raise FileNotFoundError(f"No scenes under {split_dir}")

    per_class_counts = Counter()
    per_class_scene_presence = Counter()
    total_points_per_class = Counter()
    scenes_with_no_eval_instance = []

    for scene_dir in scenes:
        seg_path_20 = os.path.join(scene_dir, "segment20.npy")
        seg_path_200 = os.path.join(scene_dir, "segment200.npy")
        inst_path = os.path.join(scene_dir, "instance.npy")
        if not os.path.exists(inst_path):
            continue
        if os.path.exists(seg_path_20):
            segment = np.load(seg_path_20).reshape(-1).astype(np.int64)
        elif os.path.exists(seg_path_200):
            segment = np.load(seg_path_200).reshape(-1).astype(np.int64)
        else:
            continue
        instance = np.load(inst_path).reshape(-1).astype(np.int64)

        for c, n in zip(*np.unique(segment, return_counts=True)):
            total_points_per_class[int(c)] += int(n)

        keep = ~np.isin(segment, SEGMENT_IGNORE_INDEX) & (
            instance != INSTANCE_IGNORE_INDEX
        )
        if not keep.any():
            scenes_with_no_eval_instance.append(os.path.basename(scene_dir))
            continue

        seg_kept = segment[keep]
        inst_kept = instance[keep]
        unique_pairs = set(zip(inst_kept.tolist(), seg_kept.tolist()))
        scene_classes = set()
        for _, cls in unique_pairs:
            per_class_counts[int(cls)] += 1
            scene_classes.add(int(cls))
        for cls in scene_classes:
            per_class_scene_presence[cls] += 1

    return {
        "num_scenes": len(scenes),
        "per_class_counts": per_class_counts,
        "per_class_scene_presence": per_class_scene_presence,
        "total_points_per_class": total_points_per_class,
        "scenes_with_no_eval_instance": scenes_with_no_eval_instance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/scannet")
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    r = count_split(args.data_root, args.split)
    n = r["num_scenes"]
    print(f"Split: {args.split}  Scenes: {n}")
    print(f"Segment ignore index: {SEGMENT_IGNORE_INDEX}")
    print()
    print("Per-class GT instance counts (excluding ignored segments):")
    print(f"{'idx':>4}  {'class':<10}  {'instances':>10}  {'scenes':>8}  {'points':>12}")
    for idx, name in enumerate(CLASS_NAMES):
        n_inst = r["per_class_counts"].get(idx, 0)
        n_scenes = r["per_class_scene_presence"].get(idx, 0)
        n_pts = r["total_points_per_class"].get(idx, 0)
        marker = "  (ignored in eval)" if idx in SEGMENT_IGNORE_INDEX else ""
        print(f"{idx:>4}  {name:<10}  {n_inst:>10d}  {n_scenes:>8d}  {n_pts:>12d}{marker}")

    no_eval = r["scenes_with_no_eval_instance"]
    if no_eval:
        print(f"\nScenes with zero evaluable instances ({len(no_eval)}):")
        for s in no_eval[:20]:
            print(f"  {s}")
        if len(no_eval) > 20:
            print(f"  ... and {len(no_eval) - 20} more")


if __name__ == "__main__":
    main()
