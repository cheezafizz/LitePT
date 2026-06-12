"""Save eval-style GLB visualizations (RGB + GT instances) for a single
preprocessed ScanNet scene. Mirrors what InsSegEvaluator._save_viz emits
during validation, minus the model-prediction view.

Usage:
    python tools/visualize_scene.py \
        --scene scene70_0001 --split train --out /tmp/scene70_0001
"""
import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from engines.hooks.insseg_viz import (
    colorize_gt_instances,
    instance_palette,
    save_pointcloud_glb,
    voxel_downsample_indices,
)


GRID_SIZE = 0.005
IGNORE_INSTANCE = -1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="scene70_0001")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--data-root", default=os.path.join(_REPO_ROOT, "data", "scannet"))
    p.add_argument("--out", default=None,
                   help="Output directory (default: /tmp/<scene>)")
    p.add_argument("--min-region-size", type=int, default=100,
                   help="Drop GT instances with fewer points than this from AABB overlay.")
    return p.parse_args()


def print_dataset_summary(scene_dir, coord, color, normal, segment20, instance):
    print("=" * 72)
    print(f"scene at {scene_dir}")
    print("=" * 72)
    fields = [
        ("coord.npy",     coord,     "XYZ in meters (axis-aligned)"),
        ("color.npy",     color,     "RGB 0-255 from PLY vertex colors"),
        ("normal.npy",    normal,    "area-weighted vertex normals"),
        ("segment20.npy", segment20, "ScanNet-20 class id; -1 = ignore"),
        ("instance.npy",  instance,  "aggregation group id; -1 = ignore"),
    ]
    for name, arr, desc in fields:
        rng = f"[{arr.min()}, {arr.max()}]" if arr.size else "[empty]"
        print(f"  {name:14s} shape={str(arr.shape):16s} dtype={str(arr.dtype):8s} range={rng}  {desc}")
    n_inst = np.unique(instance[instance != IGNORE_INSTANCE]).size
    n_sem = np.unique(segment20[segment20 != -1]).size
    n_ignored = int((instance == IGNORE_INSTANCE).sum())
    print(f"  points={coord.shape[0]:,}  gt_instances={n_inst}  semantic_classes={n_sem}  unlabeled_pts={n_ignored:,}")
    print("=" * 72)


def main():
    args = parse_args()
    scene_dir = os.path.join(args.data_root, args.split, args.scene)
    if not os.path.isdir(scene_dir):
        raise SystemExit(f"scene dir not found: {scene_dir}")
    out_dir = args.out or os.path.join("/tmp", args.scene)

    coord = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(scene_dir, "color.npy"))
    normal = np.load(os.path.join(scene_dir, "normal.npy"))
    segment20 = np.load(os.path.join(scene_dir, "segment20.npy"))
    instance = np.load(os.path.join(scene_dir, "instance.npy"))

    print_dataset_summary(scene_dir, coord, color, normal, segment20, instance)

    # AABBs on full-resolution coords so boxes hug true extents
    # (matches InsSegEvaluator._save_viz at engines/hooks/evaluator.py:484-492).
    uniq_gt = np.unique(instance[instance != IGNORE_INSTANCE])
    palette = instance_palette(len(uniq_gt), seed=1)
    gt_boxes = []
    for k, inst in enumerate(uniq_gt):
        m = instance == inst
        if int(m.sum()) < args.min_region_size:
            continue
        pts = coord[m]
        gt_boxes.append((pts.min(0), pts.max(0), palette[k]))

    # 0.005 m voxel downsample for lighter GLBs (same as evaluator.py:494).
    kept = voxel_downsample_indices(coord, GRID_SIZE)
    coord_s = coord[kept]
    color_s = color[kept].astype(np.uint8)
    instance_s = instance[kept]
    gt_color = colorize_gt_instances(instance_s, ignore_value=IGNORE_INSTANCE)

    rgb_path = os.path.join(out_dir, f"{args.scene}_rgb.glb")
    gt_path = os.path.join(out_dir, f"{args.scene}_gt.glb")
    save_pointcloud_glb(rgb_path, coord_s, color_s)
    save_pointcloud_glb(gt_path, coord_s, gt_color, boxes=gt_boxes)

    print(f"downsampled: {coord.shape[0]:,} -> {coord_s.shape[0]:,} points "
          f"(grid={GRID_SIZE} m), {len(gt_boxes)} GT boxes drawn "
          f"(>= {args.min_region_size} pts)")
    print(f"wrote {rgb_path}")
    print(f"wrote {gt_path}")


if __name__ == "__main__":
    main()
