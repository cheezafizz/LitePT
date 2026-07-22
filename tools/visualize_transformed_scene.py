"""Apply a config's transform pipeline to a single preprocessed scene and save
a GLB colored by GT instance labels.

Unlike tools/visualize_scene.py (which renders the raw on-disk scene), this runs
an exact transform chain from the config so you see what the model actually
receives. Pick the pipeline with --pipeline:
  * train (default): `cfg.data.train.transform` — the full augmented chain
    (CenterShift, RandomRotate/Scale/Flip, Chromatic*, GridSample, SphereCrop, ...),
    i.e. what the model sees during training.
  * val: `cfg.data.val.transform` — the deterministic, augmentation-free chain
    used at evaluation/inference (CenterShift, GridSample, NormalizeColor, ...),
    i.e. "as if it is done in evaluation".
Coloring + AABB boxes reuse the same helpers as the validation evaluator
(engines/hooks/insseg_viz.py).

CPU-only: no model, no GPU.

Usage:
    # what the model receives at EVALUATION, on a train-split scene:
    python tools/visualize_transformed_scene.py \
        --scene scene70_0001 --split train --pipeline val --save-rgb
"""
import argparse
import os
import random
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from datasets import build_dataset  # noqa: E402
from engines.hooks.insseg_viz import (  # noqa: E402
    colorize_gt_instances,
    instance_palette,
    save_pointcloud_glb,
    voxel_downsample_indices,
)
from utils.config import Config  # noqa: E402


IGNORE_INSTANCE = -1
_DEFAULT_CONFIG = os.path.join(
    _REPO_ROOT,
    "configs",
    "scannet-v1.1.1-2of3",
    "insseg-litept-small-v1m2-2of3.py",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-file", default=_DEFAULT_CONFIG,
                   help="Config whose data.<pipeline>.transform is applied.")
    p.add_argument("--pipeline", default="train", choices=["train", "val"],
                   help="Which transform chain to apply: 'train' (augmented) or "
                        "'val' (deterministic eval/inference pipeline).")
    p.add_argument("--scene", default="scene72_9456")
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="Split directory the scene lives in (data_root/<split>/<scene>).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the random augmentations (reproducible). Use -1 to randomize.")
    p.add_argument("--min-region-size", type=int, default=100,
                   help="Drop GT instances with fewer points than this from the AABB overlay.")
    p.add_argument("--voxel-size", type=float, default=0.005,
                   help="Voxel size for the final GLB downsample (≈no-op; train GridSample is 0.005).")
    p.add_argument("--out", default=None,
                   help="Output directory (default: /tmp/<scene>_transformed).")
    p.add_argument("--save-rgb", action="store_true",
                   help="Also write the augmented-RGB companion {scene}_rgb_transformed.glb.")
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(_REPO_ROOT)  # so the config's relative data_root resolves

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    suffix = "eval" if args.pipeline == "val" else "transformed"
    out_dir = args.out or os.path.join("/tmp", f"{args.scene}_{suffix}")

    cfg = Config.fromfile(args.config_file)
    pipeline_cfg = cfg.data.val if args.pipeline == "val" else cfg.data.train
    scene_dir = os.path.join(pipeline_cfg.data_root, args.split, args.scene)
    if not os.path.isdir(scene_dir):
        raise SystemExit(f"scene dir not found: {scene_dir}")

    # Build the dataset (test_mode=False) and point it at the single scene so
    # prepare_train_data runs the exact selected transform chain on it.
    dataset = build_dataset(pipeline_cfg)
    dataset.data_list = [scene_dir]
    data = dataset[0]

    # Train Collect keys: coord, instance, feat (= color|normal, color in cols 0-2).
    coord = data["coord"].numpy().astype(np.float32)
    inst = data["instance"].numpy().astype(np.int64)
    color = np.clip(data["feat"][:, :3].numpy() * 255.0, 0, 255).astype(np.uint8)

    # AABBs on the (transformed) coords so boxes hug the augmented instance extents.
    uniq_gt = np.unique(inst[inst != IGNORE_INSTANCE])
    palette = instance_palette(len(uniq_gt), seed=1)
    gt_boxes = []
    for k, i in enumerate(uniq_gt):
        m = inst == i
        if int(m.sum()) < args.min_region_size:
            continue
        pts = coord[m]
        gt_boxes.append((pts.min(0), pts.max(0), palette[k]))

    # Final voxel downsample for a lighter GLB (matches evaluator / visualize_scene).
    kept = voxel_downsample_indices(coord, args.voxel_size)
    coord_s = coord[kept]
    color_s = color[kept]
    inst_s = inst[kept]
    gt_color = colorize_gt_instances(inst_s, ignore_value=IGNORE_INSTANCE)

    gt_path = os.path.join(out_dir, f"{args.scene}_gt_{suffix}.glb")
    save_pointcloud_glb(gt_path, coord_s, gt_color, boxes=gt_boxes)
    if args.save_rgb:
        rgb_path = os.path.join(out_dir, f"{args.scene}_rgb_{suffix}.glb")
        save_pointcloud_glb(rgb_path, coord_s, color_s)

    seed_note = "random" if args.seed < 0 else f"seed={args.seed}"
    print("=" * 72)
    print(f"scene {args.scene} ({args.split}) — {args.pipeline.upper()} transforms "
          f"applied ({seed_note})")
    print("=" * 72)
    print(f"  points after transforms: {coord.shape[0]:,}  ->  "
          f"{coord_s.shape[0]:,} after {args.voxel_size} m downsample")
    print(f"  GT instances: {len(uniq_gt)}  |  AABB boxes drawn: {len(gt_boxes)} "
          f"(>= {args.min_region_size} pts)")
    print(f"  ignore/stuff points (instance==-1): {int((inst == IGNORE_INSTANCE).sum()):,}")
    print(f"wrote {gt_path}")
    if args.save_rgb:
        print(f"wrote {rgb_path}")


if __name__ == "__main__":
    main()
