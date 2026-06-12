"""Run a trained PointGroup instance-segmentation model on the test split and
write per-scene GLB visualizations using the same helpers the training-time
evaluator uses (engines/hooks/insseg_viz.py).

Single-GPU, batch size 1, no DDP. Mirrors InsSegEvaluator._save_viz exactly.
"""

import argparse
import os
import re
import sys
from collections import OrderedDict
from functools import partial

import numpy as np
import pointops
import torch
from torch.utils.data import DataLoader

# Ensure the repo root is on sys.path so `engines`, `datasets`, `models` resolve
# when the script is invoked as `python tools/test_insseg_viz.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import build_dataset, point_collate_fn  # noqa: E402
from engines.hooks.insseg_viz import (  # noqa: E402
    colorize_gt_instances,
    colorize_predicted_instances,
    instance_palette,
    save_pointcloud_glb,
    voxel_downsample_indices,
)
from models import build_model  # noqa: E402
from utils.config import Config  # noqa: E402


DEFAULT_EXP_DIR = (
    "exp/scannet-v1.0.1-s4/insseg-litept-small-v1m2-vicinity-labelled-s4-ep200"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-file",
        default=os.path.join(DEFAULT_EXP_DIR, "config.py"),
        help="Path to the saved training config.",
    )
    p.add_argument(
        "--weight",
        default=os.path.join(DEFAULT_EXP_DIR, "model/model_best.pth"),
        help="Path to the model checkpoint.",
    )
    p.add_argument(
        "--data-root",
        default="data/scannet-v1.0.1-s4",
        help="Dataset root that contains the split directory.",
    )
    p.add_argument("--split", default="test", help="Split name (subdir under data-root).")
    p.add_argument("--max-scenes", type=int, default=20, help="0 means all scenes.")
    p.add_argument(
        "--out-dir",
        default=os.path.join(DEFAULT_EXP_DIR, "test_viz"),
        help="Where to write the GLB files.",
    )
    p.add_argument(
        "--min-region-size",
        type=int,
        default=50,
        help="Drop proposals / GT instances with fewer points than this when drawing AABBs.",
    )
    p.add_argument(
        "--voxel-size", type=float, default=0.005, help="Voxel size for GLB downsampling."
    )
    p.add_argument(
        "--save-rgb",
        action="store_true",
        help="Also save the {scene}_rgb.glb companion file.",
    )
    return p.parse_args()


def load_checkpoint(model, weight_path):
    """Mirror engines/test.py:61-73 for the single-GPU (world_size==1) case."""
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"No checkpoint at {weight_path}")
    print(f"=> Loading weight: {weight_path}")
    ckpt = torch.load(weight_path, weights_only=False, map_location="cpu")
    state = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        # world_size == 1 → strip the `module.` prefix DDP added during training.
        state[k[7:] if k.startswith("module.") else k] = v
    model.load_state_dict(state, strict=True)
    print(f"=> Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    return model


def scene_tag_from_name(raw_name, fallback_idx):
    if isinstance(raw_name, (list, tuple)) and raw_name:
        raw_name = raw_name[0]
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode()
    if isinstance(raw_name, str) and raw_name:
        tag = os.path.splitext(os.path.basename(raw_name))[0]
        return re.sub(r"[^A-Za-z0-9._-]", "_", tag)
    return f"scene{fallback_idx}"


def save_scene_viz(
    input_dict,
    output_dict,
    idx,
    scene_tag,
    out_dir,
    min_region_sizes,
    voxel_size,
    instance_ignore_index,
    save_rgb,
):
    """Direct port of InsSegEvaluator._save_viz (evaluator.py:456-538) with
    parameterised paths so it can run outside a trainer context."""
    coord = input_dict["origin_coord"].detach().cpu().numpy()
    # The val Collect concatenates color+normal into `feat` (cols 0-2 = color, 3-5 = normal).
    color_voxel = input_dict["feat"][:, :3].detach().cpu().numpy()
    idx_np = idx.cpu().numpy() if hasattr(idx, "cpu") else np.asarray(idx)
    color = np.clip(color_voxel[idx_np] * 255.0, 0, 255).astype(np.uint8)

    pred = output_dict["pred_masks"].detach().cpu().numpy().astype(bool)
    scores = output_dict["pred_scores"].detach().cpu().numpy()
    gt_inst = input_dict["origin_instance"].detach().cpu().numpy()

    # AABBs are computed on the full origin coord (before downsampling) so the
    # boxes hug the true extents of each instance.
    P = pred.shape[0]
    palette_pred = instance_palette(P)
    pred_boxes = []
    for p_idx in range(P):
        m = pred[p_idx]
        if m.sum() < min_region_sizes:
            continue
        pts = coord[m]
        pred_boxes.append((pts.min(0), pts.max(0), palette_pred[p_idx]))

    uniq_gt = np.unique(gt_inst[gt_inst != instance_ignore_index])
    palette_gt = instance_palette(len(uniq_gt), seed=1)
    gt_boxes = []
    for k, inst in enumerate(uniq_gt):
        m = gt_inst == inst
        if m.sum() < min_region_sizes:
            continue
        pts = coord[m]
        gt_boxes.append((pts.min(0), pts.max(0), palette_gt[k]))

    kept = voxel_downsample_indices(coord, voxel_size)
    coord_s = coord[kept]
    color_s = color[kept]
    pred_s = pred[:, kept] if pred.shape[0] > 0 else pred
    gt_inst_s = gt_inst[kept]

    pred_color = colorize_predicted_instances(coord_s.shape[0], pred_s, scores)
    gt_color = colorize_gt_instances(gt_inst_s, ignore_value=instance_ignore_index)

    if save_rgb:
        save_pointcloud_glb(
            os.path.join(out_dir, f"{scene_tag}_rgb.glb"), coord_s, color_s
        )
    save_pointcloud_glb(
        os.path.join(out_dir, f"{scene_tag}_pred.glb"),
        coord_s,
        pred_color,
        boxes=pred_boxes,
    )
    save_pointcloud_glb(
        os.path.join(out_dir, f"{scene_tag}_gt.glb"),
        coord_s,
        gt_color,
        boxes=gt_boxes,
    )
    return len(pred_boxes), len(gt_boxes)


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = Config.fromfile(args.config_file)
    # Point the val dataset at the test split.
    cfg.data.val.data_root = args.data_root
    cfg.data.val.split = args.split

    print(f"=> Dataset: {cfg.data.val.type} | root={cfg.data.val.data_root} | "
          f"split={cfg.data.val.split}")
    dataset = build_dataset(cfg.data.val)
    if args.max_scenes > 0:
        dataset.data_list = sorted(dataset.data_list)[: args.max_scenes]
    else:
        dataset.data_list = sorted(dataset.data_list)
    print(f"=> Iterating {len(dataset.data_list)} scenes")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=partial(point_collate_fn, mix_prob=0),
    )

    print("=> Building model")
    model = build_model(cfg.model)
    model = load_checkpoint(model, args.weight).cuda().eval()

    instance_ignore_index = -1  # matches evaluator default + cfg hook setting

    for i, input_dict in enumerate(loader):
        assert len(input_dict["offset"]) == 1, "batch size 1 required"
        for k in list(input_dict.keys()):
            if isinstance(input_dict[k], torch.Tensor):
                input_dict[k] = input_dict[k].cuda(non_blocking=True)

        with torch.no_grad():
            output_dict = model(input_dict)

        # Map voxel-resolution predictions back to origin resolution — same
        # nearest-neighbour query as engines/hooks/evaluator.py:564-572.
        idx, _ = pointops.knn_query(
            1,
            input_dict["coord"].float(),
            input_dict["offset"].int(),
            input_dict["origin_coord"].float(),
            input_dict["origin_offset"].int(),
        )
        idx = idx.cpu().flatten().long()
        output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]

        scene_tag = scene_tag_from_name(input_dict.get("name"), i)
        n_pred, n_gt = save_scene_viz(
            input_dict,
            output_dict,
            idx,
            scene_tag,
            args.out_dir,
            args.min_region_size,
            args.voxel_size,
            instance_ignore_index,
            args.save_rgb,
        )
        print(
            f"[{i + 1}/{len(loader)}] {scene_tag}: "
            f"{output_dict['pred_masks'].shape[0]} proposals "
            f"({n_pred} drawn) | {n_gt} GT instances drawn"
        )

    print(f"=> Done. GLBs written to {args.out_dir}")


if __name__ == "__main__":
    main()
