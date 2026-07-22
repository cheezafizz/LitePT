"""
Phase-0 confirmation (no training): re-evaluate an existing InsSeg checkpoint on the
val split while sweeping the PointGroup clustering post-processing parameters.

Why: the 2of3 industrial scenes pack objects as close as <1 cm apart (a mix of same-
and different-class neighbors). The PointGroup clustering radius is
`cluster_thresh * voxel_size`; same-class neighbors get NO semantic protection in the
BFS connected-component pass, so a radius that is too coarse merges them, while a radius
that is too tight (or count thresholds that are too high for the ~6x denser 2 mm cloud)
fragments / deletes small objects. This script sweeps BOTH the effective radius and the
density-dependent point-count thresholds on a single fixed checkpoint, to find the best
post-processing operating point and to separate the clustering amplifier from any model
deficit (receptive-field / scale mismatch).

It reuses the *exact* AP logic from `engines/hooks/evaluator.py:InsSegEvaluator`
(`associate_instances` + `evaluate_matches`) so numbers are comparable to training-time evals.

Each val scene is loaded ONCE and the backbone + heads run ONCE per scene; every
clustering setting is evaluated on those identical cached per-point predictions, so the
only variable across settings is the post-processing.

IMPORTANT: clustering tuning only helps when the offset head already shifts points well.
Run against a checkpoint with healthy offsets (val mAP >~ 0.3) -- NOT the collapsed run.

Run with the litept env python, fp32 (NOT bf16 -- spconv autotuner crashes in bf16 eval):
  /home/fai/miniconda3/envs/litept/bin/python tools/reeval_insseg_cluster_sweep.py \
      --config-file configs/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3.py \
      --weight <HEALTHY-OFFSET checkpoint .pth> \
      --num-scenes 200 \
      --output logs/cluster_sweep_radius.csv

First, size the count thresholds from the data (no checkpoint needed):
  /home/fai/miniconda3/envs/litept/bin/python tools/reeval_insseg_cluster_sweep.py \
      --config-file configs/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3.py \
      --measure --num-scenes 200
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pointops  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.env import set_seed  # noqa: E402
from datasets import build_dataset, collate_fn  # noqa: E402
from models import build_model  # noqa: E402
from engines.hooks.evaluator import InsSegEvaluator  # noqa: E402


# ----------------------------------------------------------------------------
# Sweep grid: vary the effective radius AND the density-dependent count thresholds.
# Physical ball radius = cluster_thresh * voxel_size. With voxel_size pinned to the
# 2 mm grid, cluster_thresh = radius_m / 0.002. The radii below span 1 cm (tight, for
# the closest same-class pairs) up to the legacy 3 cm.
#
# The count combos (closed_points, propose_points, min_points) should be bounded by the
# `--measure` output: set propose/min just BELOW the smallest real object's point count
# (at the 2 mm GridSample resolution -- that is the resolution the proposals live in)
# so real small objects survive but spurious fragments do not. Adjust after measuring.
# ----------------------------------------------------------------------------
VOXEL_SIZE = 0.002
# A 30-scene smoke sweep on the 2of3 `rigidaug` checkpoint showed mAP is dominated by the
# radius and peaks sharply at ~1 cm (3cm=0.016 -> 1cm=0.73 mAP), so this grid resolves the
# sub-1cm region finely. Count thresholds barely move mAP in the good-radius region, so only
# two representative combos are kept (object p1=7/p5=372 pts at 2mm -> keep propose low for
# small-object recall; see --measure).
RADII_M = [0.005, 0.0075, 0.010, 0.0125, 0.015, 0.020, 0.030]
COUNT_COMBOS = [
    # (closed_points, propose_points, min_points)
    (3000, 100, 50),
    (3000, 300, 100),
]


def build_sweep():
    # Keep the legacy "as-run" main-config setting first as a reference baseline row.
    sweep = [("baseline(0.02/1.5)", 0.020, 1.5, 600, 200, 50)]
    for r in RADII_M:
        thresh = round(r / VOXEL_SIZE, 4)
        for (closed, propose, minpts) in COUNT_COMBOS:
            label = f"r{int(round(r * 1000))}mm_p{propose}_m{minpts}"
            sweep.append((label, VOXEL_SIZE, thresh, closed, propose, minpts))
    return sweep


SWEEP = build_sweep()


def load_weight(model, weight_path):
    # weights_only=False: our own trusted checkpoint carries numpy scalars
    # (best_metric_value) that the PyTorch>=2.6 safe unpickler rejects by default.
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"[ckpt] loaded {weight_path} | epoch={epoch} "
          f"| missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"[ckpt] missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"[ckpt] unexpected (first 5): {unexpected[:5]}")
    return model


def build_val_loader(cfg, num_scenes):
    val_data = build_dataset(cfg.data.val)
    n = min(num_scenes, len(val_data))
    val_data = torch.utils.data.Subset(val_data, range(n))
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=1, shuffle=False, num_workers=4,
        pin_memory=False, collate_fn=collate_fn,
    )
    print(f"[data] {n} val scenes from {cfg.data.val.data_root}")
    return val_loader, n


def _pct(values):
    """Summary percentiles of a 1-D list. Returns None if empty."""
    a = np.sort(np.asarray(values, dtype=np.float64))
    if a.size == 0:
        return None

    def q(p):
        return float(a[min(a.size - 1, int(p * a.size))])

    return dict(n=int(a.size), min=float(a[0]), p1=q(0.01), p5=q(0.05),
                p25=q(0.25), median=q(0.50))


def _instance_stats(coord, segment, instance, ignore_set):
    """Per GT instance (at the model/GridSample resolution): (class, point_count,
    subsampled metric points for gap calc). Skips ignored classes and instance id < 0."""
    stats = []
    for iid in torch.unique(instance).tolist():
        if iid < 0:
            continue
        sel = instance == iid
        cnt = int(sel.sum().item())
        if cnt == 0:
            continue
        cls = int(torch.mode(segment[sel])[0].item())
        if cls in ignore_set:
            continue
        pts = coord[sel]
        k = min(cnt, 512)
        if cnt > k:
            sub = pts[torch.randperm(cnt, device=pts.device)[:k]]
        else:
            sub = pts
        stats.append((cls, cnt, sub))
    return stats


def measure(cfg, num_scenes, names, ignore_set):
    """Checkpoint-free GT diagnostic: per-class instance point-count distribution (at
    the 2 mm model resolution -> bounds for min/propose_points) and nearest same-class
    surface-gap distribution (-> ceiling for the clustering radius)."""
    val_loader, _ = build_val_loader(cfg, num_scenes)
    counts = defaultdict(list)   # cls -> [point_count, ...]
    gaps = defaultdict(list)     # cls -> [nearest same-class surface gap (m), ...]

    for input_dict in tqdm(val_loader, desc="measure"):
        coord = input_dict["coord"].float().cuda()
        segment = input_dict["segment"].cuda()
        instance = input_dict["instance"].cuda()
        stats = _instance_stats(coord, segment, instance, ignore_set)

        by_cls = defaultdict(list)
        for (cls, cnt, sub) in stats:
            counts[cls].append(cnt)
            by_cls[cls].append(sub)

        for cls, subs in by_cls.items():
            if len(subs) < 2:
                continue
            for i, a in enumerate(subs):
                best = float("inf")
                for j, b in enumerate(subs):
                    if i == j:
                        continue
                    d = torch.cdist(a, b).min().item()
                    if d < best:
                        best = d
                if best < float("inf"):
                    gaps[cls].append(best)

    print("\n==================== GT INSTANCE SIZE (points @ 2mm model res) ====================")
    print(f"{'class':10s} {'n_inst':>7s} {'min':>8s} {'p1':>8s} {'p5':>8s} {'p25':>8s} {'median':>8s}")
    for cls in sorted(counts):
        s = _pct(counts[cls])
        cn = names[cls] if cls < len(names) else str(cls)
        print(f"{cn:10s} {s['n']:7d} {s['min']:8.0f} {s['p1']:8.0f} {s['p5']:8.0f} "
              f"{s['p25']:8.0f} {s['median']:8.0f}")
    print("  -> set cluster_propose_points / min_points just BELOW the smallest real object (min/p1).")

    print("\n==================== NEAREST SAME-CLASS SURFACE GAP (meters) ====================")
    print(f"{'class':10s} {'n_pair':>7s} {'min':>8s} {'p1':>8s} {'p5':>8s} {'p25':>8s} {'median':>8s}")
    for cls in sorted(gaps):
        s = _pct(gaps[cls])
        if s is None:
            continue
        cn = names[cls] if cls < len(names) else str(cls)
        print(f"{cn:10s} {s['n']:7d} {s['min']:8.4f} {s['p1']:8.4f} {s['p5']:8.4f} "
              f"{s['p25']:8.4f} {s['median']:8.4f}")
    print("  -> keep radius (cluster_thresh*voxel_size) BELOW the small-percentile same-class gap.")
    print("==================================================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weight", default=None,
                    help="checkpoint .pth (required unless --measure)")
    ap.add_argument("--num-scenes", type=int, default=200,
                    help="number of val scenes (subset, in order) to evaluate")
    ap.add_argument("--output", default="logs/cluster_sweep.csv")
    ap.add_argument("--measure", action="store_true",
                    help="checkpoint-free: print GT instance size + same-class gap stats, then exit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    ignore_set = set(cfg.model.segment_ignore_index)

    if args.measure:
        measure(cfg, args.num_scenes, cfg.data.names, ignore_set)
        return

    if args.weight is None:
        ap.error("--weight is required unless --measure is set")

    val_loader, n = build_val_loader(cfg, args.num_scenes)

    model = build_model(cfg.model)
    model = load_weight(model, args.weight)
    model = model.cuda().eval()  # fp32; no autocast (bf16 breaks spconv autotuner)

    # One evaluator instance reused for the AP logic; trainer shim only needs .cfg.
    evaluator = InsSegEvaluator(
        segment_ignore_index=cfg.model.segment_ignore_index,
        instance_ignore_index=cfg.model.instance_ignore_index,
    )
    evaluator.trainer = argparse.Namespace(cfg=cfg)
    evaluator.valid_class_names = [
        cfg.data.names[i]
        for i in range(cfg.data.num_classes)
        if i not in cfg.model.segment_ignore_index
    ]

    # Per-setting accumulators.
    scenes = {label: [] for (label, *_rest) in SWEEP}
    diag = {label: {"n_pred": [], "ppp": []} for (label, *_rest) in SWEEP}

    print(f"[sweep] {len(SWEEP)} settings x {n} scenes "
          f"(backbone cached once per scene; only post-processing varies)")

    for input_dict in tqdm(val_loader, desc="scenes"):
        for k in list(input_dict.keys()):
            if isinstance(input_dict[k], torch.Tensor):
                input_dict[k] = input_dict[k].cuda(non_blocking=True)

        # Origin mapping is independent of clustering -> compute once per scene.
        idx = None
        if "origin_coord" in input_dict:
            idx, _ = pointops.knn_query(
                1,
                input_dict["coord"].float(), input_dict["offset"].int(),
                input_dict["origin_coord"].float(), input_dict["origin_offset"].int(),
            )
            idx = idx.cpu().flatten().long()
            segment = input_dict["origin_segment"]
            instance = input_dict["origin_instance"]
        else:
            segment = input_dict["segment"]
            instance = input_dict["instance"]

        # Cache backbone + heads ONCE per scene; sweep only re-runs _cluster.
        with torch.no_grad():
            bias_pred, logit_pred = model._forward_backbone(input_dict)
        coord = input_dict["coord"]
        offset = input_dict["offset"]

        for (label, vsize, thresh, closed, propose, minpts) in SWEEP:
            model.voxel_size = vsize
            model.cluster_thresh = thresh
            model.cluster_closed_points = closed
            model.cluster_propose_points = propose
            model.cluster_min_points = minpts
            out = model._cluster(coord, bias_pred, logit_pred, offset)

            # Diagnostics at model resolution (before remap to origin).
            n_pred = int(out["pred_masks"].shape[0])
            diag[label]["n_pred"].append(n_pred)
            diag[label]["ppp"].append(
                float(out["pred_masks"].sum(1).float().mean().item()) if n_pred > 0 else 0.0
            )

            if idx is not None:
                out["pred_masks"] = out["pred_masks"][:, idx]
            gt_i, pred_i = evaluator.associate_instances(out, segment, instance)
            scenes[label].append(dict(gt=gt_i, pred=pred_i))

    # Score each setting and emit a table + CSV.
    rows = []
    print("\n==================== CLUSTER SWEEP RESULTS ====================")
    print(f"{'setting':22s} {'radius_m':>8s} {'mAP':>8s} {'AP50':>8s} {'AP25':>8s} "
          f"{'n_pred':>7s} {'pts/prop':>9s}")
    for (label, vsize, thresh, closed, propose, minpts) in SWEEP:
        ap_scores = evaluator.evaluate_matches(scenes[label])
        mAP = ap_scores["all_ap"]
        ap50 = ap_scores["all_ap_50%"]
        ap25 = ap_scores["all_ap_25%"]
        avg_npred = float(np.mean(diag[label]["n_pred"])) if diag[label]["n_pred"] else 0.0
        avg_ppp = float(np.mean(diag[label]["ppp"])) if diag[label]["ppp"] else 0.0
        radius_m = thresh * vsize
        print(f"{label:22s} {radius_m:8.4f} {mAP:8.4f} {ap50:8.4f} {ap25:8.4f} "
              f"{avg_npred:7.1f} {avg_ppp:9.1f}")
        row = dict(setting=label, voxel_size=vsize, cluster_thresh=thresh,
                   closed_points=closed, propose_points=propose, min_points=minpts,
                   radius_m=radius_m, mAP=mAP, AP50=ap50, AP25=ap25,
                   avg_pred_instances=avg_npred, avg_points_per_proposal=avg_ppp)
        for cname in evaluator.valid_class_names:
            row[f"AP_{cname}"] = ap_scores["classes"][cname]["ap"]
        rows.append(row)
    print("==============================================================\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[out] wrote {args.output}")


if __name__ == "__main__":
    main()
