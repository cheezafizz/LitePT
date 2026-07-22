"""
Phase-0 ORACLE diagnostic (no training): attribute the InsSeg over/under-segmentation
ceiling to the CLUSTERING ALGORITHM vs the PER-POINT PREDICTIONS, on the SYNTHETIC val
split (which -- unlike the VGGT val_sample scenes -- has clean 3D instance GT + a working
mAP evaluator, and is in-domain so no sim2real gap confounds the algorithm test).

The question this answers: would replacing offset+BFS clustering with a query-based mask
head (Mask2Former/Mask3D-style) help, or is the real problem just degraded predictions?

Method: run the model's own _cluster() post-processing but SUBSTITUTE oracle per-point
inputs at the call site -- NO model edit, NO edit to reeval_insseg_cluster_sweep.py:
  * oracle OFFSET  = near-perfect centroid shift (points moved most of the way to their
    instance centroid; see ORACLE_SHRINK for why not all the way), falling back to the
    predicted offset on instance-ignored points (which the offset loss never supervises,
    so "perfect" is undefined there).
  * oracle SEMANTIC = a one-hot logit from the GT `segment` (so _cluster's
    argmax(softmax(logit)) == the GT class; ignored classes stay masked as usual).

Four arms per scene, all on the SAME cached backbone predictions:
  baseline  : predicted offset + predicted semantic  (== the deployed number)
  +oseg     : predicted offset + ORACLE  semantic     (isolates the semantic-head deficit)
  +obias    : ORACLE  offset + predicted semantic     (isolates the offset-head deficit)
  +both     : ORACLE  offset + ORACLE  semantic       (the CLUSTERING CEILING)

Reading the result:
  * If +both mAP is HIGH (~>=0.9) but baseline is much lower -> clustering is FINE given
    good inputs; the loss is in the PREDICTIONS -> on VGGT that means the DOMAIN GAP
    (data / Phase 1) is the lever, NOT a new head.
  * If +both mAP is capped well below 1.0 even with perfect inputs -> the offset+BFS
    ALGORITHM cannot separate the (touching, same-class) instances -> a query-based head
    (Phase 2) is justified. The residual is the algorithmic ceiling.

Reuses the EXACT AP logic and per-scene caching pattern of
tools/reeval_insseg_cluster_sweep.py and engines/hooks/evaluator.py:InsSegEvaluator, so
numbers are comparable to training-time evals.

Run with the litept env, fp32 (bf16 breaks the spconv autotuner in eval):
  /home/fai/miniconda3/envs/litept/bin/python tools/oracle_insseg_diagnostic.py \
      --config-file configs/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3.py \
      --weight exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed/model/model_best.pth \
      --num-scenes 200 --output logs/oracle_insseg_diagnostic.csv
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

# Reuse the checkpoint loader + val-loader builder verbatim from the sweep tool.
from reeval_insseg_cluster_sweep import load_weight, build_val_loader  # noqa: E402


# (label, use_oracle_bias, use_oracle_seg). The clustering RADIUS/count thresholds are
# left at whatever the config sets (the 5 mm operating point) so the ONLY variables are
# the two oracle toggles -- this isolates input quality from post-processing tuning.
ARMS = [
    ("baseline", False, False),
    ("+oseg", False, True),
    ("+obias", True, False),
    ("+both", True, True),
]


# Fraction of each instance's original spread to PRESERVE when applying the oracle
# offset. A perfect offset (bias = centroid - coord) collapses every instance point to
# the exact same centroid; ballquery_batch_p then hits its per-point neighbour cap on the
# coincident cloud and BFS only reaches a bounded ~1k-point subset, so proposals come out
# pure but truncated (verified: IoU ~0.31). Preserving 20% of the spread keeps normal
# point density (ball-query behaves) while instances -- whose centroids are cm apart --
# still separate cleanly (verified: IoU ~0.98-1.0). This measures the CLUSTERING CEILING
# under near-perfect offsets without the degeneracy.
ORACLE_SHRINK = 0.2


def oracle_bias(input_dict, coord, bias_pred):
    """Near-perfect centroid offset where the point belongs to a real instance, else the
    predicted offset (instance-ignored points are unsupervised by the offset loss, so a GT
    offset is undefined there). Points are moved (1-ORACLE_SHRINK) of the way to their
    instance centroid (see ORACLE_SHRINK). instance_centroid is broadcast per-instance by
    InstanceParser."""
    ic = input_dict["instance_centroid"].float()
    inst = input_dict["instance"]
    bias_gt = (ic - coord) * (1.0 - ORACLE_SHRINK)
    valid = (inst >= 0).unsqueeze(-1)
    return torch.where(valid, bias_gt, bias_pred)


def oracle_logit(input_dict, n_cls, like):
    """One-hot logit from GT `segment` so argmax(softmax(logit)) == the GT class. Ignored
    class -1 clamps to 0, which _cluster masks out via segment_ignore_index anyway."""
    seg = input_dict["segment"].long()
    logit = torch.full((seg.shape[0], n_cls), -10.0, device=like.device, dtype=like.dtype)
    logit.scatter_(1, seg.clamp(min=0).unsqueeze(1), 20.0)
    return logit


def count_gt_instances(segment, instance, ignore_set, instance_ignore_index):
    """# of scoreable GT instances (id >= 0 and class not ignored) -- the reference the
    predicted instance count is over/under- relative to."""
    seg = segment.cpu().numpy()
    inst = instance.cpu().numpy()
    ids, idx = np.unique(inst, return_index=True)
    n = 0
    for i, iid in zip(idx, ids):
        if iid == instance_ignore_index:
            continue
        if int(seg[i]) in ignore_set:
            continue
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weight", required=True, help="checkpoint .pth")
    ap.add_argument("--num-scenes", type=int, default=200,
                    help="number of val scenes (subset, in order) to evaluate")
    ap.add_argument("--output", default="logs/oracle_insseg_diagnostic.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    ignore_set = set(cfg.model.segment_ignore_index)
    instance_ignore_index = cfg.model.instance_ignore_index

    val_loader, n = build_val_loader(cfg, args.num_scenes)

    model = build_model(cfg.model)
    model = load_weight(model, args.weight)
    model = model.cuda().eval()  # fp32; bf16 breaks the spconv autotuner in eval
    n_cls = model.seg_head.out_features
    radius_m = float(model.cluster_thresh) * float(model.voxel_size)
    print(f"[cfg] clustering radius = cluster_thresh({model.cluster_thresh}) * "
          f"voxel_size({model.voxel_size}) = {radius_m*1000:.1f} mm | "
          f"propose>{model.cluster_propose_points} min>{model.cluster_min_points} | "
          f"instance_embedding={getattr(model, 'instance_embedding', False)}")

    # One evaluator instance reused for the AP logic; trainer shim only needs .cfg.
    evaluator = InsSegEvaluator(
        segment_ignore_index=cfg.model.segment_ignore_index,
        instance_ignore_index=cfg.model.instance_ignore_index,
    )
    evaluator.trainer = argparse.Namespace(cfg=cfg)
    evaluator.valid_class_names = [
        cfg.data.names[i] for i in range(cfg.data.num_classes)
        if i not in cfg.model.segment_ignore_index
    ]

    scenes = {label: [] for (label, *_r) in ARMS}
    diag = {label: {"n_pred": []} for (label, *_r) in ARMS}
    gt_counts = []

    print(f"[oracle] {len(ARMS)} arms x {n} scenes "
          f"(backbone cached once per scene; only the oracle toggles vary)")

    for input_dict in tqdm(val_loader, desc="scenes"):
        for k in list(input_dict.keys()):
            if isinstance(input_dict[k], torch.Tensor):
                input_dict[k] = input_dict[k].cuda(non_blocking=True)

        idx = None
        if "origin_coord" in input_dict:
            idx, _ = pointops.knn_query(
                1, input_dict["coord"].float(), input_dict["offset"].int(),
                input_dict["origin_coord"].float(), input_dict["origin_offset"].int())
            idx = idx.cpu().flatten().long()
            segment = input_dict["origin_segment"]
            instance = input_dict["origin_instance"]
        else:
            segment = input_dict["segment"]
            instance = input_dict["instance"]
        gt_counts.append(
            count_gt_instances(segment, instance, ignore_set, instance_ignore_index))

        # Cache backbone + heads ONCE per scene; each arm only re-runs _cluster.
        with torch.no_grad():
            bias_pred, logit_pred = model._forward_backbone(input_dict)
        coord = input_dict["coord"]
        offset = input_dict["offset"]

        bias_o = oracle_bias(input_dict, coord, bias_pred)
        logit_o = oracle_logit(input_dict, n_cls, logit_pred)

        for (label, use_bias, use_seg) in ARMS:
            b = bias_o if use_bias else bias_pred
            lg = logit_o if use_seg else logit_pred
            with torch.no_grad():
                out = model._cluster(coord, b, lg, offset)
            diag[label]["n_pred"].append(int(out["pred_masks"].shape[0]))
            if idx is not None:
                out["pred_masks"] = out["pred_masks"][:, idx]
            gt_i, pred_i = evaluator.associate_instances(out, segment, instance)
            scenes[label].append(dict(gt=gt_i, pred=pred_i))

    avg_gt = float(np.mean(gt_counts)) if gt_counts else 0.0
    rows = []
    print("\n==================== ORACLE INSSEG DIAGNOSTIC ====================")
    print(f"scenes={n}  radius={radius_m*1000:.1f}mm  avg GT instances/scene={avg_gt:.1f}")
    print(f"{'arm':10s} {'mAP':>8s} {'AP50':>8s} {'AP25':>8s} {'n_pred':>8s} {'pred/gt':>8s}")
    for (label, use_bias, use_seg) in ARMS:
        ap_scores = evaluator.evaluate_matches(scenes[label])
        mAP = ap_scores["all_ap"]
        ap50 = ap_scores["all_ap_50%"]
        ap25 = ap_scores["all_ap_25%"]
        avg_np = float(np.mean(diag[label]["n_pred"])) if diag[label]["n_pred"] else 0.0
        print(f"{label:10s} {mAP:8.4f} {ap50:8.4f} {ap25:8.4f} {avg_np:8.1f} "
              f"{(avg_np/avg_gt if avg_gt else 0):8.2f}")
        row = dict(arm=label, oracle_bias=use_bias, oracle_seg=use_seg,
                   radius_m=radius_m, mAP=mAP, AP50=ap50, AP25=ap25,
                   avg_pred_instances=avg_np, avg_gt_instances=avg_gt)
        for cname in evaluator.valid_class_names:
            row[f"AP_{cname}"] = ap_scores["classes"][cname]["ap"]
        rows.append(row)
    print("==================================================================\n")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[out] wrote {args.output}")


if __name__ == "__main__":
    main()
