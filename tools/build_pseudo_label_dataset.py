"""
Phase-1b: build per-point pseudo-label TRAINING scenes from real VGGT clouds, so a
LitePT InsSeg model can be co-trained in the VGGT domain (see the plan; Phase-0 showed the
deployment failure is the sim2real domain gap, dominated by the semantic head).

For each fused VGGT scene it writes a training-layout scene dir
(coord.npy / color.npy / normal.npy / segment20.npy / instance.npy) exactly like
datasets/preprocessing/v1_0_1/preprocess_v1_0_1.py, where:

  * SEMANTIC (segment20) = the D-FINE 2D `sem_seg` argmax, reprojected per point, REMAPPED
    from the D-FINE label scheme to the model's TRAINING class scheme (they differ! see
    LABEL_TO_TRAIN). Per-point, no cross-view linking needed -- the clean, high-value target.
  * INSTANCE (instance) = the current clustering's proposals (the ONLY cross-view linker,
    since 2D-mask ids are unique per (view, object)), remapped grid->origin. Points in a
    dropped/low-confidence proposal or none get instance_ignore_index (-1) -> the offset loss
    ignores them, while the semantic loss still supervises every point.

This is a self-training TEACHER = (D-FINE semantic) + (clustering instances). Expect the
student to beat it on SEMANTICS (multi-view denoising / 3D disambiguation / completeness /
no 2D dependency at inference) but NOT on the clustering's systematic instance errors.

Reuses tools/cluster_labeled_scenes.py wholesale (fuse + backbone + cluster) and only adds
the training-scene writer + the class remap. Run with the litept env, fp32:
  /home/fai/miniconda3/envs/litept/bin/python tools/build_pseudo_label_dataset.py \
      --out-root data/scannet-vggt-pseudo --split train \
      --clustering-config embed-maskclust-strong-rag-svdom-t03 --min-score 0.3
"""
import argparse
import os
import sys
import traceback

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
from models import build_model  # noqa: E402
import infer_insseg as ifs  # noqa: E402
import cluster_labeled_scenes as cls  # noqa: E402

# D-FINE sem_seg scheme (cls.LABEL_NAMES = background,machine,board,tray,paper,object,oob)
# -> model TRAINING scheme (config class_names = floor,machine,board,tray,paper,table,object;
# see datasets/preprocessing/v1_0_1/preprocess_v1_0_1.py + tools/verify_merge_counts.py).
# By NAME: background->floor(0), machine(1), board(2), tray(3), paper(4), object->object(6);
# D-FINE has no 'table' (5) -> pseudo scenes carry no table supervision. oob->object(6) per
# the synthetic convention (oob junk lives in class 6 alongside real objects).
LABEL_TO_TRAIN = np.array([0, 1, 2, 3, 4, 6, 6], dtype=np.int64)


def build_one(model, scene, transform, cluster_baseline, cfg_path, min_score):
    """Fuse+backbone+cluster one scene (reusing cluster_labeled_scenes) and return
    origin-resolution (coord, color, normal, segment_train, instance) ready to write, plus
    a per-scene stats dict. semantic_source='labels' -> the given D-FINE sem_seg drives it."""
    cached = cls.run_backbone(model, scene, transform, zero_offset=False,
                              semantic_source="labels")
    # reset clustering attrs to the snapshot baseline, then apply the chosen config
    for k, v in cluster_baseline.items():
        setattr(model, k, v)
    ifs.apply_cluster_config(model, cfg_path)
    out = cls.cluster_one(model, cached)

    # grid -> origin (fused) remap, exactly as cluster_labeled_scenes.write_outputs.
    input_dict = cached["input_dict"]
    origin_coord = input_dict["origin_coord"]
    o2g, _ = pointops.knn_query(
        1, input_dict["coord"].float(), input_dict["offset"].int(),
        origin_coord.float(), input_dict["origin_offset"].int())
    o2g = o2g.cpu().flatten().long()

    n_origin = origin_coord.shape[0]
    pred_masks = out["pred_masks"]          # (P, n_grid) int at grid res
    pred_scores = out["pred_scores"]
    P = int(pred_masks.shape[0])
    masks_origin = (pred_masks[:, o2g].bool().numpy() if P > 0
                    else np.zeros((0, n_origin), dtype=bool))
    scores = pred_scores.numpy() if P > 0 else np.zeros(0, np.float32)

    # Per-point instance id: keep only proposals with score >= min_score; disjoint masks
    # (post-clustering) so a simple assignment suffices. Unassigned/dropped -> -1 (ignore).
    instance = np.full(n_origin, -1, dtype=np.int64)
    kept = 0
    for p in range(P):
        if scores[p] < min_score:
            continue
        instance[masks_origin[p]] = kept
        kept += 1

    # Per-point semantic: D-FINE sem_seg (LABEL scheme, origin-res) remapped to train scheme.
    seg_label = scene["segment"].astype(np.int64)          # (n_origin,) in LABEL_NAMES scheme
    seg_label = np.clip(seg_label, 0, LABEL_TO_TRAIN.shape[0] - 1)
    segment_train = LABEL_TO_TRAIN[seg_label].astype(np.int32)

    coord = scene["coord"].astype(np.float32)
    color = np.clip(scene["color"], 0, 255).astype(np.uint8)
    normal = scene["normal"].astype(np.float32)
    assert coord.shape[0] == n_origin == segment_train.shape[0] == instance.shape[0], \
        f"length mismatch: coord{coord.shape[0]} origin{n_origin} " \
        f"seg{segment_train.shape[0]} inst{instance.shape[0]}"

    ucls, ccls = np.unique(segment_train, return_counts=True)
    stats = dict(
        n_pts=int(n_origin), n_prop=P, n_kept_inst=kept,
        n_fg_pts=int((instance >= 0).sum()),
        seg_hist={cfg_cls_name(c): int(n) for c, n in zip(ucls.tolist(), ccls.tolist())},
    )
    return coord, color, normal, segment_train, instance.astype(np.int32), stats


_CFG_NAMES = None


def cfg_cls_name(c):
    return _CFG_NAMES[c] if _CFG_NAMES and 0 <= c < len(_CFG_NAMES) else str(c)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=cls._DEFAULT_ROOT)
    ap.add_argument("--aligned-root", default=cls._DEFAULT_ALIGNED)
    ap.add_argument("--config-file", default=os.path.join(cls._DEFAULT_EXP, "config.py"))
    ap.add_argument("--weight", default=os.path.join(cls._DEFAULT_EXP, "model", "model_best.pth"))
    ap.add_argument("--clustering-dir", default=cls._DEFAULT_CLUSTERING_DIR)
    ap.add_argument("--clustering-config", default="embed-maskclust-strong-rag-svdom-t03",
                    help="clustering config name (in --clustering-dir) used as the instance teacher")
    ap.add_argument("--out-root", default="data/scannet-vggt-pseudo")
    ap.add_argument("--split", default="train")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="subset of '<group>/<seq>' (default: all under --root)")
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="drop clustering proposals below this confidence (-> instance ignore)")
    ap.add_argument("--dry-run", action="store_true",
                    help="compute + print stats but do NOT write .npy files")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    global _CFG_NAMES
    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    _CFG_NAMES = list(cfg.data.names)

    model = build_model(cfg.model)
    model = ifs.load_weight(model, args.weight)
    model = model.cuda().eval()  # fp32; bf16 breaks the spconv autotuner in eval
    if getattr(model, "embedding_head", None) is None:
        raise SystemExit("checkpoint has no embedding head; use the embed exp for embed* configs")

    base_voxel = float(model.voxel_size)
    transform = ifs.build_infer_transform(grid_size=base_voxel)
    fa = cls.fuse_args(base_voxel)
    cluster_baseline = {k: getattr(model, k) for k in ifs.CLUSTER_KEYS if hasattr(model, k)}
    cfgs = cls.load_clustering_configs(args.clustering_dir, [args.clustering_config])
    cfg_name, cfg_path = cfgs[0]

    scenes = args.scenes or cls.discover_scenes(args.root)
    out_split = os.path.join(args.out_root, args.split)
    print(f"[pseudo] {len(scenes)} scenes | teacher-cluster={cfg_name} | min_score={args.min_score} "
          f"| remap LABEL->train {LABEL_TO_TRAIN.tolist()} | -> {out_split}/ "
          f"{'(DRY RUN)' if args.dry_run else ''}")

    n_ok = n_fail = 0
    for rel in scenes:
        seq_dir = os.path.join(args.root, rel)
        aligned_seq_dir = os.path.join(args.aligned_root, rel)
        scene_name = rel.replace("/", "__")
        try:
            scene = cls.build_labeled_scene(seq_dir, aligned_seq_dir, fa)
            coord, color, normal, seg, inst, st = build_one(
                model, scene, transform, cluster_baseline, cfg_path, args.min_score)
            print(f"  [{scene_name}] {st['n_pts']:,} pts | {st['n_kept_inst']}/{st['n_prop']} "
                  f"inst kept | {st['n_fg_pts']:,} fg pts | seg {st['seg_hist']}")
            if not args.dry_run:
                out_dir = os.path.join(out_split, scene_name)
                os.makedirs(out_dir, exist_ok=True)
                np.save(os.path.join(out_dir, "coord.npy"), coord)
                np.save(os.path.join(out_dir, "color.npy"), color)
                np.save(os.path.join(out_dir, "normal.npy"), normal)
                np.save(os.path.join(out_dir, "segment20.npy"), seg)
                np.save(os.path.join(out_dir, "instance.npy"), inst)
            n_ok += 1
        except Exception as e:  # noqa: BLE001 -- keep the batch going
            n_fail += 1
            print(f"  [ERROR] {rel}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"[done] {n_ok} scenes ok, {n_fail} failed -> {out_split}/")


if __name__ == "__main__":
    main()
