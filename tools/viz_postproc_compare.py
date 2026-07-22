"""
Compare post-processing of query-InsSeg (MQ-v1m1) predictions in viser: for each
scene, dump the RAW model output plus the full post-processing pipeline
(higher mask_threshold + connected-component split, applied together) at three
threshold values, as separate viewer "model" dirs.

Arms (each -> a Model-dropdown entry in tools/viser_insseg_viewer.py):
  baseline   mask_threshold=0.5, no split   ("before applying")
  both60     mask_threshold=0.6, CC split
  both70     mask_threshold=0.7, CC split
  both80     mask_threshold=0.8, CC split
  pseudo-gt  GT instances (written once)

mask_threshold is a settable model attribute (models/mask_query/mask_query_v1m1.py);
decoding is identical across thresholds (masked-attn gate is hard-coded 0.5), so
each threshold only re-binarizes the same soft masks -- we forward once per
distinct threshold and cache the result.

  /home/fai/miniconda3/envs/litept/bin/python tools/viz_postproc_compare.py \
      --exp-dir exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-query-realft-muon-nosem \
      --scenes-dir viz/query_nosem_train_manyinst/pseudo-gt \
      --out-dir viz/postproc_compare
  .venv-viser/bin/python tools/viser_insseg_viewer.py \
      --results-dir viz/postproc_compare --host 0.0.0.0 --port 8083
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.config import Config  # noqa: E402
from utils.env import set_seed  # noqa: E402
from datasets import build_dataset  # noqa: E402
from datasets.utils import collate_fn  # noqa: E402
from models import build_model  # noqa: E402
from tools.viz_query_insseg_val import load_weight  # noqa: E402
from tools.viz_query_insseg_cases import distinct_palette, save_npz  # noqa: E402
from tools.postprocess_insseg import split_disconnected_masks  # noqa: E402

# (arm name, mask_threshold, apply CC split). baseline == raw model output.
ARMS = [
    ("baseline", 0.5, False),
    ("both60", 0.6, True),
    ("both70", 0.7, True),
    ("both80", 0.8, True),
]


def flatten_masks(coord, masks, scores):
    """(M, N) bool masks -> per-point inst_id; higher-score masks win overlaps."""
    inst_id = np.full(coord.shape[0], -1, dtype=np.int32)
    for m in np.argsort(scores):
        inst_id[masks[m]] = m
    return inst_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--weight", default=None,
                    help="default: <exp-dir>/model/model_best.pth")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="explicit train scene names")
    ap.add_argument("--scenes-dir", default=None,
                    help="dir of <scene>_pred.npz to take scene names from "
                         "(e.g. viz/query_nosem_train_manyinst/pseudo-gt)")
    ap.add_argument("--split", default="train",
                    help="dataset split the scenes belong to (default: train)")
    ap.add_argument("--out-dir", required=True)
    # 2cm: predicted SURFACE masks are holey (sub-threshold gaps), so a small
    # 5mm radius shatters single objects into dozens of fragments. Empirically
    # (radius diagnostic) fragmentation plateaus at ~GT count by 20mm, and since
    # objects here sit 6-18cm apart, 2cm bridges within-object holes without ever
    # merging two distinct objects. Far-apart merges still split; touching
    # (<2cm) objects do not (CC split cannot separate those regardless).
    ap.add_argument("--split-radius", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.scenes:
        scenes = list(args.scenes)
    elif args.scenes_dir:
        scenes = sorted(os.path.basename(p)[: -len("_pred.npz")]
                        for p in glob.glob(os.path.join(args.scenes_dir, "*_pred.npz")))
    else:
        raise SystemExit("pass --scenes or --scenes-dir")
    if not scenes:
        raise SystemExit(f"no scenes found (scenes-dir={args.scenes_dir!r})")

    cfg = Config.fromfile(os.path.join(args.exp_dir, "config.py"))
    set_seed(args.seed)
    weight = args.weight or os.path.join(args.exp_dir, "model", "model_best.pth")
    class_names = list(cfg.data.names)

    model = build_model(cfg.model).cuda().eval()
    load_weight(model, weight)

    data_cfg = cfg.data.val
    data_cfg.split = args.split
    dataset = build_dataset(data_cfg)
    names = [dataset.get_data_name(i) for i in range(len(dataset))]

    for arm, _, _ in ARMS:
        os.makedirs(os.path.join(args.out_dir, arm), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "pseudo-gt"), exist_ok=True)

    print(f"[data] {len(scenes)} scenes, split={args.split}, "
          f"arms={[a for a, _, _ in ARMS]}")
    thresholds = sorted({t for _, t, _ in ARMS})

    for si, scene in enumerate(scenes):
        if scene not in names:
            print(f"[skip] {scene} not in {args.split} split")
            continue
        i = names.index(scene)
        batch = collate_fn([dataset[i]])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)

        coord = batch["coord"].cpu().numpy().astype(np.float32)
        rgb = ((batch["feat"][:, :3].cpu().numpy() + 1.0) * 127.5)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        # forward once per distinct threshold; cache raw (masks, scores, classes)
        cache = {}
        for t in thresholds:
            model.mask_threshold = t
            with torch.no_grad():
                out = model(batch)
            cache[t] = (
                out["pred_masks"].cpu().numpy().astype(bool),
                out["pred_scores"].cpu().numpy().astype(np.float32),
                out["pred_classes"].cpu().numpy().astype(np.int32),
            )

        arm_msgs = []
        for arm, t, do_split in ARMS:
            masks, scores, classes = (a.copy() for a in cache[t])
            if do_split:
                masks, scores, classes, st = split_disconnected_masks(
                    coord, masks, scores, classes, radius=args.split_radius)
                arm_msgs.append(f"{arm}:{len(scores)}m(+{st['n_out']-st['n_in']}"
                                f"/{st['n_split']}spl)")
            else:
                arm_msgs.append(f"{arm}:{len(scores)}m")
            inst_id = flatten_masks(coord, masks, scores)
            save_npz(os.path.join(args.out_dir, arm, f"{scene}_pred.npz"),
                     coord, rgb, inst_id, classes, scores, class_names, scene, arm)

        # pseudo-GT (written once)
        gt_inst = batch["instance"].cpu().numpy().astype(np.int64)
        gt_seg = batch["segment"].cpu().numpy().astype(np.int64)
        ids = np.unique(gt_inst[gt_inst >= 0])
        gt_id = np.full(coord.shape[0], -1, dtype=np.int32)
        gt_cls, gt_score = [], []
        for k, iid in enumerate(ids):
            m = gt_inst == iid
            gt_id[m] = k
            segs = gt_seg[m]
            segs = segs[segs >= 0]
            gt_cls.append(int(np.bincount(segs).argmax()) if len(segs) else 0)
            gt_score.append(1.0)
        save_npz(os.path.join(args.out_dir, "pseudo-gt", f"{scene}_pred.npz"),
                 coord, rgb, gt_id,
                 np.array(gt_cls, np.int32), np.array(gt_score, np.float32),
                 class_names, scene, "pseudo-gt")

        print(f"[{si + 1}/{len(scenes)}] {scene} {coord.shape[0]}pts | "
              f"gt {len(ids)} | " + " ".join(arm_msgs))


if __name__ == "__main__":
    main()
