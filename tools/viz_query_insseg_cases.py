"""
Visualize query-InsSeg (MQ-v1m1) predictions AND pseudo-GT for hand-picked
train/val scenes, dumping .npz files consumable by tools/viser_insseg_viewer.py.

Like tools/viz_query_insseg_val.py, but:
  * takes explicit --val-scenes / --train-scenes lists (train scenes are viewed
    through the deterministic val pipeline via a split override);
  * additionally writes each scene's pseudo-GT instances (the post-pipeline
    `instance` / `segment` labels) as a second "model" dir ("pseudo-gt"), so the
    viewer's Model dropdown flips between prediction and pseudo-GT.

  /home/fai/miniconda3/envs/litept/bin/python tools/viz_query_insseg_cases.py \
      --exp-dir exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-query-realft-muon-nosem \
      --val-scenes A B --train-scenes C D --out-dir viz/query_nosem_cases
  .venv-viser/bin/python tools/viser_insseg_viewer.py \
      --results-dir viz/query_nosem_cases --port 8080
"""
import argparse
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
from tools.postprocess_insseg import split_disconnected_masks  # noqa: E402


def distinct_palette(n):
    """Deterministic distinct RGB palette built by greedy farthest-point
    selection over an HSV candidate grid, so even ~50 entries keep a large
    worst-pair color distance (stays clear of gray, which marks unassigned)."""
    import colorsys
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    cand = np.array([colorsys.hsv_to_rgb(h, s, v)
                     for h in np.linspace(0.0, 1.0, 48, endpoint=False)
                     for s in (0.45, 0.75, 1.0)
                     for v in (0.55, 0.8, 1.0)], dtype=np.float32) * 255.0
    # drop candidates near the viewer's "unassigned" gray (160,160,160)
    cand = cand[np.linalg.norm(cand - 160.0, axis=1) > 60.0]
    # weight green channel up (eye is most luminance-sensitive there)
    w = np.array([1.0, 1.6, 0.8], dtype=np.float32)
    picked = [int(np.argmax(cand.sum(1)))]  # start from the brightest color
    d = np.linalg.norm((cand - cand[picked[0]]) * w, axis=1)
    while len(picked) < min(n, len(cand)):
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm((cand - cand[nxt]) * w, axis=1))
    cols = cand[np.array(picked)]
    if n > len(cols):  # absurdly many instances: cycle
        cols = cols[np.arange(n) % len(cols)]
    return cols.astype(np.uint8)


def save_npz(path, coord, rgb, inst_id, classes, scores, class_names, scene, model):
    np.savez_compressed(
        path, coord=coord, rgb=rgb, inst_id=inst_id,
        inst_class=classes, inst_score=scores,
        inst_color=distinct_palette(len(scores)),
        class_names=np.array(class_names), scene=scene, model=model,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--val-scenes", nargs="*", default=[])
    ap.add_argument("--train-scenes", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-split-disconnected", action="store_true",
                    help="skip splitting spatially-disconnected components out "
                         "of each predicted mask (on by default)")
    ap.add_argument("--split-radius", type=float, default=0.005,
                    help="connectivity radius in meters for mask splitting")
    args = ap.parse_args()

    cfg = Config.fromfile(os.path.join(args.exp_dir, "config.py"))
    set_seed(args.seed)
    weight = args.weight or os.path.join(args.exp_dir, "model", "model_best.pth")
    model_name = args.model_name or os.path.basename(os.path.normpath(args.exp_dir))
    class_names = list(cfg.data.names)

    model = build_model(cfg.model).cuda().eval()
    load_weight(model, weight)

    pred_dir = os.path.join(args.out_dir, model_name)
    gt_dir = os.path.join(args.out_dir, "pseudo-gt")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for split, scene_list in [("val", args.val_scenes), ("train", args.train_scenes)]:
        if not scene_list:
            continue
        data_cfg = cfg.data.val
        data_cfg.split = split
        dataset = build_dataset(data_cfg)
        names = [dataset.get_data_name(i) for i in range(len(dataset))]
        for scene in scene_list:
            i = names.index(scene)
            batch = collate_fn([dataset[i]])
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda(non_blocking=True)
            with torch.no_grad():
                out = model(batch)

            coord = batch["coord"].cpu().numpy().astype(np.float32)
            rgb = ((batch["feat"][:, :3].cpu().numpy() + 1.0) * 127.5)
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            # --- prediction ---------------------------------------------------
            masks = out["pred_masks"].cpu().numpy().astype(bool)
            scores = out["pred_scores"].cpu().numpy().astype(np.float32)
            classes = out["pred_classes"].cpu().numpy().astype(np.int32)
            split_stats = None
            if not args.no_split_disconnected:
                masks, scores, classes, split_stats = split_disconnected_masks(
                    coord, masks, scores, classes, radius=args.split_radius)
            inst_id = np.full(coord.shape[0], -1, dtype=np.int32)
            for m in np.argsort(scores):  # higher score wins overlaps
                inst_id[masks[m]] = m
            save_npz(os.path.join(pred_dir, f"{scene}_pred.npz"),
                     coord, rgb, inst_id, classes, scores, class_names,
                     scene, model_name)

            # --- pseudo-GT ----------------------------------------------------
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
            save_npz(os.path.join(gt_dir, f"{scene}_pred.npz"),
                     coord, rgb, gt_id,
                     np.array(gt_cls, np.int32), np.array(gt_score, np.float32),
                     class_names, scene, "pseudo-gt")

            n_conf = int((scores >= 0.5).sum())
            split_msg = ""
            if split_stats:
                split_msg = (f" | split {split_stats['n_split']} masks -> "
                             f"+{split_stats['n_out'] - split_stats['n_in']} frags "
                             f"({split_stats['n_dropped']} tiny dropped)")
            print(f"[{split}:{scene}] {coord.shape[0]} pts | "
                  f"pred {len(scores)} masks ({n_conf} conf>=0.5) | "
                  f"gt {len(ids)} instances{split_msg}")


if __name__ == "__main__":
    main()
