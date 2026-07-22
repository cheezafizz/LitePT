"""
Run the query-based InsSeg model (MQ-v1m1) on a sample of val scenes and dump
per-scene .npz files consumable by tools/viser_insseg_viewer.py.

Loads the TRAINING config saved in the exp dir (so the val pipeline / model are
exactly what the evaluator used), builds the val dataset, forwards each sampled
scene in eval mode (fp32 -- spconv autotuner crashes in bf16 eval), and flattens
the returned pred_masks (M, N) / pred_scores / pred_classes into a per-point
inst_id (higher-score masks win overlaps).

  /home/fai/miniconda3/envs/litept/bin/python tools/viz_query_insseg_val.py \
      --exp-dir exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-query-realft-muon \
      --num-scenes 10 --out-dir viz/query_realft_muon_val
  .venv-viser/bin/python tools/viser_insseg_viewer.py \
      --results-dir viz/query_realft_muon_val --port 8080
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
from engines.hooks.insseg_viz import instance_palette  # noqa: E402
from tools.postprocess_insseg import split_disconnected_masks  # noqa: E402


def load_weight(model, weight_path):
    # weights_only=False: our checkpoints carry numpy scalars the >=2.6 unpickler rejects.
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cleaned = {(k[len("module."):] if k.startswith("module.") else k): v
               for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"[ckpt] {weight_path} | epoch={epoch} "
          f"| missing={len(missing)} unexpected={len(unexpected)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--weight", default=None,
                    help="default: <exp-dir>/model/model_best.pth")
    ap.add_argument("--num-scenes", type=int, default=10)
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="explicit val scene names (override --num-scenes sampling)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-name", default=None,
                    help="subdir / dropdown label; default: exp dir basename")
    ap.add_argument("--pool", type=int, default=None,
                    help="infer this many evenly-spaced scenes but only SAVE the "
                         "--num-scenes with the most confident (score>=0.5) masks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=None,
                    help="override the split of cfg.data.val (e.g. 'train' to view "
                         "train scenes through the deterministic val pipeline)")
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

    if args.split:
        cfg.data.val.split = args.split
    dataset = build_dataset(cfg.data.val)
    names = [dataset.get_data_name(i) for i in range(len(dataset))]
    if args.scenes:
        idxs = [names.index(s) for s in args.scenes]
    else:
        # evenly spaced sample across the sorted val list -> covers machines/seqs
        n_sample = args.pool or args.num_scenes
        idxs = np.linspace(0, len(dataset) - 1, n_sample).round().astype(int)
        idxs = sorted(set(int(i) for i in idxs))
    print(f"[data] val has {len(dataset)} scenes; inferring {len(idxs)}"
          + (f", keeping top {args.num_scenes} by confident-mask count"
             if args.pool else ""))

    model = build_model(cfg.model).cuda().eval()
    load_weight(model, weight)

    out_dir = os.path.join(args.out_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i in idxs:
        name = names[i]
        batch = collate_fn([dataset[i]])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)
        with torch.no_grad():
            out = model(batch)

        coord = batch["coord"].cpu().numpy().astype(np.float32)
        # feat = (color, normal); NormalizeColor maps color to [-1, 1]
        rgb = ((batch["feat"][:, :3].cpu().numpy() + 1.0) * 127.5)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        masks = out["pred_masks"].cpu().numpy().astype(bool)      # (M, N)
        scores = out["pred_scores"].cpu().numpy().astype(np.float32)
        classes = out["pred_classes"].cpu().numpy().astype(np.int32)
        split_stats = None
        if not args.no_split_disconnected:
            masks, scores, classes, split_stats = split_disconnected_masks(
                coord, masks, scores, classes, radius=args.split_radius)

        inst_id = np.full(coord.shape[0], -1, dtype=np.int32)
        # ascending-score order: higher-score masks overwrite on overlap
        for m in np.argsort(scores):
            inst_id[masks[m]] = m

        n_conf = int((scores >= 0.5).sum())
        kept = (inst_id >= 0).mean()
        split_msg = ""
        if split_stats:
            split_msg = (f" | split {split_stats['n_split']} masks -> "
                         f"+{split_stats['n_out'] - split_stats['n_in']} frags "
                         f"({split_stats['n_dropped']} tiny dropped)")
        print(f"[{name}] {coord.shape[0]} pts | {len(scores)} masks "
              f"({n_conf} conf>=0.5) | {kept:.1%} pts assigned{split_msg}"
              if len(scores) else f"[{name}] {coord.shape[0]} pts | 0 masks")
        results.append((n_conf, name, coord, rgb, inst_id, classes, scores))

    if args.pool:
        results.sort(key=lambda r: -r[0])
        dropped = results[args.num_scenes:]
        results = results[: args.num_scenes]
        if dropped:
            print(f"[select] kept {len(results)} scenes "
                  f"({results[-1][0]}-{results[0][0]} confident masks); "
                  f"dropped {len(dropped)}")

    for n_conf, name, coord, rgb, inst_id, classes, scores in results:
        np.savez_compressed(
            os.path.join(out_dir, f"{name}_pred.npz"),
            coord=coord, rgb=rgb, inst_id=inst_id,
            inst_class=classes, inst_score=scores,
            inst_color=instance_palette(len(scores)),
            class_names=np.array(class_names), scene=name, model=model_name,
        )


if __name__ == "__main__":
    main()
