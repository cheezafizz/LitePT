"""
GT-free instance-segmentation inference on a single preprocessed scene directory
(coord.npy / color.npy / normal.npy), writing a predicted-instance GLB.

Works on any scene dir in the LitePT/ScanNet schema -- a `tools/vggt_to_scene.py`
output, or an existing `data/.../<split>/<scene>` (its GT is simply ignored). It
reuses the exact prediction path from
`tools/reeval_insseg_cluster_sweep.py` (`_forward_backbone` + `_cluster`, with the
`pointops.knn_query` remap back to the full cloud) and the GLB helpers from
`engines/hooks/insseg_viz.py`. No ground-truth labels are needed: the forward pass
+ clustering depend only on coord / grid_coord / feat / offset.

Run with the litept env, fp32 (NOT bf16 -- spconv autotuner crashes in bf16 eval):
  /home/fai/miniconda3/envs/litept/bin/python tools/infer_insseg.py \
      --scene-dir data/scannet-v1.1.1-2of3-vggt/test/scene_vggt0 \
      --out /tmp/scene_vggt0_pred.glb --save-rgb

The TRAINING config (--config-file) builds the backbone/heads and selects the checkpoint;
how the per-point predictions are grouped into instances can be read from a SEPARATE
clustering config (--cluster-config), e.g. configs/scannet-v1.1.1-2of3/clustering/*.py.
This decouples "what the network is" from "how predictions are clustered":
  /home/fai/miniconda3/envs/litept/bin/python tools/infer_insseg.py \
      --config-file exp/.../config.py --weight exp/.../model/model_best.pth \
      --cluster-config configs/scannet-v1.1.1-2of3/clustering/embed-maskclust-allviews.py \
      --scene-dir data/scannet-v1.1.1-2of3-vggt/test/scene_vggt0 --out /tmp/pred.glb
Precedence: training-config model defaults < --cluster-config < the individual CLI
overrides (--cluster-thresh, ..., --no-mask-constraint).
"""
import argparse
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pointops  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.env import set_seed  # noqa: E402
from datasets.transform import Compose  # noqa: E402
from datasets.utils import collate_fn  # noqa: E402
from models import build_model  # noqa: E402
from engines.hooks.insseg_viz import (  # noqa: E402
    assign_instance_ids,
    colorize_predicted_instances,
    instance_palette,
    save_pointcloud_glb,
    voxel_downsample_indices,
)

_DEFAULT_EXP = os.path.join(
    _REPO_ROOT, "exp", "scannet-v1.1.1-2of3",
    "v1.1.1-2of3-ep1600-eval1600-lr6e-3-rigidaug-cluster-5mm",
)


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
    return model


# Inference-time clustering attributes the model exposes and a --cluster-config may set.
# These mirror the PG-v1m2 post-processing knobs read in _cluster() -- geometric
# ball-query+BFS, the embedding-space split, and the 2D-mask cannot-link. Training-only
# params (embed_delta_*/loss/reg) and the architectural embedding_dim are intentionally
# excluded: they are fixed by the checkpoint, not by inference-time clustering.
CLUSTER_KEYS = (
    "voxel_size",
    "cluster_thresh",
    "cluster_closed_points",
    "cluster_propose_points",
    "cluster_min_points",
    "segment_ignore_index",
    "instance_embedding",
    "embed_bandwidth",
    "embed_min_points",
    "cluster_mask_constraint",
    "mask_constraint_multiview",
    "mask_constraint_strict",
    "mask_split_radius",
    "cluster_mask_filter",
    "cluster_mask_rag_merge",
    "rag_merge_thresh",
    "rag_adjacency_radius",
    "rag_sim_metric",
)


def apply_cluster_config(model, cluster_config_path):
    """Override the model's inference-time clustering attributes from a SEPARATE
    clustering config (a .py file with a top-level ``cluster = dict(...)`` of the
    keys in CLUSTER_KEYS). Returns the resolved cluster dict for logging.

    Unknown keys raise (catch typos loudly). Requesting the embedding head while the
    checkpoint/train-config built none raises a clear error; ignoring a trained head
    only warns. The mask cannot-link knobs are pure inference flags with no checkpoint
    dependency -- they are consumed downstream by the mask-label block / _cluster().
    """
    cfg = Config.fromfile(cluster_config_path)
    if "cluster" not in cfg:
        raise SystemExit(
            f"--cluster-config {cluster_config_path} must define a top-level "
            f"`cluster = dict(...)`")
    cluster = dict(cfg.cluster)
    unknown = set(cluster) - set(CLUSTER_KEYS)
    if unknown:
        raise SystemExit(
            f"--cluster-config {cluster_config_path}: unknown clustering key(s) "
            f"{sorted(unknown)}; allowed keys are {list(CLUSTER_KEYS)}")
    for k, v in cluster.items():
        setattr(model, k, v)
    # Validate the embedding head: it exists only if the checkpoint was trained with
    # instance_embedding=True (PG-v1m2 builds self.embedding_head in __init__).
    if getattr(model, "instance_embedding", False):
        if not hasattr(model, "embedding_head"):
            raise SystemExit(
                f"--cluster-config {cluster_config_path} sets instance_embedding=True "
                f"but the checkpoint/train-config has no embedding head; either use a "
                f"checkpoint trained with instance_embedding=True or a clustering config "
                f"with instance_embedding=False (e.g. clustering/geometric.py).")
    elif hasattr(model, "embedding_head"):
        print("[cluster] WARN checkpoint has an embedding head but the clustering config "
              "sets instance_embedding=False; the trained embedding head will be ignored.")
    return cluster


def build_infer_transform(grid_size):
    """Minimal GT-free pipeline: the val pipeline minus Copy-of-labels / InstanceParser.

    Keeps origin_coord (full-res, pre-GridSample) so predictions can be mapped back
    onto the input cloud for visualization -- mirrors the val transform at
    configs/.../insseg-litept-small-v1m2-2of3-rigidaug.py:179-221.
    """
    return Compose([
        dict(type="CenterShift", apply_z=True),
        dict(type="Copy", keys_dict={"coord": "origin_coord"}),
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
             mode="train", return_grid_coord=True),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ToTensor"),
        dict(type="Collect",
             keys=("coord", "grid_coord", "origin_coord", "name"),
             feat_keys=("color", "normal"),
             offset_keys_dict=dict(offset="coord", origin_offset="origin_coord")),
    ])


def load_scene(scene_dir):
    def _req(name):
        p = os.path.join(scene_dir, name)
        if not os.path.isfile(p):
            raise SystemExit(f"missing required {name} in {scene_dir}")
        return np.load(p)

    coord = _req("coord.npy").astype(np.float32)
    color = _req("color.npy").astype(np.float32)   # 0-255; NormalizeColor divides by 255
    normal = _req("normal.npy").astype(np.float32)
    if not (coord.shape[0] == color.shape[0] == normal.shape[0]):
        raise SystemExit(
            f"point-count mismatch: coord {coord.shape} color {color.shape} normal {normal.shape}")
    return coord, color, normal


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", required=True,
                    help="dir with coord.npy / color.npy / normal.npy")
    ap.add_argument("--config-file", default=os.path.join(_DEFAULT_EXP, "config.py"))
    ap.add_argument("--weight", default=os.path.join(_DEFAULT_EXP, "model", "model_best.pth"))
    ap.add_argument("--out", default=None, help="output GLB (default /tmp/<scene>_pred.glb)")
    ap.add_argument("--save-rgb", action="store_true",
                    help="also write an RGB GLB of the input cloud next to --out")
    ap.add_argument("--save-npz", nargs="?", const="", default=None,
                    help="also write a structured .npz result for the viser viewer; "
                         "optional PATH (default: sibling of --out, '<scene>.npz')")
    ap.add_argument("--model-tag", default=None,
                    help="model label stored in the .npz (default: parent dir of --out)")
    ap.add_argument("--rgb-voxel", type=float, default=0.005,
                    help="voxel (m) for the RGB GLB downsample (<=0 = full-res)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster-config", default=None,
                    help="separate clustering config (.py with a top-level `cluster=dict(...)`) "
                         "overriding the train-config's inference-time clustering knobs; see "
                         "configs/scannet-v1.1.1-2of3/clustering/*.py")
    # optional clustering overrides (default: use the config / checkpoint values)
    ap.add_argument("--voxel-size", type=float, default=None)
    ap.add_argument("--cluster-thresh", type=float, default=None)
    ap.add_argument("--cluster-closed-points", type=int, default=None)
    ap.add_argument("--cluster-propose-points", type=int, default=None)
    ap.add_argument("--cluster-min-points", type=int, default=None)
    ap.add_argument("--no-mask-constraint", action="store_true",
                    help="ignore <scene-dir>/mask_instance.npy even if present "
                         "(A/B baseline vs the 2D-mask-constrained run)")
    ap.add_argument("--no-mask-filter", action="store_true",
                    help="disable the pre-clustering 2D-mask filter even if the config "
                         "enables cluster_mask_filter (A/B baseline vs the filtered run)")
    ap.add_argument("--highlight-split-cause", action="store_true",
                    help="also write <out>_splitcause.glb: the cannot-link SEAM points that "
                         "split touching objects are painted red, the carved-off fragment "
                         "instances stay vivid, and all other assigned points are dimmed. "
                         "Needs a mask cannot-link cluster-config (else nothing to highlight); "
                         "adds split_cause/split_fragment bool fields to --save-npz.")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = Config.fromfile(args.config_file)
    names = cfg.data.names
    scene_name = os.path.basename(os.path.normpath(args.scene_dir))
    out_path = args.out or os.path.join("/tmp", f"{scene_name}_pred.glb")

    # --- model ---------------------------------------------------------------
    model = build_model(cfg.model)
    model = load_weight(model, args.weight)
    model = model.cuda().eval()  # fp32; no autocast (bf16 breaks spconv autotuner)
    # Clustering precedence: train-config model defaults < --cluster-config < per-param
    # CLI flags. The cluster config is applied first so individual flags still win.
    if args.cluster_config is not None:
        apply_cluster_config(model, args.cluster_config)
        print(f"[cluster] applied cluster-config {args.cluster_config}")
    for attr, val in [("voxel_size", args.voxel_size),
                      ("cluster_thresh", args.cluster_thresh),
                      ("cluster_closed_points", args.cluster_closed_points),
                      ("cluster_propose_points", args.cluster_propose_points),
                      ("cluster_min_points", args.cluster_min_points)]:
        if val is not None:
            setattr(model, attr, val)
    # CLI A/B kill-switches: clear the model attribute itself so _cluster truly skips the
    # behavior even when per-point labels are loaded for the OTHER mask mechanism (labels
    # are shared, so suppressing only the tool-side flag would leave the model still acting).
    if args.no_mask_constraint:
        model.cluster_mask_constraint = False
    if args.no_mask_filter:
        model.cluster_mask_filter = False
    if args.highlight_split_cause:
        model.record_split_cause = True
    print(f"[cluster] voxel_size={model.voxel_size} thresh={model.cluster_thresh} "
          f"(radius={model.cluster_thresh*model.voxel_size*1000:.1f} mm) "
          f"closed={model.cluster_closed_points} propose={model.cluster_propose_points} "
          f"min={model.cluster_min_points}")
    print(f"[cluster] instance_embedding={getattr(model, 'instance_embedding', False)} "
          f"(bandwidth={getattr(model, 'embed_bandwidth', None)} "
          f"embed_min={getattr(model, 'embed_min_points', None)}) | "
          f"mask_constraint={getattr(model, 'cluster_mask_constraint', False)} "
          f"multiview={getattr(model, 'mask_constraint_multiview', False)} "
          f"split_radius={getattr(model, 'mask_split_radius', None)} | "
          f"mask_filter={getattr(model, 'cluster_mask_filter', False)}")
    print(f"[cluster] rag_merge={getattr(model, 'cluster_mask_rag_merge', False)} "
          f"(thresh={getattr(model, 'rag_merge_thresh', None)} "
          f"adj_radius={getattr(model, 'rag_adjacency_radius', None)} "
          f"metric={getattr(model, 'rag_sim_metric', None)})")

    # --- data ----------------------------------------------------------------
    coord_raw, color_raw, normal = load_scene(args.scene_dir)
    print(f"[scene] {scene_name}: {coord_raw.shape[0]:,} points | "
          f"extent(m)={(coord_raw.max(0) - coord_raw.min(0)).round(3).tolist()}")
    transform = build_infer_transform(grid_size=model.voxel_size)
    data_dict = dict(coord=coord_raw.copy(), color=color_raw.copy(),
                     normal=normal.copy(), name=scene_name)
    input_dict = collate_fn([transform(data_dict)])
    for k in list(input_dict.keys()):
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = input_dict[k].cuda(non_blocking=True)

    # --- optional per-point 2D-mask labels (mask-constrained clustering) ------
    # When the model enables cluster_mask_constraint and the scene carries baked mask
    # labels (origin-resolution, from tools/vggt_to_scene.py --coco), remap them to the
    # grid cloud and feed _cluster so two points in different 2D masks of the same view
    # are never grouped together. No-op otherwise.
    #   * Multi-view: mask_per_view.npy (N,S) + model.mask_constraint_multiview -> the
    #     cannot-link fires across ALL views (each point reprojected with z-buffer).
    #   * Single-view (legacy): mask_instance.npy / mask_view.npy -> origin-view only.
    point_view = point_mask = point_mask_views = None
    origin_has_label = None  # (n_origin,) bool: origin point falls in >=1 2D mask
    constraint_on = (getattr(model, "cluster_mask_constraint", False)
                     and not args.no_mask_constraint)
    filter_on = (getattr(model, "cluster_mask_filter", False)
                 and not args.no_mask_filter)
    # Mask labels are loaded once and serve BOTH the cannot-link constraint and the
    # pre-clustering filter; the model gates which actually fires.
    need_masks = constraint_on or filter_on
    mask_path = os.path.join(args.scene_dir, "mask_instance.npy")
    mask_views_path = os.path.join(args.scene_dir, "mask_per_view.npy")
    n_origin = int(input_dict["origin_coord"].shape[0])

    def _origin_to_grid_idx():
        """For each grid point, the index of its nearest origin point (long, on cpu)."""
        idx_o, _ = pointops.knn_query(
            1, input_dict["origin_coord"].float(), input_dict["origin_offset"].int(),
            input_dict["coord"].float(), input_dict["offset"].int())
        return idx_o.cpu().flatten().long()

    if (need_masks and getattr(model, "mask_constraint_multiview", False)
            and os.path.isfile(mask_views_path)):
        mask_per_view = np.load(mask_views_path).astype(np.int64)  # (N_origin, S)
        if mask_per_view.ndim != 2 or mask_per_view.shape[0] != n_origin:
            print(f"[mask] WARN mask_per_view {mask_per_view.shape} != (origin pts "
                  f"{n_origin}, S); skipping mask labels")
        else:
            point_mask_views = torch.from_numpy(mask_per_view)[_origin_to_grid_idx()]
            origin_has_label = (mask_per_view >= 0).any(axis=1)  # origin-resolution
            n_lab = int((point_mask_views >= 0).any(dim=1).sum())
            n_inst = int(len(torch.unique(point_mask_views[point_mask_views >= 0])))
            print(f"[mask] multi-view labels (constraint={constraint_on} "
                  f"filter={filter_on}): {n_lab}/{point_mask_views.shape[0]} "
                  f"grid pts labeled in >=1 of {mask_per_view.shape[1]} views, "
                  f"{n_inst} distinct 2D instances")
    elif need_masks and os.path.isfile(mask_path):
        mask_instance = np.load(mask_path).astype(np.int64)
        view_path = os.path.join(args.scene_dir, "mask_view.npy")
        mask_view = (np.load(view_path).astype(np.int64) if os.path.isfile(view_path)
                     else np.zeros_like(mask_instance))
        if mask_instance.shape[0] != n_origin:
            print(f"[mask] WARN mask_instance ({mask_instance.shape[0]}) != origin pts "
                  f"({n_origin}); skipping mask labels")
        else:
            # each grid point inherits the (view, mask) of its nearest origin point
            idx_o = _origin_to_grid_idx()
            point_view = torch.from_numpy(mask_view)[idx_o]
            point_mask = torch.from_numpy(mask_instance)[idx_o]
            origin_has_label = mask_instance >= 0  # origin-resolution
            n_lab = int((point_mask >= 0).sum())
            n_inst = int(len(torch.unique(point_mask[point_mask >= 0])))
            print(f"[mask] single-view labels (constraint={constraint_on} "
                  f"filter={filter_on}): {n_lab}/{point_mask.numel()} grid "
                  f"pts labeled, {n_inst} distinct 2D instances")
    elif need_masks:
        print(f"[mask] no mask labels in {args.scene_dir}; "
              f"clustering without constraint/filter")

    # --- predict (no GT) -----------------------------------------------------
    # An instance-embedding model (PG-v1m2 with instance_embedding=True) splits
    # touching same-class proposals in embedding space, but ONLY when an embedding is
    # passed to _cluster -- _forward_backbone returns just bias/logit, so we must compute
    # the embedding from the backbone feat and feed it in. Base models
    # (instance_embedding=False) take the original bias/logit-only path unchanged.
    with torch.no_grad():
        if getattr(model, "instance_embedding", False):
            feat = model._backbone_feat(input_dict)
            bias_pred = model.bias_head(feat)
            logit_pred = model.seg_head(feat)
            embedding = model.embedding_head(feat)
            out = model._cluster(
                input_dict["coord"], bias_pred, logit_pred, input_dict["offset"],
                embedding=embedding, point_view=point_view, point_mask=point_mask,
                point_mask_views=point_mask_views)
        else:
            bias_pred, logit_pred = model._forward_backbone(input_dict)
            out = model._cluster(
                input_dict["coord"], bias_pred, logit_pred, input_dict["offset"],
                point_view=point_view, point_mask=point_mask,
                point_mask_views=point_mask_views)

    pred_masks = out["pred_masks"]            # (P, n_grid) int, on cpu
    pred_scores = out["pred_scores"]
    pred_classes = out["pred_classes"]
    n_pred = int(pred_masks.shape[0])

    # --- remap grid-resolution masks back to the full input cloud ------------
    origin_coord = input_dict["origin_coord"]
    idx, _ = pointops.knn_query(
        1, input_dict["coord"].float(), input_dict["offset"].int(),
        origin_coord.float(), input_dict["origin_offset"].int())
    idx = idx.cpu().flatten().long()
    origin_coord_np = origin_coord.cpu().numpy()
    if n_pred > 0:
        masks_origin = pred_masks[:, idx].bool().numpy()
    else:
        masks_origin = np.zeros((0, origin_coord_np.shape[0]), dtype=bool)

    # --- split-cause seam points (grid indices recorded inside _cluster) ----------
    # Lift the recorded grid endpoints to origin resolution via the SAME origin->grid
    # nearest-neighbor map (idx). (n_origin,) bool at full res, filtered alongside the
    # output cloud below so it stays index-aligned with origin_coord_np / masks_origin.
    seam_origin = None
    if args.highlight_split_cause:
        cause_chunks = getattr(model, "_split_cause_idx", None) or []
        cause_grid = (np.unique(np.concatenate(cause_chunks)) if cause_chunks
                      else np.empty(0, dtype=np.int64))
        seam_origin = np.isin(idx.numpy(), cause_grid)
        print(f"[splitcause] {cause_grid.size} grid seam pts -> "
              f"{int(seam_origin.sum())} origin seam pts")

    # --- pre-clustering 2D-mask filter: keep only masked points in the output ----
    # The model already restricted clustering to points inside some 2D mask; here we
    # also drop the unmasked points from the output cloud entirely so only masked
    # points remain (origin-resolution labels, index-aligned with coord/color_raw).
    if filter_on and origin_has_label is not None:
        keep = origin_has_label  # (n_origin,) bool
        n_before = origin_coord_np.shape[0]
        if keep.shape[0] == n_before:
            origin_coord_np = origin_coord_np[keep]
            masks_origin = masks_origin[:, keep]
            if seam_origin is not None and seam_origin.shape[0] == keep.shape[0]:
                seam_origin = seam_origin[keep]
            if coord_raw.shape[0] == keep.shape[0]:
                coord_raw = coord_raw[keep]
                color_raw = color_raw[keep]
            print(f"[mask-filter] kept {origin_coord_np.shape[0]:,}/{n_before:,} "
                  f"masked points in output")
        else:
            print(f"[mask-filter] WARN origin_has_label ({keep.shape[0]}) != origin pts "
                  f"({n_before}); output not filtered")
    elif filter_on:
        print("[mask-filter] enabled but no per-point 2D-mask labels; output unchanged")

    # --- report --------------------------------------------------------------
    print(f"[pred] {n_pred} instance proposals")
    for pid in range(n_pred):
        cls = int(pred_classes[pid].item())
        cname = names[cls] if cls < len(names) else str(cls)
        print(f"   #{pid:03d} class={cname:8s} score={float(pred_scores[pid]):.3f} "
              f"pts={int(masks_origin[pid].sum()):,}")

    # --- visualize -----------------------------------------------------------
    colors = colorize_predicted_instances(
        origin_coord_np.shape[0], masks_origin,
        pred_scores.numpy() if n_pred > 0 else np.zeros(0))
    save_pointcloud_glb(out_path, origin_coord_np, colors)
    print(f"[viz] wrote {out_path}  (gray = unassigned)")

    # --- split-cause highlight: seam (red) + carved-off fragments (vivid) ----
    # The fragments are exactly the final instances that own >=1 seam point -- each was
    # separated from a touching neighbor by a 2D-mask cannot-link edge. Everything else
    # assigned is dimmed so the over-segmented pieces and their seams stand out.
    if args.highlight_split_cause and seam_origin is not None:
        n_pts = origin_coord_np.shape[0]
        pred_scores_np = pred_scores.numpy() if n_pred > 0 else np.zeros(0, np.float32)
        if seam_origin.shape[0] != n_pts:
            print(f"[splitcause] WARN seam mask ({seam_origin.shape[0]}) != pts ({n_pts}); "
                  f"skipping highlight GLB")
        else:
            inst_id = assign_instance_ids(n_pts, masks_origin, pred_scores_np)
            frag_ids = np.unique(inst_id[seam_origin & (inst_id >= 0)])
            frag_origin = np.isin(inst_id, frag_ids)
            hl = colorize_predicted_instances(n_pts, masks_origin, pred_scores_np)
            dim = (~frag_origin) & (inst_id >= 0)  # assigned but not a fragment -> mute
            hl[dim] = (hl[dim].astype(np.float32) * 0.35 + 160 * 0.65).astype(np.uint8)
            hl[seam_origin] = np.array([255, 0, 0], dtype=np.uint8)  # seam drawn on top
            hl_path = out_path.replace(".glb", "_splitcause.glb")
            save_pointcloud_glb(hl_path, origin_coord_np, hl)
            print(f"[splitcause] {int(seam_origin.sum())} seam pts, {frag_ids.size} "
                  f"fragment instances -> wrote {hl_path}")

    # --- structured result for the viser viewer ------------------------------
    if args.save_npz is not None:
        npz_path = args.save_npz or (out_path[:-4] if out_path.endswith(".glb")
                                     else out_path) + ".npz"
        n_pts = origin_coord_np.shape[0]
        pred_scores_np = pred_scores.numpy() if n_pred > 0 else np.zeros(0, np.float32)
        pred_classes_np = (pred_classes.numpy() if n_pred > 0
                           else np.zeros(0, np.int32)).astype(np.int32)
        inst_id = assign_instance_ids(n_pts, masks_origin, pred_scores_np)
        inst_color = instance_palette(n_pred)  # (P,3) uint8; matches GLB colors
        # color_raw is the full-res input color, index-aligned to origin_coord_np
        if color_raw.shape[0] == n_pts:
            rgb = np.clip(color_raw, 0, 255).astype(np.uint8)
        else:
            print(f"[npz] WARN color/coord count mismatch "
                  f"({color_raw.shape[0]} vs {n_pts}); rgb set to gray")
            rgb = np.full((n_pts, 3), 160, dtype=np.uint8)
        model_tag = args.model_tag or os.path.basename(os.path.dirname(os.path.abspath(out_path)))
        os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
        # optional split-cause overlay fields for the viser viewer (per-point bool)
        extra = {}
        if (args.highlight_split_cause and seam_origin is not None
                and seam_origin.shape[0] == n_pts):
            frag_ids = np.unique(inst_id[seam_origin & (inst_id >= 0)])
            extra["split_cause"] = seam_origin
            extra["split_fragment"] = np.isin(inst_id, frag_ids)
        np.savez_compressed(
            npz_path,
            coord=origin_coord_np.astype(np.float32),
            rgb=rgb,
            inst_id=inst_id,
            inst_class=pred_classes_np,
            inst_score=pred_scores_np.astype(np.float32),
            inst_color=inst_color,
            class_names=np.asarray([str(x) for x in names]),
            scene=scene_name,
            model=model_tag,
            **extra,
        )
        print(f"[npz] wrote {npz_path}  ({n_pts:,} pts, {n_pred} instances, model={model_tag})")

    if args.save_rgb:
        rgb_path = out_path.replace(".glb", "_rgb.glb")
        if args.rgb_voxel > 0:
            kept = voxel_downsample_indices(coord_raw, args.rgb_voxel)
        else:
            kept = np.arange(coord_raw.shape[0])
        save_pointcloud_glb(rgb_path, coord_raw[kept], color_raw[kept].astype(np.uint8))
        print(f"[viz] wrote {rgb_path}")


if __name__ == "__main__":
    main()
