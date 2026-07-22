"""
Density-robust comparison of VGGT surface-normal quality *before vs after* applying the
stored validity masks, and across an additional confidence floor — to decide whether
`tools/vggt_to_scene.py` should apply `filtered_valid_mask` and whether a higher
confidence threshold buys cleaner normals.

READ-ONLY: loads each VGGT `data.npz`, rebuilds the fused -> voxel-downsampled ->
camera-oriented-normal cloud IN MEMORY under several masking configs, prints metrics,
and writes nothing. It reuses `compute_normals` / `voxel_first_hit` from the v1_0_1
preprocessing (same helpers `vggt_to_scene.py` uses) so the geometry matches production.

WHY fixed-radius metrics (not the fixed-k metric in analyze_normal_gap.py):
  The masks here drop 30-90% of points. A fixed-k neighborhood then spans a *larger*
  physical area on a sparser cloud, which inflates angular spread and confounds the
  before/after comparison (a mask can look "worse" purely from lower density). All
  metrics below use a FIXED spatial radius, so configs are compared at the same scale.

Metrics, each the median over <= --max-points sampled query points, at each --radii r:
  - angular consistency (deg): mean angle between a point's normal and the normals of its
    neighbors within r. Lower = locally agreeing normals.
  - surface noise (x1000): smallest-eigenvalue fraction of the radius-neighborhood
    covariance (lambda0 / sum(lambda)). A density-tolerant planarity/thickness measure;
    lower = flatter/thinner surface = less reconstruction noise.
  - retention: final voxelized point count and % vs the `none` (finite-only) build.

Configs (override with --configs): none, valid, filtered, then `filtered` plus a
confidence-percentile sweep evaluated within the filtered region (filtered+p25/50/75).

Example:
  /home/fai/miniconda3/envs/litept/bin/python tools/analyze_mask_normal_quality.py \
      --input-glob '/home/fai/workspace/jhp/dataset/val_sample_output/*/*/data.npz'
"""
import argparse
import glob
import importlib.util
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_preprocess_helpers():
    """Import compute_normals / voxel_first_hit from the v1_0_1 preprocessing script.

    That directory is not a package (no __init__.py), so load it by file path — the same
    pattern tools/vggt_to_scene.py uses, so both build identical geometry.
    """
    path = os.path.join(
        _REPO_ROOT, "datasets", "preprocessing", "v1_0_1", "preprocess_v1_0_1.py"
    )
    spec = importlib.util.spec_from_file_location("preprocess_v1_0_1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_normals, mod.voxel_first_hit


# config name -> (stored mask key or None, conf percentile within kept region)
CONFIGS = {
    "none": (None, 0.0),
    "valid": ("valid_mask", 0.0),
    "filtered": ("filtered_valid_mask", 0.0),
    "filtered+p25": ("filtered_valid_mask", 25.0),
    "filtered+p50": ("filtered_valid_mask", 50.0),
    "filtered+p75": ("filtered_valid_mask", 75.0),
}
DEFAULT_ORDER = list(CONFIGS)


def camera_origins(extrinsic):
    """(S,3,4)/(S,4,4) world->cam extrinsic -> (S,3) camera centers C = -R^T t."""
    E = np.asarray(extrinsic, dtype=np.float64)
    R = E[:, :3, :3]
    t = E[:, :3, 3]
    return np.einsum("sij,sj->si", np.transpose(R, (0, 2, 1)), -t).astype(np.float32)


def build_cloud(store, mask_key, conf_pct, voxel, normal_k, compute_normals, voxel_first_hit):
    """Fuse -> voxel-downsample -> camera-oriented normals under one masking config."""
    wp = np.asarray(store["world_points"], dtype=np.float32)
    rgb = np.asarray(store["rgb"])
    conf = np.asarray(store["confidence"], dtype=np.float32)
    cam_origins = camera_origins(store["extrinsic"])
    S, H, W, _ = wp.shape

    region = np.isfinite(wp).all(axis=-1)
    if mask_key is not None:
        region = region & np.asarray(store[mask_key]).astype(bool)
    conf_thr = float(np.percentile(conf[region], conf_pct)) if conf_pct > 0 and region.any() else 0.0

    coords, colors, cam_idxs = [], [], []
    for s in range(S):
        v = region[s] & (conf[s] >= conf_thr)
        if not v.any():
            continue
        coords.append(wp[s][v])
        colors.append(rgb[s][v])
        cam_idxs.append(np.full(int(v.sum()), s, dtype=np.int32))
    if not coords:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    coord = np.concatenate(coords).astype(np.float32)
    color = np.concatenate(colors).astype(np.uint8)
    cam_idx = np.concatenate(cam_idxs).astype(np.int32)

    dummy = np.zeros(coord.shape[0], dtype=np.int64)
    coord, color, _c, _a, cam_idx = voxel_first_hit(coord, color, dummy, dummy, cam_idx, voxel)
    normal = compute_normals(coord, cam_origins, cam_idx, k=normal_k)
    return coord, normal


def _smallest_eig_frac(cov):
    """Vectorized lambda_min / trace for a stack of symmetric 3x3 matrices (Q,3,3).

    Closed-form (Wikipedia 3x3-symmetric eigenvalue algorithm) instead of
    np.linalg.eigvalsh: LAPACK eigvalsh hits an intermittent numpy-2.x thread-safety
    bug ('Float64DType has no attribute single') in a hot loop after scipy's threaded
    kd-tree query. The analytic form is deterministic and faster here.
    """
    a00, a11, a22 = cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2]
    a01, a02, a12 = cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 2]
    tr = a00 + a11 + a22
    q = tr / 3.0
    p1 = a01 ** 2 + a02 ** 2 + a12 ** 2
    p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2.0 * p1
    p = np.sqrt(np.maximum(p2 / 6.0, 0.0))
    pe = np.where(p > 0, p, 1.0)
    b00, b11, b22 = (a00 - q) / pe, (a11 - q) / pe, (a22 - q) / pe
    b01, b02, b12 = a01 / pe, a02 / pe, a12 / pe
    detB = (b00 * (b11 * b22 - b12 * b12)
            - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02))
    phi = np.arccos(np.clip(detB / 2.0, -1.0, 1.0)) / 3.0
    eig_min = np.where(p > 0, q + 2.0 * p * np.cos(phi + 2.0 * np.pi / 3.0), q)
    return np.clip(eig_min / np.where(tr > 0, tr, 1.0), 0.0, None)


def radius_metrics(coord, normal, radius, max_points, seed=0):
    """Median (angular consistency deg, surface noise x1000) over a fixed-radius nbhd."""
    n = coord.shape[0]
    if n < 6:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(n, min(max_points, n), replace=False)
    tree = cKDTree(coord)
    nrm = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    nbrs = tree.query_ball_point(coord[q_idx], r=radius, workers=-1)
    angs, covs = [], []
    for qi, idx in zip(q_idx, nbrs):
        idx = [j for j in idx if j != qi]
        if len(idx) < 5:
            continue
        dots = np.abs(nrm[idx] @ nrm[qi])
        angs.append(np.degrees(np.arccos(np.clip(dots, 0.0, 1.0))).mean())
        P = coord[idx].astype(np.float64)
        P = P - P.mean(axis=0)
        covs.append((P.T @ P) / len(idx))
    if not angs:
        return float("nan"), float("nan")
    noise = _smallest_eig_frac(np.stack(covs))
    return float(np.median(angs)), float(np.median(noise) * 1e3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-glob",
                    default="/home/fai/workspace/jhp/dataset/val_sample_output/*/*/data.npz",
                    help="glob of VGGT data.npz files to analyze")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_ORDER,
                    help=f"subset/order of configs to run (choices: {DEFAULT_ORDER})")
    ap.add_argument("--radii", type=float, nargs="+", default=[0.010, 0.020],
                    help="fixed neighborhood radii in meters")
    ap.add_argument("--voxel-size", type=float, default=0.002,
                    help="fuse/downsample voxel (m); match dataset build (2 mm)")
    ap.add_argument("--normal-k", type=int, default=16, help="k-NN for PCA normals")
    ap.add_argument("--max-points", type=int, default=40000,
                    help="max query points per scene for the metrics")
    ap.add_argument("--per-scene", action="store_true",
                    help="print a per-scene table (default: aggregate only)")
    args = ap.parse_args()

    for c in args.configs:
        if c not in CONFIGS:
            raise SystemExit(f"unknown config {c!r}; choices: {DEFAULT_ORDER}")
    compute_normals, voxel_first_hit = _load_preprocess_helpers()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"no files matched {args.input_glob!r}")
    print(f"[scan] {len(files)} scenes | configs={args.configs} | "
          f"radii(mm)={[int(r * 1000) for r in args.radii]} | voxel={args.voxel_size * 1000:.1f}mm\n")

    # accumulator: config -> metric name -> list across scenes
    acc = {c: {"ret": [], "n": [], **{f"ang@{int(r*1000)}": [] for r in args.radii},
               **{f"noise@{int(r*1000)}": [] for r in args.radii}} for c in args.configs}

    hdr_metrics = "".join(f" {('ang@'+str(int(r*1000))):>8s} {('noise@'+str(int(r*1000))):>10s}"
                          for r in args.radii)
    for f in files:
        name = "/".join(f.split(os.sep)[-3:-1])
        store = np.load(f, allow_pickle=True)
        store = {k: store[k] for k in store.files}
        base_n = None
        if args.per_scene:
            print(f"== {name} ==")
            print(f"  {'config':14s} {'Npts':>9s} {'ret':>6s}{hdr_metrics}")
        for c in args.configs:
            mask_key, conf_pct = CONFIGS[c]
            coord, normal = build_cloud(store, mask_key, conf_pct, args.voxel_size,
                                        args.normal_k, compute_normals, voxel_first_hit)
            n = coord.shape[0]
            if c == "none":
                base_n = n
            ret = (100.0 * n / base_n) if base_n else float("nan")
            acc[c]["n"].append(n)
            acc[c]["ret"].append(ret)
            cells = ""
            for r in args.radii:
                ang, noise = radius_metrics(coord, normal, r, args.max_points)
                acc[c][f"ang@{int(r*1000)}"].append(ang)
                acc[c][f"noise@{int(r*1000)}"].append(noise)
                cells += f" {ang:8.2f} {noise:10.3f}"
            if args.per_scene:
                print(f"  {c:14s} {n:>9,d} {ret:5.1f}%{cells}")
        if args.per_scene:
            print()

    print(f"=== aggregate across {len(files)} scenes (median of per-scene values) ===")
    print(f"  {'config':14s} {'med Npts':>9s} {'ret':>6s}{hdr_metrics}")
    for c in args.configs:
        med_n = int(np.median(acc[c]["n"]))
        med_ret = float(np.median(acc[c]["ret"]))
        cells = ""
        for r in args.radii:
            cells += (f" {np.nanmedian(acc[c][f'ang@{int(r*1000)}']):8.2f}"
                      f" {np.nanmedian(acc[c][f'noise@{int(r*1000)}']):10.3f}")
        print(f"  {c:14s} {med_n:>9,d} {med_ret:5.1f}%{cells}")
    print("\nLower noise = flatter/cleaner surface; lower ang = locally agreeing normals.")
    print("Read quality gains against `ret` (coverage cost): aggressive conf floors clean")
    print("the surface but discard most points the InsSeg model relies on.")


if __name__ == "__main__":
    main()
