"""
Measure the surface-normal quality gap between the synthesized training set and
VGGT-reconstructed scenes, to calibrate the normal-robustness augmentation
(`NormalJitter` / `NormalDropout`).

READ-ONLY: only loads coord.npy / normal.npy and prints statistics. Writes nothing.

Metric — *local angular consistency*: for each point, the mean angle (degrees)
between its normal and the normals of its k nearest neighbors. On a clean surface
this is small (normals agree); sensor/reconstruction noise inflates it. We compare
the per-point distribution (mean / median / p90 / p95) between the two domains.

It also runs a *calibration sweep*: it applies the candidate `NormalJitter` model
(n' = normalize(n + N(0, sigma))) to the synthetic normals at several sigmas and
reports which sigma makes the synthetic consistency distribution match VGGT — that
sigma is the value to put in the training config.

Example:
  /home/fai/miniconda3/envs/litept/bin/python tools/analyze_normal_gap.py \
      --synthetic-root data/scannet-v1.1.1-2of3/train \
      --vggt-root      data/scannet-v1.1.1-2of3-vggt/test \
      --n-synthetic 8 --max-points 60000
"""
import argparse
import glob
import os

import numpy as np
from scipy.spatial import cKDTree


def local_angle_deg(coord, normal, k=16, max_points=60000, seed=0):
    """Per-point mean angle (deg) to its k-NN's normals. Returns 1-D array."""
    n = coord.shape[0]
    if n < k + 1:
        return np.empty(0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    # KDTree is built on the full cloud (neighbor structure must be exact); we only
    # subsample the *query* points to bound runtime on large scenes.
    if n > max_points:
        q_idx = rng.choice(n, max_points, replace=False)
    else:
        q_idx = np.arange(n)
    tree = cKDTree(coord)
    nrm = normal / np.clip(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12, None)
    _, nn = tree.query(coord[q_idx], k=k + 1, workers=-1)  # col 0 is self
    nn = nn[:, 1:]
    dots = np.einsum("qi,qki->qk", nrm[q_idx], nrm[nn])  # (Q, k)
    ang = np.degrees(np.arccos(np.clip(np.abs(dots), 0.0, 1.0)))  # abs: ignore sign flips
    return ang.mean(axis=1).astype(np.float32)


def jitter_normals(normal, sigma, rng):
    """Candidate NormalJitter model: additive Gaussian then renormalize."""
    noisy = normal + rng.normal(0.0, sigma, size=normal.shape)
    return noisy / np.clip(np.linalg.norm(noisy, axis=1, keepdims=True), 1e-12, None)


def describe(name, vals):
    if vals.size == 0:
        print(f"  {name:32s} (no data)")
        return None
    q = np.percentile(vals, [50, 90, 95, 99])
    print(f"  {name:32s} mean={vals.mean():6.2f}  median={q[0]:6.2f}  "
          f"p90={q[1]:6.2f}  p95={q[2]:6.2f}  p99={q[3]:6.2f}  (deg)")
    return q[0]


def scene_dirs(root):
    return sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isfile(os.path.join(d, "normal.npy")))


def load_domain(dirs, k, max_points, label):
    angs, nmags = [], []
    for d in dirs:
        coord = np.load(os.path.join(d, "coord.npy")).astype(np.float64)
        normal = np.load(os.path.join(d, "normal.npy")).astype(np.float64)
        nmags.append(np.linalg.norm(normal, axis=1))
        a = local_angle_deg(coord, normal, k=k, max_points=max_points)
        if a.size:
            angs.append(a)
        print(f"    [{label}] {os.path.basename(d):28s} N={coord.shape[0]:>8,d}")
    angs = np.concatenate(angs) if angs else np.empty(0, np.float32)
    nmags = np.concatenate(nmags) if nmags else np.empty(0, np.float32)
    return angs, nmags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic-root", default="data/scannet-v1.1.1-2of3/train")
    ap.add_argument("--vggt-root", default="data/scannet-v1.1.1-2of3-vggt/test")
    ap.add_argument("--n-synthetic", type=int, default=8,
                    help="number of synthetic scenes to sample (0 = all)")
    ap.add_argument("--k", type=int, default=16, help="k-NN for consistency metric")
    ap.add_argument("--max-points", type=int, default=60000,
                    help="max query points per scene (caps runtime)")
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
                    help="NormalJitter sigmas to sweep for calibration")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    syn_all = scene_dirs(args.synthetic_root)
    vggt = scene_dirs(args.vggt_root)
    rng = np.random.default_rng(args.seed)
    if args.n_synthetic and len(syn_all) > args.n_synthetic:
        syn = [syn_all[i] for i in sorted(rng.choice(len(syn_all), args.n_synthetic, replace=False))]
    else:
        syn = syn_all
    print(f"[scan] synthetic: {len(syn)}/{len(syn_all)} scenes from {args.synthetic_root}")
    print(f"[scan] vggt:      {len(vggt)} scenes from {args.vggt_root}\n")
    if not syn or not vggt:
        raise SystemExit("missing scenes in one domain; check roots / run vggt_to_scene first")

    print("[load] synthetic")
    syn_ang, syn_nmag = load_domain(syn, args.k, args.max_points, "syn")
    print("[load] vggt")
    vggt_ang, vggt_nmag = load_domain(vggt, args.k, args.max_points, "vggt")

    print("\n=== |normal| sanity (should be ~1.0) ===")
    print(f"  synthetic |n| mean={syn_nmag.mean():.4f}  vggt |n| mean={vggt_nmag.mean():.4f}")

    print("\n=== local angular consistency (lower = cleaner) ===")
    syn_med = describe("synthetic", syn_ang)
    vggt_med = describe("vggt", vggt_ang)

    print("\n=== calibration sweep: NormalJitter(sigma) on synthetic ===")
    print(f"  target = vggt median {vggt_med:.2f} deg")
    best = None
    for sigma in args.sigmas:
        jit_ang = []
        for d in syn:
            coord = np.load(os.path.join(d, "coord.npy")).astype(np.float64)
            normal = np.load(os.path.join(d, "normal.npy")).astype(np.float64)
            nj = jitter_normals(normal, sigma, rng)
            a = local_angle_deg(coord, nj, k=args.k, max_points=args.max_points)
            if a.size:
                jit_ang.append(a)
        jit_ang = np.concatenate(jit_ang)
        med = describe(f"syn + jitter(sigma={sigma:.2f})", jit_ang)
        if best is None or abs(med - vggt_med) < abs(best[1] - vggt_med):
            best = (sigma, med)
    print(f"\n[calibration] closest sigma to VGGT median: sigma={best[0]:.2f} "
          f"(synthetic median {best[1]:.2f} vs vggt {vggt_med:.2f} deg)")
    print("  -> use this as NormalJitter.sigma; widen the range above/below it for "
          "domain randomization, and consider corrupt_ratio if VGGT p99 >> synthetic p99.")


if __name__ == "__main__":
    main()
