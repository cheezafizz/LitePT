#!/usr/bin/env python
"""Forecast time-to-convergence from a partial training log (probe-run extrapolation).

Parses the periodic `Val result: mAP/AP50/AP25 a/b/c.` lines that InsSegEvaluator
writes every `eval_step_interval` optimizer steps, fits a saturating curve to a
robust upper envelope of the AP50 series (the raw series has crash-dips, e.g. the
embed run swings 0.66 -> 0.05 -> 0.69 between adjacent evals), and reports the
predicted plateau AP50, the step/epoch where the fit reaches within `--eps` of the
plateau, and the corresponding wall-clock.

Probe protocol:
  1. Launch training normally (the forecast only reads the log).
  2. After ~3 epochs:  python tools/estimate_convergence.py <exp>/train.log \
         --sec-per-step 6.4 --sec-per-eval 256
  3. If predicted convergence exceeds your budget, kill the run and change levers
     (context_grid_factor=3, more GPUs) instead of finding out days later.

Backtest mode: `--first-k-epochs N` truncates the series before fitting, so a
completed log with known ground truth (e.g. the embed run: plateau AP50 ~0.82,
best reached ~epoch 21/200) measures forecast accuracy vs probe length.

Reference timings (RTX 5090, measured 2026-07): query@K=50 6.4 s/step + 256 s/eval;
embed 0.56 s/step + ~32 s/eval.
"""

import argparse
import math
import re
import sys

import numpy as np

MIN_EVALS = 10

VAL_RE = re.compile(
    r"Val result: mAP/AP50/AP25 ([0-9.]+)/([0-9.]+)/([0-9.]+)\."
)
TRAIN_RE = re.compile(r"Train: \[(\d+)/(\d+)\]\[(\d+)/(\d+)\]")
EVAL_INT_RE = re.compile(r"eval_step_interval\s*=\s*(\d+)")


def parse_log(path):
    """Return (steps, ap50, steps_per_epoch, eval_interval, n_epochs_seen)."""
    ap50 = []
    steps_per_epoch = None
    eval_interval = None
    last_epoch = 1
    with open(path, errors="replace") as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                steps_per_epoch = int(m.group(4))
                last_epoch = int(m.group(1))
                continue
            m = VAL_RE.search(line)
            if m:
                ap50.append(float(m.group(2)))
                continue
            if eval_interval is None:
                m = EVAL_INT_RE.search(line)
                if m:
                    eval_interval = int(m.group(1))
    if eval_interval is None:
        eval_interval = 1000
    if steps_per_epoch is None:
        sys.exit("error: no 'Train: [e/E][i/I]' lines found -- is this a train.log?")
    steps = np.arange(1, len(ap50) + 1, dtype=float) * eval_interval
    return steps, np.asarray(ap50), steps_per_epoch, eval_interval, last_epoch


def upper_envelope(steps, ap50, window=5, q=0.9):
    """Windowed upper-quantile envelope -- robust to eval crash-dips."""
    env = np.empty_like(ap50)
    half = window // 2
    for i in range(len(ap50)):
        lo, hi = max(0, i - half), min(len(ap50), i + half + 1)
        env[i] = np.quantile(ap50[lo:hi], q)
    return steps, env


def fit_exponential(steps, env):
    """AP50(s) = A - B*exp(-s/tau); returns (params, rmse) or None."""
    from scipy.optimize import curve_fit

    def f(s, A, B, tau):
        return A - B * np.exp(-s / tau)

    smax = steps[-1]
    try:
        p, _ = curve_fit(
            f, steps, env,
            p0=[min(env[-1] + 0.1, 1.0), max(env[-1] - env[0], 0.1), smax / 3],
            bounds=([0, 0, 1.0], [1.0, 1.0, smax * 50]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None
    rmse = float(np.sqrt(np.mean((f(steps, *p) - env) ** 2)))
    return ("exponential", f, p, rmse)


def fit_powerlaw(steps, env):
    """AP50(s) = A - B*s^(-c); returns (params, rmse) or None."""
    from scipy.optimize import curve_fit

    def f(s, A, B, c):
        return A - B * np.power(s, -c)

    try:
        p, _ = curve_fit(
            f, steps, env,
            p0=[min(env[-1] + 0.1, 1.0), max(env[-1] - env[0], 0.1) * steps[0] ** 0.5, 0.5],
            bounds=([0, 1e-6, 0.01], [1.0, 1e6, 5.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None
    rmse = float(np.sqrt(np.mean((f(steps, *p) - env) ** 2)))
    return ("powerlaw", f, p, rmse)


def convergence_step(f, p, eps):
    """Smallest step where the fit reaches (1-eps)*A. None if it never does."""
    A = p[0]
    target = (1.0 - eps) * A
    lo, hi = 1.0, 1e9
    if f(hi, *p) < target:
        return None
    for _ in range(200):
        mid = math.sqrt(lo * hi)  # geometric bisection: s* spans orders of magnitude
        if f(mid, *p) >= target:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.001:
            break
    return hi


def bootstrap_ci(steps, env, fitter, eps, n_boot, rng):
    """Residual-resampling bootstrap -> 80% CI on (plateau A, convergence step)."""
    base = fitter(steps, env)
    if base is None:
        return None
    _, f, p, _ = base
    resid = env - f(steps, *p)
    A_samples, s_samples = [], []
    for _ in range(n_boot):
        fake = f(steps, *p) + rng.choice(resid, size=len(resid), replace=True)
        r = fitter(steps, np.clip(fake, 0.0, 1.0))
        if r is None:
            continue
        _, rf, rp, _ = r
        s_star = convergence_step(rf, rp, eps)
        if s_star is None:
            continue
        A_samples.append(rp[0])
        s_samples.append(s_star)
    if len(s_samples) < max(10, n_boot // 10):
        return None
    return (
        np.percentile(A_samples, [10, 90]),
        np.percentile(s_samples, [10, 90]),
    )


def fmt_wallclock(seconds):
    d = seconds / 86400.0
    return f"{seconds / 3600.0:.1f} h ({d:.1f} d)" if d >= 0.5 else f"{seconds / 3600.0:.1f} h"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="path to train.log")
    ap.add_argument("--sec-per-step", type=float, default=6.4,
                    help="measured steady-state seconds per optimizer step (default: query@K=50)")
    ap.add_argument("--sec-per-eval", type=float, default=256.0,
                    help="measured seconds per periodic eval (default: query 500-scene subset)")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="converged = fit within eps fraction of predicted plateau")
    ap.add_argument("--first-k-epochs", type=int, default=None,
                    help="fit using only evals from the first K epochs (backtest / probe mode)")
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap refits for the 80%% CI (0 = off)")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--quantile", type=float, default=0.9)
    args = ap.parse_args()

    steps, ap50, spe, eval_int, epochs_seen = parse_log(args.log)
    print(f"parsed {len(ap50)} evals (every {eval_int} steps, {spe} steps/epoch, "
          f"log covers ~{epochs_seen} epochs)")

    if args.first_k_epochs is not None:
        keep = steps <= args.first_k_epochs * spe
        steps, ap50 = steps[keep], ap50[keep]
        print(f"truncated to first {args.first_k_epochs} epochs -> {len(ap50)} evals")

    if len(ap50) < MIN_EVALS:
        sys.exit(f"error: only {len(ap50)} evals -- need >= {MIN_EVALS} for a meaningful fit. "
                 f"Let the probe run longer (~{MIN_EVALS * eval_int} steps minimum).")

    xs, env = upper_envelope(steps, ap50, window=args.window, q=args.quantile)

    fits = [r for r in (fit_exponential(xs, env), fit_powerlaw(xs, env)) if r is not None]
    if not fits:
        sys.exit("error: neither curve form converged on this data")

    rng = np.random.default_rng(0)
    for name, f, p, rmse in sorted(fits, key=lambda r: r[3]):
        A = p[0]
        s_star = convergence_step(f, p, args.eps)
        print(f"\n[{name}] rmse={rmse:.4f}  predicted plateau AP50 = {A:.3f}")
        if s_star is None:
            print("  fit never reaches (1-eps)*plateau within 1e9 steps -- treat as non-converging fit")
            continue
        epoch_star = s_star / spe
        wall = s_star * args.sec_per_step + (s_star / eval_int) * args.sec_per_eval
        print(f"  converged (within {args.eps:.0%} of plateau) at step ~{s_star:,.0f} "
              f"= epoch ~{epoch_star:.1f}")
        print(f"  wall-clock @ {args.sec_per_step}s/step + {args.sec_per_eval}s/eval: {fmt_wallclock(wall)}")
        if args.bootstrap:
            fitter = fit_exponential if name == "exponential" else fit_powerlaw
            ci = bootstrap_ci(xs, env, fitter, args.eps, args.bootstrap, rng)
            if ci is None:
                print("  bootstrap: too few successful refits for a CI")
            else:
                (a_lo, a_hi), (s_lo, s_hi) = ci
                w_lo = s_lo * args.sec_per_step + (s_lo / eval_int) * args.sec_per_eval
                w_hi = s_hi * args.sec_per_step + (s_hi / eval_int) * args.sec_per_eval
                print(f"  80% CI: plateau [{a_lo:.3f}, {a_hi:.3f}]  "
                      f"epoch [{s_lo / spe:.1f}, {s_hi / spe:.1f}]  "
                      f"wall [{fmt_wallclock(w_lo)}, {fmt_wallclock(w_hi)}]")


if __name__ == "__main__":
    main()
