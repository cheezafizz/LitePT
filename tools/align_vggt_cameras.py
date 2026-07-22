"""
Align VGGT camera parameters for the val_sample dataset to the real calibrated rig.

For every VGGT scene under <val-output-root>/<cat>/<id>/data.npz this does two things:

  Task 2 (intrinsics) -- "follow the preprocess that shaped the point cloud":
    The calibrated camera.json describes 640x480 images, but the point cloud lives on
    the 392x518 grid VGGT produced. We push each view's calibrated K (and distortion)
    through the EXACT recon-time preprocess
        undistort (getOptimalNewCameraMatrix, alpha=0) -> rotate(0/180) -> center_pp(pad) -> resize(crop)
    by reusing `preprocess_view` from VisionSCO's load_fn.py (single source of truth,
    same call pcdRecon.py uses). The resulting 3x3 K matches the point-cloud image shape.
    The K transform is image-content-independent, so we feed a dummy 640x480 frame --
    no image files are read (robust to naming, and faster).

  Task 1 (extrinsics) -- "PnP to minimize the reprojection error of the point cloud":
    Each VGGT pixel (u,v) maps to a 3D point world_points[v,u], giving exact 2D-3D
    correspondences. We run cv2.solvePnP (seeded from the stored extrinsic, LM, which
    minimizes total reprojection error) with the aligned K from Task 2 and no distortion
    (the 392x518 grid is already undistorted). The stored VGGT extrinsics are only
    ~2-6 px consistent with the real camera model; PnP drives this to ~0.7-1.7 px.

Outputs are non-destructive:
  * data.npz gains keys `extrinsic_pnp` (S,3,4 float32, world->cam, VGGT world frame),
    `reproj_error_px` (S,), `reproj_error_px_stored` (S,), `intrinsic_aligned` (S,3,3).
    All original keys are preserved (load-all -> add -> atomic re-save).
  * <dfine-root>/<cat>/<id>/camera_aligned.json gets per-view K (392x518), the rotated
    calibrated extrinsic (reference), the PnP extrinsic, and the reprojection error.

Convention: extrinsics are world->cam 3x4 (OpenCV), matching the stored `extrinsic`.
`extrinsic_pnp` is in the VGGT world frame (object points are VGGT world_points), so it
directly replaces the stored extrinsic when projecting the point cloud.

Run with an env that has cv2 + numpy + torch/torchvision/PIL (preprocess_view needs them),
e.g. /home/fai/miniconda3/envs/da3/bin/python. The litept env lacks cv2.

Example:
  /home/fai/miniconda3/envs/da3/bin/python tools/align_vggt_cameras.py --dry-run
  /home/fai/miniconda3/envs/da3/bin/python tools/align_vggt_cameras.py
  /home/fai/miniconda3/envs/da3/bin/python tools/align_vggt_cameras.py --scene toyosu/000294
"""
import argparse
import glob
import importlib.util
import json
import os
import zlib

import cv2
import numpy as np

VAL_OUTPUT_ROOT = "/home/fai/workspace/jhp/dataset/val_sample_output"
DFINE_ROOT = "/home/fai/workspace/jhp/D-FINE-seg/val_sample"
LOAD_FN_PATH = "/home/fai/workspace/jhp/VisionSCO/models/vggt/vggt/utils/load_fn.py"

# Recon-time preprocess config, pinned from VisionSCO/modules/pcdRecon.py:716.
TARGET_SIZE = (392, 518)   # (H, W)
ORIGINAL_SIZE = (640, 480)  # (W, H) -- all rig cameras are vga 640x480
RESIZE_MODE = "crop"
CENTER_PP = True


def load_load_fn():
    """Import preprocess_view / DEFAULT_ROTATION_MAP by path."""
    spec = importlib.util.spec_from_file_location("vggt_load_fn", LOAD_FN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cameras(cj_path, order):
    """Tolerant per-view loader for camera.json (matches load_fn semantics where present).

    Returns lists aligned to `order` of: K (3x3), dist_coeffs (8,) or None, extrinsic (3x4
    world->cam, t in METERS) or None. Unlike load_fn.load_cameras_from_json, R/t are optional
    -- several val_sample scenes ship K/distCoef only (no extrinsics). Neither task needs the
    calibrated R/t (Task 2 only transforms K; Task 1 PnP seeds from the stored VGGT extrinsic),
    so a missing R/t yields extrinsic=None and is otherwise handled gracefully.
    """
    cfg = json.load(open(cj_path))
    lookup = {c["name"]: c for c in cfg["cameras"]}
    Ks, dists, exts = [], [], []
    for v in order:
        cam = lookup[f"{v}_left"]
        Ks.append(np.array(cam["K"], dtype=np.float64).reshape(3, 3))
        dists.append(np.array(cam["distCoef"], dtype=np.float64).flatten() if "distCoef" in cam else None)
        if "R" in cam and "t" in cam:
            R = np.array(cam["R"], dtype=np.float64).reshape(3, 3)
            t = np.array(cam["t"], dtype=np.float64).flatten()[:3] / 1000.0  # mm -> m
            exts.append(np.hstack([R, t.reshape(3, 1)]))
        else:
            exts.append(None)
    return Ks, dists, exts


def atomic_save_npz(path, arrays):
    """Crash-safe in-place overwrite of an .npz (compressed, matching the source)."""
    tmp = path + ".tmp.npz"  # ends in .npz so np.savez writes exactly this path
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def reproj_errors(R, t, P, u, v, K):
    """Per-point reprojection error (px) of 3D points P onto (u,v) under [R|t], K (pinhole)."""
    Pc = (R @ P.T).T + t
    z = Pc[:, 2]
    ru = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    rv = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    return np.sqrt((ru - u) ** 2 + (rv - v) ** 2)


def solve_view(P, u, v, K, R0, t0, ransac_px, subsample, rng):
    """Return (R, t, median_err_px, method) minimizing reprojection error.

    Primary: SOLVEPNP_ITERATIVE (LM) seeded from the stored extrinsic.
    Fallback: solvePnPRansac (EPNP, no guess) + LM refine on inliers, if the seeded
    solve fails / is worse than `ransac_px` / is worse than the stored pose.
    Always keeps the better of {stored, solved}.
    """
    N = len(P)
    e_stored = float(np.median(reproj_errors(R0, t0, P, u, v, K)))
    if N < 6:
        return R0, t0, e_stored, "stored(too_few_pts)"

    img_pts = np.stack([u, v], 1)
    idx = rng.choice(N, size=min(subsample, N), replace=False)
    Ps = np.ascontiguousarray(P[idx]).reshape(-1, 1, 3)
    ips = np.ascontiguousarray(img_pts[idx]).reshape(-1, 1, 2)

    rvec0, _ = cv2.Rodrigues(R0)
    tvec0 = t0.reshape(3, 1).copy()
    best_R, best_t, best_e, method = R0, t0, e_stored, "stored(kept)"

    ok, rvec, tvec = cv2.solvePnP(
        Ps, ips, K, None, rvec0.copy(), tvec0.copy(),
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok:
        R, _ = cv2.Rodrigues(rvec)
        e = float(np.median(reproj_errors(R, tvec.flatten(), P, u, v, K)))
        if e < best_e:
            best_R, best_t, best_e, method = R, tvec.flatten(), e, "iterative"

    if (not ok) or best_e > ransac_px:
        try:
            ok2, rvec2, tvec2, inl = cv2.solvePnPRansac(
                Ps, ips, K, None, iterationsCount=200,
                reprojectionError=float(ransac_px), flags=cv2.SOLVEPNP_EPNP,
            )
            if ok2 and inl is not None and len(inl) >= 6:
                inl = inl.flatten()
                rvec2, tvec2 = cv2.solvePnPRefineLM(Ps[inl], ips[inl], K, None, rvec2, tvec2)
                R2, _ = cv2.Rodrigues(rvec2)
                e2 = float(np.median(reproj_errors(R2, tvec2.flatten(), P, u, v, K)))
                if e2 < best_e:
                    best_R, best_t, best_e, method = R2, tvec2.flatten(), e2, "ransac"
        except cv2.error:
            pass

    return best_R, best_t, best_e, method


def process_scene(npz_path, val_output_root, dfine_root, lf, ransac_px, subsample, seed, dry_run):
    rel = os.path.relpath(os.path.dirname(npz_path), val_output_root)
    cj = os.path.join(dfine_root, rel, "camera.json")
    if not os.path.isfile(cj):
        print(f"  WARN {rel}: no camera.json at {cj} -> skip")
        return None

    # Per-scene deterministic RNG: the PnP subsample (and thus the result) is reproducible
    # for a given scene regardless of how many other scenes run in the same invocation.
    rng = np.random.RandomState((seed + zlib.crc32(rel.encode())) & 0xFFFFFFFF)

    d = np.load(npz_path, allow_pickle=True)
    order = [str(x) for x in d["input_order_list"]]
    world = d["world_points"].astype(np.float64)      # (S,H,W,3)
    ext_stored = d["extrinsic"].astype(np.float64)    # (S,3,4)
    mask = d["filtered_valid_mask"]                   # (S,H,W) bool
    S, H, W, _ = world.shape

    Ks, dists, exts = load_cameras(cj, order)
    us, vs = np.meshgrid(np.arange(W), np.arange(H))  # (H,W) each; u=col, v=row
    dummy = np.zeros((ORIGINAL_SIZE[1], ORIGINAL_SIZE[0], 3), np.uint8)
    target_H, target_W = TARGET_SIZE
    identity_ext = np.hstack([np.eye(3), np.zeros((3, 1))])

    K_list, ext_calib_rot, ext_pnp = [], [], []
    err_pnp, err_stored, methods = [], [], []

    for s, view in enumerate(order):
        K = Ks[s]
        dist = dists[s]
        extr = exts[s]
        has_d = dist is not None and np.any(np.abs(dist) > 1e-9)

        # Task 2: aligned intrinsic. The K transform is independent of the extrinsic, so when
        # camera.json omits R/t we pass identity and ignore the rotated-extrinsic output.
        r = lf.preprocess_view(
            img_np=dummy.copy(), K=K.copy(),
            extrinsic=(extr if extr is not None else identity_ext).copy(), cam_view=view,
            dist_coeffs=dist if has_d else None, original_size=ORIGINAL_SIZE,
            do_undistort=has_d, center_pp=CENTER_PP, target_size=TARGET_SIZE,
            resize_mode=RESIZE_MODE, rotation_map=lf.DEFAULT_ROTATION_MAP,
        )
        Kp = np.asarray(r["K"], dtype=np.float64)
        K_list.append(Kp)
        ext_calib_rot.append(np.asarray(r["extrinsic"], dtype=np.float64) if extr is not None else None)

        # Task 1: PnP extrinsic minimizing reprojection error of the point cloud.
        m = mask[s]
        P = world[s][m]
        u = us[m].astype(np.float64)
        v = vs[m].astype(np.float64)
        R0 = ext_stored[s][:, :3]
        t0 = ext_stored[s][:, 3]
        R, t, e, method = solve_view(P, u, v, Kp, R0, t0, ransac_px, subsample, rng)
        e0 = float(np.median(reproj_errors(R0, t0, P, u, v, Kp))) if len(P) else float("nan")

        ext_pnp.append(np.hstack([R, t.reshape(3, 1)]))
        err_pnp.append(e)
        err_stored.append(e0)
        methods.append(method)
        print(f"  {rel:32} {view:3} stored={e0:6.2f} -> pnp={e:6.2f} px  [{method}]")

    ext_pnp = np.stack(ext_pnp).astype(np.float32)
    K_arr = np.stack(K_list).astype(np.float64)
    err_pnp = np.asarray(err_pnp, dtype=np.float32)
    err_stored = np.asarray(err_stored, dtype=np.float32)

    if not dry_run:
        # 1) augment data.npz (preserve all original keys)
        arrays = {k: d[k] for k in d.files}
        arrays["extrinsic_pnp"] = ext_pnp
        arrays["intrinsic_aligned"] = K_arr.astype(np.float32)
        arrays["reproj_error_px"] = err_pnp
        arrays["reproj_error_px_stored"] = err_stored
        atomic_save_npz(npz_path, arrays)

        # 2) write camera_aligned.json sidecar
        sidecar = {
            "grid_size": [target_W, target_H],  # [W, H] = [518, 392]
            "source_camera_json": cj,
            "convention": "extrinsic world->cam (OpenCV) 3x4; extrinsic_pnp in VGGT world frame",
            "views": [
                {
                    "name": view,
                    "rotation_deg": int(lf.DEFAULT_ROTATION_MAP[view]),
                    "K": K_list[s].tolist(),
                    "extrinsic_calib_rot": (
                        ext_calib_rot[s].tolist() if ext_calib_rot[s] is not None else None),
                    "extrinsic_pnp": ext_pnp[s].astype(np.float64).tolist(),
                    "reproj_error_px": float(err_pnp[s]),
                    "reproj_error_px_stored": float(err_stored[s]),
                    "pnp_method": methods[s],
                }
                for s, view in enumerate(order)
            ],
        }
        atomic_write_json(os.path.join(dfine_root, rel, "camera_aligned.json"), sidecar)

    return {"rel": rel, "err_pnp": err_pnp, "err_stored": err_stored, "methods": methods}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-output-root", default=VAL_OUTPUT_ROOT)
    ap.add_argument("--dfine-root", default=DFINE_ROOT)
    ap.add_argument("--scene", default=None,
                    help="optional '<cat>/<id>' filter (substring match); default: all scenes")
    ap.add_argument("--ransac-fallback-px", type=float, default=3.0,
                    help="median px above which the seeded PnP triggers a RANSAC fallback")
    ap.add_argument("--subsample", type=int, default=40000,
                    help="max correspondences used in the PnP solve (error reported on all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="compute + print, write nothing")
    args = ap.parse_args()

    lf = load_load_fn()

    npzs = sorted(glob.glob(os.path.join(args.val_output_root, "*", "*", "data.npz")))
    if args.scene:
        npzs = [p for p in npzs if args.scene in os.path.relpath(os.path.dirname(p), args.val_output_root)]
    print(f"{'DRY-RUN: ' if args.dry_run else ''}aligning {len(npzs)} scene(s) "
          f"(ransac_fallback={args.ransac_fallback_px}px, subsample={args.subsample})\n")

    results = []
    for npz_path in npzs:
        res = process_scene(npz_path, args.val_output_root, args.dfine_root, lf,
                            args.ransac_fallback_px, args.subsample, args.seed, args.dry_run)
        if res:
            results.append(res)

    # Summary
    if results:
        all_pnp = np.concatenate([r["err_pnp"] for r in results])
        all_stored = np.concatenate([r["err_stored"] for r in results])
        worst = max(results, key=lambda r: np.nanmax(r["err_pnp"]))
        wi = int(np.nanargmax(worst["err_pnp"]))
        print(f"\nDone. {len(results)} scenes, {len(all_pnp)} views.")
        print(f"  median reproj  stored={np.nanmedian(all_stored):.2f}px  pnp={np.nanmedian(all_pnp):.2f}px")
        print(f"  max view reproj  stored={np.nanmax(all_stored):.2f}px  pnp={np.nanmax(all_pnp):.2f}px")
        print(f"  worst pnp view: {worst['rel']} [{worst['methods'][wi]}] = {worst['err_pnp'][wi]:.2f}px")
        n_regress = int(np.sum(all_pnp > all_stored + 1e-6))
        print(f"  views where pnp worse than stored: {n_regress} (should be 0)")


if __name__ == "__main__":
    main()
