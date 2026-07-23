"""Convert VisionSCO SSL instance-label exports to LitePT scene dirs.

Input: <ssl_root>/<scene>.npz produced by VisionSCO's save_instance_labels.py with
    points   (M, 3) float32  world-space metres
    colors   (M, 3) uint8    RGB
    view_ids (M,)   int32    source camera view per point
    seg_ids  (M,)   int32    >=0 matched 3D instance; -2 object point left
                             ungrouped by the matcher; -1 background/board

Output: one compressed <out_root>/{train,val}/<scene>.npz per scene holding
coord/color/normal/segment/instance arrays (read by RealSSLDataset), with the
real-scene partial-label encoding used by the -query-realft finetune config:

    seg_id >= 0  -> segment = OBJECT_CLASS (6), instance = dense id 0..K-1
    seg_id == -2 -> segment = OBJECT_CLASS (6), instance = -1  (instance-ignored)
    seg_id == -1 -> segment = UNKNOWN_BG (7),   instance = -1  (not-object loss only)

Sentinel 7 (== num_classes) means "class unknown but definitely NOT object"; the
MQ-v1m1 not-object loss consumes it and remaps it to ignore (-1) before CE/Lovasz.

Normals: PCA k-NN normals via the same compute_normals used by the v1_0_1
preprocessing (reused by file path, as tools/vggt_to_scene.py does). Orientation
needs a camera origin per point; if --camera-root is given, per-scene
<camera-root>/<scene>/camera.json supplies true camera centers (matched to view_ids
by camera order); otherwise a virtual top-down camera above the scene centroid is
used for every point (fine for checkout scenes seen from above).

Usage:
  python tools/convert_ssl_scenes.py \
      --ssl-root ~/workspace/jhp/ssl_labels_out \
      --out-root data/real-ssl-v1 \
      [--camera-root /path/to/ssl/dataset] [--val-stores 001,004] \
      [--min-instance-points 100] [--limit N] [--workers 16]

Split: scene names are {type}_{store_id}_{machine_id}_{num}; scenes whose store_id is
in --val-stores go to val, everything else to train. Names that don't match the
4-field format are skipped with a warning.
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

OBJECT_CLASS = 6  # index of "object" in the 7-class v1.1.1 label set
UNKNOWN_BG = 7    # sentinel: known not-object, class otherwise unknown


def _load_compute_normals():
    path = os.path.join(
        _REPO_ROOT, "datasets", "preprocessing", "v1_0_1", "preprocess_v1_0_1.py"
    )
    spec = importlib.util.spec_from_file_location("preprocess_v1_0_1", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_normals


def _camera_origins(camera_json):
    """camera.json -> (V, 3) world-space camera centers, in file order.

    Entries hold world-to-camera R (3x3) and t (3,): center c = -R^T @ t.
    Many 80k scenes lack true R/t and instead carry backfilled pseudo_R/pseudo_t
    (borrowed from the same machine) or weak_pseudo_R/weak_pseudo_t (different
    machine) — see tools/fill_pseudo_extrinsics.py. Fall back in that order;
    return None if any camera has no extrinsics at all (caller then uses the
    virtual top-down camera for the whole scene).
    """
    with open(camera_json) as f:
        cams = json.load(f)["cameras"]
    origins = []
    for cam in cams:
        for rk, tk in (("R", "t"), ("pseudo_R", "pseudo_t"),
                       ("weak_pseudo_R", "weak_pseudo_t")):
            if rk in cam and tk in cam:
                R = np.asarray(cam[rk], dtype=np.float64).reshape(3, 3)
                t = np.asarray(cam[tk], dtype=np.float64).reshape(3)
                origins.append(-R.T @ t)
                break
        else:
            return None
    return np.asarray(origins, dtype=np.float32)


def convert_scene(
    npz_path, out_dir, compute_normals, camera_root=None, min_instance_points=0
):
    data = np.load(npz_path)
    coord = data["points"].astype(np.float32)
    color = data["colors"].astype(np.uint8)
    seg_ids = data["seg_ids"].astype(np.int32)
    view_ids = data["view_ids"].astype(np.int64)

    segment = np.full(seg_ids.shape, UNKNOWN_BG, dtype=np.int32)
    segment[seg_ids >= 0] = OBJECT_CLASS
    segment[seg_ids == -2] = OBJECT_CLASS

    instance = np.full(seg_ids.shape, -1, dtype=np.int32)
    uids = np.unique(seg_ids[seg_ids >= 0])
    next_id = 0
    for uid in uids:
        m = seg_ids == uid
        if min_instance_points and int(m.sum()) < min_instance_points:
            # too small to trust: keep object semantics, drop the instance id
            continue
        instance[m] = next_id
        next_id += 1

    # normal orientation: true camera origins when available, else one virtual
    # top-down camera above the scene centroid
    scene = os.path.basename(npz_path)[: -len(".npz")]
    cam_json = (
        os.path.join(camera_root, scene, "camera.json") if camera_root else None
    )
    origins = _camera_origins(cam_json) if cam_json and os.path.isfile(cam_json) else None
    if origins is not None:
        cam_idx = np.clip(view_ids, 0, len(origins) - 1)
    else:
        center = coord.mean(axis=0)
        virtual = center + np.array([0.0, 0.0, 1.0], dtype=np.float32)
        virtual[2] = coord[:, 2].max() + 0.5
        origins = virtual[None, :]
        cam_idx = np.zeros(coord.shape[0], dtype=np.int64)
    normal = compute_normals(coord, origins, cam_idx)

    # One compressed npz per scene (NOT per-asset .npy dirs): the uncompressed
    # layout costs ~120 GB for 80k scenes and filled the disk; compressed is ~2.5x
    # smaller. RealSSLDataset.get_data reads this format directly.
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    tmp = out_dir + ".npz.tmp"
    with open(tmp, "wb") as f:
        # compact dtypes (loader casts back up): normal f16 (~0.001 resolution on a
        # unit vector), segment i1 (values 6/7), instance i2 (< 100 ids/scene).
        # coord stays f32 — f16 would cost ~1 mm at metre scale vs the 2 mm grid.
        np.savez_compressed(
            f,
            coord=coord,
            color=color,
            normal=normal.astype(np.float16),
            segment=segment.astype(np.int8),
            instance=instance.astype(np.int16),
        )
    os.replace(tmp, out_dir + ".npz")
    return dict(points=coord.shape[0], instances=next_id)


def _write_run_config(args) -> None:
    """Append this launch's provenance to <out-root>/run_config.jsonl:
    source dataset, converting script (+ git commit), and all CLI args.
    Also copies the source's own run_config.jsonl forward if present."""
    import datetime
    import json
    import shutil
    import socket
    import subprocess

    def _git(*cmd):
        try:
            return subprocess.run(
                ["git", *cmd], cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            return ""

    os.makedirs(args.out_root, exist_ok=True)
    record = {
        "launched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "script": os.path.abspath(__file__),
        "source_dataset": os.path.abspath(args.ssl_root),
        "args": vars(args),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        },
    }
    with open(os.path.join(args.out_root, "run_config.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    # chain of custody: carry the source dataset's provenance forward
    src_rc = os.path.join(args.ssl_root, "run_config.jsonl")
    if os.path.isfile(src_rc):
        shutil.copy(src_rc, os.path.join(args.out_root, "source_run_config.jsonl"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ssl-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--camera-root", default=None,
                    help="SSL dataset root with per-scene camera.json (optional)")
    ap.add_argument("--val-stores", default="001,004",
                    help="csv of store ids held out as the val split")
    ap.add_argument("--min-instance-points", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    _write_run_config(args)

    npzs = sorted(
        f for f in os.listdir(args.ssl_root) if f.endswith(".npz")
    )
    if args.limit:
        npzs = npzs[: args.limit]

    val_stores = {s.strip() for s in args.val_stores.split(",") if s.strip()}
    jobs = []
    n_skipped_name = 0
    for fname in npzs:
        scene = fname[: -len(".npz")]
        # store-based split: scene name is {type}_{store_id}_{machine_id}_{num}
        fields = scene.split("_")
        if len(fields) != 4 or not fields[1].isdigit():
            print(f"[SKIP] unparseable scene name: {scene}")
            n_skipped_name += 1
            continue
        split = "val" if fields[1] in val_stores else "train"
        out_dir = os.path.join(args.out_root, split, scene)
        if os.path.isfile(out_dir + ".npz"):
            continue  # resumable: skip already-converted scenes
        jobs.append((os.path.join(args.ssl_root, fname), out_dir, split))

    counts = {"train": 0, "val": 0}
    if args.workers > 1:
        import multiprocessing as mp

        with mp.Pool(args.workers, initializer=_worker_init) as pool:
            results = pool.imap_unordered(
                _worker_run,
                [
                    (npz, out, split, args.camera_root, args.min_instance_points)
                    for npz, out, split in jobs
                ],
                chunksize=8,
            )
            for i, (split, scene, err) in enumerate(results):
                if err:
                    print(f"[FAIL] {scene}: {err}")
                    continue
                counts[split] += 1
                if (i + 1) % 500 == 0 or i == len(jobs) - 1:
                    print(f"[{i + 1}/{len(jobs)}] {scene} -> {split}")
    else:
        compute_normals = _load_compute_normals()
        for i, (npz_path, out_dir, split) in enumerate(jobs):
            scene = os.path.basename(out_dir)
            try:
                stats = convert_scene(
                    npz_path,
                    out_dir,
                    compute_normals,
                    camera_root=args.camera_root,
                    min_instance_points=args.min_instance_points,
                )
            except Exception as e:  # noqa: BLE001 — keep the batch going
                print(f"[FAIL] {scene}: {e}")
                continue
            counts[split] += 1
            if (i + 1) % 100 == 0 or i == len(jobs) - 1:
                print(f"[{i + 1}/{len(jobs)}] {scene} -> {split} "
                      f"({stats['points']} pts, {stats['instances']} instances)")
    n_done_before = len(npzs) - len(jobs) - n_skipped_name
    print(f"done: {counts['train']} train / {counts['val']} val scenes "
          f"(skipped {n_done_before} already converted, "
          f"{n_skipped_name} unparseable names) -> {args.out_root}")


_WORKER_COMPUTE_NORMALS = None


def _worker_init():
    global _WORKER_COMPUTE_NORMALS
    _WORKER_COMPUTE_NORMALS = _load_compute_normals()


def _worker_run(job):
    npz_path, out_dir, split, camera_root, min_instance_points = job
    scene = os.path.basename(out_dir)
    try:
        convert_scene(
            npz_path,
            out_dir,
            _WORKER_COMPUTE_NORMALS,
            camera_root=camera_root,
            min_instance_points=min_instance_points,
        )
        return split, scene, None
    except Exception as e:  # noqa: BLE001
        return split, scene, str(e)


if __name__ == "__main__":
    main()
