"""Verify cross-view instance merging against the recorded object count.

For every raw sequence under <dataset_root>/<scene>/<seq>/ this reruns the real
preprocessing merge (`merge_anns_via_voxel_overlap`) and counts how many merged
instances belong to *real objects*, then checks the invariant:

    n_real_object_instances <= scene_metadata.json["num_objects"]

Why only "real objects": the preprocessor lumps real assets *and* out-of-bounds
junk ("oob") into semantic class 6, so the on-disk instance.npy cannot tell them
apart. We therefore recompute from the raw COCO annotations, where the source
`category_id` separates them (per <scene>/mapping_info.json):

    floor = -1 ; machine=1 board=2 tray=3 paper=4 table=5 oob=6 ; real asset >= 7

A merged instance is a real object iff its source annotation's category_id >= 7.
Same-asset copies share one category_id, so we count merged *instance ids*
(union-find canonical ids), not categories.

Verdicts per sequence:
  PASS    n_real <= num_objects
  EXCEED  n_real >  num_objects   (merge under-merged, or dataset over-counted)
  UNDER   n_real <  num_objects   (report-only: over-merge / occlusion / crop)

Exits non-zero if any EXCEED is found, so this doubles as a CI-style assertion.
Complement: tools/diagnose_merge_collisions.py catches the opposite failure
(over-merge: ann_ids that co-occur in one camera view fused together).

Usage:
  /home/fai/miniconda3/envs/litept/bin/python tools/verify_merge_counts.py \
    --dataset-root /home/fai/workspace/jhp/dataset/v1.1.1 \
    --voxel-size 0.002 --num-workers $(nproc) \
    --out-csv logs/verify_merge_counts.csv
"""

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Import the existing diagnostic as a module: it loads the preprocess module
# first and defers h5py/PIL imports. That import order matters -- importing
# h5py/PIL before scipy segfaults in this env. Reuse its build_cloud + pp.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DMC_PATH = os.path.join(_HERE, "diagnose_merge_collisions.py")
_spec = importlib.util.spec_from_file_location("diagnose_merge_collisions", _DMC_PATH)
dmc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dmc)
pp = dmc.pp  # the loaded preprocess_v1_0_1 module

# Source-category classification (see module docstring / mapping_info.json).
OOB_CATEGORY_ID = 6
REAL_OBJECT_MIN_CATEGORY_ID = 7

CSV_FIELDS = ["scene", "seq", "num_objects", "n_real", "n_oob", "n_fixture", "verdict"]


def verify_sequence(task):
    """Worker: rerun the merge for one sequence and classify merged instances.

    Returns a dict with CSV_FIELDS plus an internal "_status" describing failures
    (worker never raises, so the pool never crashes -- mirrors process_sequence).
    """
    seq_dir, scene, seq, voxel_size, crop_half = task
    base = {"scene": scene, "seq": seq, "num_objects": "",
            "n_real": "", "n_oob": "", "n_fixture": ""}
    try:
        with open(os.path.join(seq_dir, "scene_metadata.json")) as f:
            num_objects = int(json.load(f)["num_objects"])
        with open(os.path.join(seq_dir, "coco_annotations.json")) as f:
            coco = json.load(f)
        catid_by_ann = {int(a["id"]): int(a["category_id"]) for a in coco["annotations"]}

        cloud = dmc.build_cloud(seq_dir)
        if cloud is None:
            return {**base, "verdict": "NOVIEWS", "_status": "no valid views"}
        coord, cat_raw, ann_raw, cam_idx = cloud

        # Crop to the working-volume AABB, then merge -- same order as
        # process_sequence (crop first so far-field points don't affect merging).
        in_box = np.all((coord >= -crop_half) & (coord <= crop_half), axis=1)
        if not in_box.any():
            return {**base, "num_objects": num_objects,
                    "verdict": "NOPOINTS", "_status": "no points inside working volume"}
        coord = coord[in_box]
        cat_raw = cat_raw[in_box]
        ann_raw = ann_raw[in_box]
        cam_idx = cam_idx[in_box]

        canon = pp.merge_anns_via_voxel_overlap(coord, cat_raw, ann_raw, cam_idx, voxel_size)

        n_real = n_oob = n_fixture = 0
        for cid in np.unique(canon[canon >= 0]):
            catid = catid_by_ann.get(int(cid), -999)
            if catid >= REAL_OBJECT_MIN_CATEGORY_ID:
                n_real += 1
            elif catid == OOB_CATEGORY_ID:
                n_oob += 1
            else:
                n_fixture += 1

        if n_real > num_objects:
            verdict = "EXCEED"
        elif n_real < num_objects:
            verdict = "UNDER"
        else:
            verdict = "PASS"
        return {"scene": scene, "seq": seq, "num_objects": num_objects,
                "n_real": n_real, "n_oob": n_oob, "n_fixture": n_fixture,
                "verdict": verdict, "_status": "ok"}
    except Exception as e:  # noqa: BLE001 -- never crash the pool
        return {**base, "verdict": "FAILED", "_status": f"{type(e).__name__}: {e}"}


def load_done(out_csv):
    """Return set of already-processed (scene, seq) from an existing CSV."""
    done = set()
    if not os.path.isfile(out_csv):
        return done
    with open(out_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row["scene"]), int(row["seq"])))
            except (KeyError, ValueError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default="/home/fai/workspace/jhp/dataset/v1.1.1")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Scene IDs to verify (e.g. 70 71). Default: all.")
    ap.add_argument("--voxel-size", type=float, default=0.002,
                    help="Merge voxel size (default 0.002, matches the 2mm 2of3 build).")
    ap.add_argument("--crop-half", type=float, default=pp.WORKING_VOLUME_HALF,
                    help="Half-side of the symmetric working-volume cube (metres).")
    ap.add_argument("--num-workers", type=int, default=mp.cpu_count())
    ap.add_argument("--out-csv", default="logs/verify_merge_counts.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N (post-filter) sequences.")
    args = ap.parse_args()

    print(f"Enumerating sequences under {args.dataset_root} ...")
    sequences = pp.enumerate_sequences(args.dataset_root)  # (seq_dir, scene, seq)
    print(f"  found {len(sequences)} sequences")

    if args.scenes:
        wanted = {int(s) for s in args.scenes}
        sequences = [s for s in sequences if s[1] in wanted]
        print(f"  --scenes filter: {len(sequences)} sequences")

    out_csv = args.out_csv
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    done = load_done(out_csv)
    if done:
        before = len(sequences)
        sequences = [s for s in sequences if (s[1], s[2]) not in done]
        print(f"  resume: {len(done)} already in {out_csv}; "
              f"{len(sequences)} of {before} remaining")

    if args.limit > 0:
        sequences = sequences[: args.limit]
        print(f"  --limit set: processing only {len(sequences)} sequences")

    tasks = [(sd, scn, sq, args.voxel_size, args.crop_half) for sd, scn, sq in sequences]
    if not tasks:
        print("Nothing to do.")
    else:
        print(f"Verifying {len(tasks)} sequences with {args.num_workers} workers, "
              f"voxel={args.voxel_size} m, crop=+/-{args.crop_half} m ...")

    write_header = not os.path.isfile(out_csv) or os.path.getsize(out_csv) == 0
    counts = {"PASS": 0, "EXCEED": 0, "UNDER": 0,
              "NOVIEWS": 0, "NOPOINTS": 0, "FAILED": 0}
    exceed_rows = []

    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for i, r in enumerate(pool.map(verify_sequence, tasks), 1):
                writer.writerow({k: r[k] for k in CSV_FIELDS})
                counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
                if r["verdict"] == "EXCEED":
                    exceed_rows.append(r)
                if r["verdict"] in ("EXCEED", "FAILED", "NOVIEWS", "NOPOINTS"):
                    print(f"  [{i}/{len(tasks)}] scene{r['scene']:02d}/{r['seq']}: "
                          f"{r['verdict']} ({r['_status']}; "
                          f"n_real={r['n_real']} num_objects={r['num_objects']})")
                if i % 200 == 0:
                    f.flush()
                    print(f"  ... {i}/{len(tasks)} done "
                          f"(PASS={counts['PASS']} EXCEED={counts['EXCEED']} "
                          f"UNDER={counts['UNDER']})")

    # ---- aggregate summary (over THIS run; CSV holds the cumulative record) ----
    total = sum(counts.values())
    checked = counts["PASS"] + counts["EXCEED"] + counts["UNDER"]
    print("\n==== verify_merge_counts summary (this run) ====")
    print(f"  sequences processed : {total}")
    print(f"  checked (had points): {checked}")
    for k in ("PASS", "EXCEED", "UNDER", "NOVIEWS", "NOPOINTS", "FAILED"):
        print(f"    {k:8s}: {counts[k]}")
    if checked:
        print(f"  EXCEED rate (of checked): {counts['EXCEED'] / checked:.4%}")
    if exceed_rows:
        exceed_rows.sort(key=lambda r: r["n_real"] - r["num_objects"], reverse=True)
        print("\n  worst EXCEED offenders (n_real - num_objects):")
        for r in exceed_rows[:20]:
            print(f"    scene{r['scene']:02d}/{r['seq']}: "
                  f"n_real={r['n_real']} > num_objects={r['num_objects']} "
                  f"(+{r['n_real'] - r['num_objects']}; oob={r['n_oob']})")
    print(f"\n  full per-sequence results: {out_csv}")

    raise SystemExit(1 if counts["EXCEED"] else 0)


if __name__ == "__main__":
    main()
