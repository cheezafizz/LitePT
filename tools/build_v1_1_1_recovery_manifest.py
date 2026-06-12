"""Inventory v1.1.1 scene IDs into a recovery manifest (run BEFORE any deletion).

Records, per parent dir and per rerun_failed_cases subdir, the full sorted list of
numeric scene IDs and their count, plus the SFTP source, so the full dataset can be
re-fetched later from just these recorded numbers.
"""
import json
import os
import sys

ROOT = "/home/fai/workspace/jhp/dataset/v1.1.1"
PARENT_DIRS = ["70", "72", "73", "75", "76", "77", "78"]
SFTP_SOURCE = (
    "junhyeok.park@192.168.0.200:"
    "/datasets/fainders/vco_synthetic_3d_datasets/v1.1.1/"
)
OUT_PATHS = [
    "/home/fai/workspace/jhp/dataset/v1.1.1_recovery_manifest.json",
    "/home/fai/workspace/jhp/LitePT/tools/v1.1.1_recovery_manifest.json",
]


def list_scene_ids(path):
    if not os.path.isdir(path):
        return None
    ids = sorted(
        int(n) for n in os.listdir(path)
        if n.isdigit() and os.path.isdir(os.path.join(path, n))
    )
    return ids


def main():
    manifest = {
        "dataset": "v1.1.1",
        "root": ROOT,
        "sftp_source": SFTP_SOURCE,
        "note": (
            "Full inventory captured before storage-driven deletion. To recover any "
            "deleted scene, SFTP-get the listed scene id under the corresponding "
            "parent dir. parent_dirs = consumed by preprocessor; rerun_failed_cases "
            "was deleted (never read by the pipeline)."
        ),
        "parent_dirs": {},
        "rerun_failed_cases": {},
        "kept_stride_k": None,
        "kept": {},
        "deleted": {},
    }

    total_parent = 0
    for d in PARENT_DIRS:
        ids = list_scene_ids(os.path.join(ROOT, d))
        manifest["parent_dirs"][d] = {"count": len(ids), "scene_ids": ids}
        total_parent += len(ids)

    total_rerun = 0
    rerun_root = os.path.join(ROOT, "rerun_failed_cases")
    if os.path.isdir(rerun_root):
        for sub in sorted(os.listdir(rerun_root)):
            subp = os.path.join(rerun_root, sub)
            if sub.isdigit() and os.path.isdir(subp):
                ids = list_scene_ids(subp)
                manifest["rerun_failed_cases"][sub] = {
                    "count": len(ids), "scene_ids": ids
                }
                total_rerun += len(ids)

    manifest["totals"] = {
        "parent_dirs_scenes": total_parent,
        "rerun_failed_cases_scenes": total_rerun,
    }

    for p in OUT_PATHS:
        with open(p, "w") as f:
            json.dump(manifest, f)

    print(f"parent_dirs total scenes: {total_parent}")
    for d in PARENT_DIRS:
        print(f"  {d}: {manifest['parent_dirs'][d]['count']}")
    print(f"rerun_failed_cases total scenes: {total_rerun}")
    for k, v in manifest["rerun_failed_cases"].items():
        print(f"  {k}: {v['count']}")
    print("manifest written to:")
    for p in OUT_PATHS:
        print(f"  {p} ({os.path.getsize(p)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
