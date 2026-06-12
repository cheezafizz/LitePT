"""Storage-driven uniform subsample of v1.1.1 raw scenes.

Keeps 2 of every 3 scenes (drops rank%3==2 by ascending scene id within each parent
dir), records the kept/deleted id lists and stride into the recovery manifest, and
(with --execute) rm -rf's the dropped scene dirs. Run AFTER the recovery manifest and
the rerun_failed_cases deletion.

Dry-run by default: prints counts and projected disk usage; pass --execute to delete.
"""
import argparse
import json
import os
import shutil
import sys

ROOT = "/home/fai/workspace/jhp/dataset/v1.1.1"
PARENT_DIRS = ["70", "72", "73", "75", "76", "77", "78"]
MANIFESTS = [
    "/home/fai/workspace/jhp/dataset/v1.1.1_recovery_manifest.json",
    "/home/fai/workspace/jhp/LitePT/tools/v1.1.1_recovery_manifest.json",
]
# Keep 2 of every 3 scenes -> drop ranks where rank % 3 == 2.
DROP_MOD, DROP_REM, STRIDE_DESC = 3, 2, "keep-2-of-3 (drop ascending-rank %3==2)"


def current_scene_ids(parent):
    p = os.path.join(ROOT, parent)
    return sorted(
        int(n) for n in os.listdir(p)
        if n.isdigit() and os.path.isdir(os.path.join(p, n))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="Actually delete dropped scene dirs (default: dry-run).")
    args = ap.parse_args()

    manifest = json.load(open(MANIFESTS[0]))
    manifest["kept_stride_k"] = STRIDE_DESC
    manifest["kept"] = {}
    manifest["deleted"] = {}

    total_keep = total_drop = 0
    drop_paths = []
    for d in PARENT_DIRS:
        ids = current_scene_ids(d)  # ascending
        kept = [sid for r, sid in enumerate(ids) if r % DROP_MOD != DROP_REM]
        dropped = [sid for r, sid in enumerate(ids) if r % DROP_MOD == DROP_REM]
        manifest["kept"][d] = {"count": len(kept), "scene_ids": kept}
        manifest["deleted"][d] = {"count": len(dropped), "scene_ids": dropped}
        total_keep += len(kept)
        total_drop += len(dropped)
        drop_paths.extend(os.path.join(ROOT, d, str(s)) for s in dropped)
        print(f"  {d}: present={len(ids)} keep={len(kept)} drop={len(dropped)}")

    print(f"TOTAL keep={total_keep} drop={total_drop} "
          f"(keep_frac={total_keep/(total_keep+total_drop):.4f})")

    if not args.execute:
        print("\nDRY RUN — no files deleted. Re-run with --execute to delete.")
        return 0

    # Persist manifest BEFORE deleting so the kept/deleted record is durable.
    for p in MANIFESTS:
        with open(p, "w") as f:
            json.dump(manifest, f)
    print(f"\nManifest updated (kept/deleted/stride) at {len(MANIFESTS)} paths.")

    deleted = 0
    for path in drop_paths:
        # safety: must live directly under ROOT/<parent>/<digit>
        parent = os.path.basename(os.path.dirname(path))
        leaf = os.path.basename(path)
        if (os.path.dirname(os.path.dirname(path)) == ROOT
                and parent in PARENT_DIRS and leaf.isdigit()
                and os.path.isdir(path)):
            shutil.rmtree(path)
            deleted += 1
            if deleted % 2000 == 0:
                print(f"  deleted {deleted}/{len(drop_paths)} ...")
        else:
            print(f"  SKIP (unsafe/missing): {path}")
    print(f"Deleted {deleted} scene dirs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
