#!/usr/bin/env python3
"""Fill missing camera extrinsics in dataset_ssl_1_80k/ with *pseudo* extrinsics.

Many camera.json ship intrinsics (K/distCoef) but no extrinsics (R/t). Because the
camera rig geometry is fixed per machine, we borrow R/t from another scene with the
same machine_id and store it under NEW keys (the real R/t are never touched):

  * strong tier -- donor has the SAME machine_id  -> keys ``pseudo_R`` / ``pseudo_t``
  * weak tier   -- no same-machine donor exists    -> borrow from a DIFFERENT machine,
                    keys ``weak_pseudo_R`` / ``weak_pseudo_t``

Directory naming is ``{type}_{store_id}_{machine_id}_{num}`` (e.g. ``1_006_023_000000``).

Default is a dry-run; pass ``--apply`` to actually modify files. See the sibling plan
``check-if-camera-json-in-cheeky-metcalfe.md`` for the analysis that motivates this.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

DEFAULT_ROOT = "/home/fai/workspace/jhp/dataset/SSL_dataset/dataset_ssl_1_80k"

# The 7 views the strict loader (load_fn.load_cameras_from_json) requires, and the two
# eyes each rig exposes. All scenes carry exactly these 14 camera names.
LEFT_VIEWS = ["TB", "TC", "TF", "TR", "TL", "RW", "LW"]
ALL_CAMS = [f"{v}_{eye}" for v in LEFT_VIEWS for eye in ("left", "right")]
REQUIRED_LEFT = [f"{v}_left" for v in LEFT_VIEWS]

# Populated per-worker by _init_worker; maps donor_scene_id -> {cam_name: {"R":.., "t":..}}.
_TEMPLATES = {}


def parse_scene(scene):
    """(type, store, machine) from ``{type}_{store}_{machine}_{num}``; None if malformed."""
    p = scene.split("_")
    if len(p) < 4:
        return None
    return p[0], p[1], p[2]


def _has_rt(cam):
    return isinstance(cam, dict) and "R" in cam and "t" in cam


def scan_scene(scene):
    """Index one scene. Returns (scene, machine, type, eligible_donor, needs_fill)."""
    parts = parse_scene(scene)
    if parts is None:
        return None
    typ, _store, machine = parts
    try:
        with open(os.path.join(SCAN_ROOT, scene, "camera.json")) as f:
            look = {c.get("name"): c for c in json.load(f)["cameras"] if isinstance(c, dict)}
    except Exception:
        return (scene, machine, typ, False, False)
    eligible = all(_has_rt(look.get(c, {})) for c in ALL_CAMS)      # donor: all 14 have R/t
    needs_fill = any(not _has_rt(look.get(c, {})) for c in REQUIRED_LEFT)  # missing on left-7
    return (scene, machine, typ, eligible, needs_fill)


def load_template(scene):
    """Build {cam_name: {"R": <raw>, "t": <raw>}} from a donor scene's camera.json."""
    with open(os.path.join(SCAN_ROOT, scene, "camera.json")) as f:
        cams = json.load(f)["cameras"]
    return {c["name"]: {"R": c["R"], "t": c["t"]} for c in cams
            if isinstance(c, dict) and "name" in c and "R" in c and "t" in c}


def _init_worker(root, templates):
    global SCAN_ROOT, _TEMPLATES
    SCAN_ROOT = root
    _TEMPLATES = templates


def fill_scene(job):
    """Add (weak_)pseudo_R/_t to every camera lacking real R/t. Returns a status dict."""
    scene, machine, tier, donor_scene, apply, force = job
    key_r = "pseudo_R" if tier == "pseudo" else "weak_pseudo_R"
    key_t = "pseudo_t" if tier == "pseudo" else "weak_pseudo_t"
    src_key = "pseudo_extrinsic_source" if tier == "pseudo" else "weak_pseudo_extrinsic_source"
    template = _TEMPLATES[donor_scene]
    path = os.path.join(SCAN_ROOT, scene, "camera.json")

    try:
        with open(path) as f:
            cfg = json.load(f)
    except Exception as e:
        return {"scene": scene, "machine": machine, "tier": tier,
                "donor": donor_scene, "status": "error:%s" % str(e)[:60]}

    changed = 0
    for cam in cfg.get("cameras", []):
        if not isinstance(cam, dict):
            continue
        if _has_rt(cam):
            continue  # already has real extrinsics -- never overwrite
        if key_r in cam and key_t in cam and not force:
            continue  # already pseudo-filled (idempotent)
        donor_cam = template.get(cam.get("name"))
        if donor_cam is None:
            continue
        cam[key_r] = donor_cam["R"]
        cam[key_t] = donor_cam["t"]
        changed += 1

    if changed == 0:
        return {"scene": scene, "machine": machine, "tier": tier,
                "donor": donor_scene, "status": "skip(nochange)"}

    cfg[src_key] = donor_scene
    if apply:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, path)
    return {"scene": scene, "machine": machine, "tier": tier,
            "donor": donor_scene, "status": ("filled" if apply else "would-fill"),
            "cams": changed}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--apply", action="store_true",
                    help="actually write files (default: dry-run, no writes)")
    ap.add_argument("--force", action="store_true",
                    help="re-fill even if pseudo keys already present")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--weak-fallback-machine", default=None,
                    help="machine_id to source weak (cross-machine) donors from "
                         "(default: the type-1 machine with the most donors)")
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "pseudo_extrinsics_manifest.csv"))
    args = ap.parse_args()

    global SCAN_ROOT
    SCAN_ROOT = args.root
    scenes = sorted(s for s in os.listdir(args.root)
                    if os.path.isdir(os.path.join(args.root, s)))
    print("scanning %d scene dirs ..." % len(scenes))

    # ---- Index pass -------------------------------------------------------------
    donors_by_machine = defaultdict(list)   # machine -> [eligible donor scene ids]
    type1_donor_count = defaultdict(int)    # machine -> #donors that are type-1
    to_fill = []                            # (scene, machine) needing extrinsics
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker, initargs=(args.root, {})) as ex:
        for r in ex.map(scan_scene, scenes, chunksize=200):
            if r is None:
                continue
            scene, machine, typ, eligible, needs_fill = r
            if eligible:
                donors_by_machine[machine].append(scene)
                if typ == "1":
                    type1_donor_count[machine] += 1
            if needs_fill:
                to_fill.append((scene, machine))

    # Deterministic strong donor per machine = smallest eligible scene id.
    strong_donor = {m: min(v) for m, v in donors_by_machine.items()}

    # Weak fallback donor: representative type-1 rig (most donors), overridable.
    if args.weak_fallback_machine:
        weak_machine = args.weak_fallback_machine
        if weak_machine not in strong_donor:
            ap.error("--weak-fallback-machine %s has no donor scene" % weak_machine)
    elif type1_donor_count:
        weak_machine = max(sorted(type1_donor_count), key=lambda m: type1_donor_count[m])
    else:
        weak_machine = None
    weak_donor = strong_donor.get(weak_machine) if weak_machine else None

    # ---- Assign each to-fill scene a tier + donor -------------------------------
    jobs = []
    n_strong = n_weak = n_nodonor = 0
    for scene, machine in to_fill:
        if machine in strong_donor:
            jobs.append((scene, machine, "pseudo", strong_donor[machine]))
            n_strong += 1
        elif weak_donor is not None:
            jobs.append((scene, machine, "weak_pseudo", weak_donor))
            n_weak += 1
        else:
            n_nodonor += 1

    print("\n=== PLAN ===")
    print("  scenes needing extrinsics : %d" % len(to_fill))
    print("  strong (same machine)     : %d  keys pseudo_R/pseudo_t" % n_strong)
    print("  weak   (machine %-4s)      : %d  keys weak_pseudo_R/weak_pseudo_t"
          % (str(weak_machine), n_weak))
    print("  unfillable (no donor)     : %d" % n_nodonor)
    print("  mode                      : %s" % ("APPLY (writing files)" if args.apply
                                                else "DRY-RUN (no writes)"))

    # ---- Fill pass --------------------------------------------------------------
    needed_donors = set(strong_donor[m] for _, m in to_fill if m in strong_donor)
    if weak_donor is not None and n_weak:
        needed_donors.add(weak_donor)
    templates = {d: load_template(d) for d in needed_donors}

    work = [(s, m, tier, donor, args.apply, args.force) for (s, m, tier, donor) in jobs]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker, initargs=(args.root, templates)) as ex:
        for res in ex.map(fill_scene, work, chunksize=200):
            results.append(res)

    status_counts = defaultdict(int)
    for r in results:
        status_counts[r["status"].split(":")[0]] += 1
    print("\n=== RESULT ===")
    for k in sorted(status_counts):
        print("  %-16s %d" % (k, status_counts[k]))

    # ---- Manifest ---------------------------------------------------------------
    with open(args.manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "machine_id", "tier", "donor_scene_id", "status"])
        for r in results:
            w.writerow([r["scene"], r["machine"], r["tier"], r["donor"], r["status"]])
    print("\nmanifest -> %s (%d rows)" % (args.manifest, len(results)))
    if not args.apply:
        print("\nDRY-RUN only. Re-run with --apply to write the %d files." % (n_strong + n_weak))


if __name__ == "__main__":
    main()
