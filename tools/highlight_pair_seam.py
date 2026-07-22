"""
Isolate the cannot-link SEAM between a SPECIFIC pair of predicted instances, from a
tools/infer_insseg.py `--save-npz --highlight-split-cause` result. Pure npz->npz
(numpy + scipy; no torch / repo build / GPU), so it is instant and re-runnable.

The global split-cause view (Split-cause mode in tools/viser_insseg_viewer.py) paints
EVERY cannot-link seam red. This tool restricts that to the boundary between two chosen
final instances A and B: the points of A within `--radius` of B (and vice versa), kept
only where they are already global cannot-link seam points (split_cause). The result is
written as a "focused" npz that the EXISTING viewer Split-cause mode renders unchanged --
A and B vivid, their shared seam red, everything else dimmed/off.

  /home/fai/miniconda3/envs/litept/bin/python tools/highlight_pair_seam.py \
      --npz viz/splitcause/welstory_001361/pred.npz --pair 16 18
  # -> viz/splitcause/seam-16x18/welstory_001361_pred.npz
  # then (re)start tools/viser_insseg_viewer.py --results-dir viz/splitcause and pick
  # the model "Split-cause: seam-16x18".

Instance ids are the `inst_id` baked into the source npz (the #NNN proposal order printed
by tools/infer_insseg.py); they are frozen by that run, so no re-inference is needed.
"""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True,
                    help="source <scene>_pred.npz from infer_insseg.py "
                         "(--save-npz --highlight-split-cause)")
    ap.add_argument("--pair", required=True, nargs=2, type=int, action="append",
                    metavar=("A", "B"),
                    help="two instance ids whose interface to isolate; repeatable "
                         "(e.g. --pair 16 18 --pair 3 7)")
    ap.add_argument("--radius", type=float, default=0.005,
                    help="contact radius in meters (default 0.005 = "
                         "mask_split_radius*voxel_size = 2.5*0.002)")
    ap.add_argument("--out-root", default="viz/splitcause",
                    help="results root; each pair -> <out-root>/seam-AxB/<scene>_pred.npz")
    ap.add_argument("--scene-name", default=None,
                    help="output scene basename (default: <src>_pred.npz stem, else npz 'scene')")
    return ap.parse_args()


def pair_seam(coord, inst_id, split_cause, A, B, r):
    """Boolean (N,) mask of the cannot-link seam between instances A and B.

    contact band = points of A within r of B + points of B within r of A; the seam is
    that band intersected with the global cannot-link split_cause. Returns
    (seam_mask, n_a, n_b, n_band, used_fallback)."""
    a_sel = np.nonzero(inst_id == A)[0]
    b_sel = np.nonzero(inst_id == B)[0]
    band = np.zeros(coord.shape[0], dtype=bool)
    if a_sel.size and b_sel.size:
        da, _ = cKDTree(coord[b_sel]).query(coord[a_sel], k=1)
        db, _ = cKDTree(coord[a_sel]).query(coord[b_sel], k=1)
        band[a_sel[da <= r]] = True
        band[b_sel[db <= r]] = True
    seam = band & split_cause
    used_fallback = False
    if not seam.any() and band.any():
        # A and B touch but their interface is not a 2D-mask cannot-link seam (split was
        # geometric/embedding); fall back to the raw contact band so something is shown.
        seam = band
        used_fallback = True
    return seam, a_sel.size, b_sel.size, int(band.sum()), used_fallback


def main():
    args = parse_args()
    z = np.load(args.npz, allow_pickle=True)
    if "split_cause" not in z.files:
        raise SystemExit(
            f"{args.npz} has no `split_cause` field; produce it with "
            f"infer_insseg.py --highlight-split-cause --save-npz")
    coord = z["coord"]
    inst_id = z["inst_id"].astype(np.int64)
    split_cause = z["split_cause"].astype(bool)
    # carry every source field forward; only split_cause / split_fragment are overridden
    base = {k: z[k] for k in z.files}

    # name the output scene so it lands under the SAME viewer Scene entry as the source.
    # Prefer the npz's own `scene` field (e.g. "welstory_001361"); the filename stem is
    # unreliable (the source may be plain "pred.npz").
    src_stem = os.path.basename(args.npz)
    src_stem = (src_stem[:-len("_pred.npz")] if src_stem.endswith("_pred.npz")
                else os.path.splitext(src_stem)[0])
    npz_scene = str(z["scene"].item()) if "scene" in z.files else ""
    scene_out = args.scene_name or npz_scene or src_stem

    for A, B in args.pair:
        seam, n_a, n_b, n_band, fb = pair_seam(coord, inst_id, split_cause, A, B, args.radius)
        frag = (inst_id == A) | (inst_id == B)
        tag = f"seam-{A}x{B}"
        out_dir = os.path.join(args.out_root, tag)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{scene_out}_pred.npz")
        out = dict(base)
        out["split_cause"] = seam
        out["split_fragment"] = frag
        out["model"] = tag
        np.savez_compressed(out_path, **out)
        note = "  [FALLBACK: contact band, NOT cannot-link]" if fb else ""
        print(f"[pair {A}x{B}] #{A}={n_a:,} pts  #{B}={n_b:,} pts  "
              f"contact band={n_band:,}  seam(red)={int(seam.sum()):,}{note}")
        print(f"           -> {out_path}")


if __name__ == "__main__":
    main()
