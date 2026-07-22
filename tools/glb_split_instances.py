"""
Re-emit each merged predicted-instance GLB from tools/infer_insseg.py as a GLB whose
instances are SEPARATE named nodes, so a glTF viewer can toggle each instance on/off.

The flat <scene>_pred.glb is one point cloud where every point carries the color of
the winner-take-all instance it belongs to (gray = unassigned). We recover the
instances by grouping points on their exact uint8 RGB, then write one named
trimesh.PointCloud node per instance inside a trimesh.Scene. Node labels (class+score)
are recovered best-effort from viz/vggt_insseg/run.log by replaying the deterministic
instance_palette() used at colorize time (color -> proposal id). No GPU, no re-inference.

  /home/fai/miniconda3/envs/litept/bin/python tools/glb_split_instances.py
  ... --glob 'viz/vggt_insseg/normaug/*_pred.glb' --log viz/vggt_insseg/run.log
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import trimesh

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from engines.hooks.insseg_viz import instance_palette, GRAY  # noqa: E402


def parse_log(log_path):
    """Map each '<...>_pred.glb' output path -> {pid: (class, score)} from run.log."""
    out = {}
    if not log_path or not os.path.isfile(log_path):
        return out
    cur = {}
    prop_re = re.compile(r"#(\d+)\s+class=(\S+)\s+score=([\d.]+)")
    viz_re = re.compile(r"\[viz\] wrote (\S+_pred\.glb)")
    with open(log_path) as f:
        for line in f:
            m = prop_re.search(line)
            if m:
                cur[int(m.group(1))] = (m.group(2), float(m.group(3)))
                continue
            v = viz_re.search(line)
            if v:
                out[os.path.normpath(v.group(1))] = cur
                cur = {}
    return out


def load_points(path):
    g = trimesh.load(path, process=False)
    pc = g if isinstance(g, trimesh.PointCloud) else list(g.geometry.values())[0]
    return np.asarray(pc.vertices, dtype=np.float32), np.asarray(pc.colors, dtype=np.uint8)


def split_one(path, labels_by_path, suffix):
    verts, colors = load_points(path)
    rgb = colors[:, :3]
    gray = np.all(rgb == GRAY, axis=1)
    inst_rgb = rgb[~gray]
    uniq = np.unique(inst_rgb, axis=0) if inst_rgb.size else np.zeros((0, 3), np.uint8)

    # color -> proposal id via the deterministic palette of size P (#proposals in log)
    labels = labels_by_path.get(os.path.normpath(path), {})
    color2pid = {}
    if labels:
        P = max(labels) + 1
        pal = instance_palette(P)
        color2pid = {tuple(int(x) for x in pal[pid]): pid for pid in range(P)}

    scene = trimesh.Scene()
    for k, c in enumerate(uniq):
        sel = np.all(rgb == c, axis=1)
        pid = color2pid.get(tuple(int(x) for x in c))
        if pid is not None and pid in labels:
            cls, score = labels[pid]
            name = f"i{pid:02d}_{cls}_{score:.2f}"
        else:
            name = f"instance_{k:02d}"
        scene.add_geometry(
            trimesh.PointCloud(vertices=verts[sel], colors=colors[sel]),
            node_name=name, geom_name=name,
        )
    if gray.any():
        scene.add_geometry(
            trimesh.PointCloud(vertices=verts[gray], colors=colors[gray]),
            node_name="unassigned", geom_name="unassigned",
        )

    out = path[: -len(".glb")] + suffix + ".glb"
    scene.export(out)
    return out, int(uniq.shape[0]), int(gray.sum())


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="viz/vggt_insseg/*/*_pred.glb")
    ap.add_argument("--log", default="viz/vggt_insseg/run.log")
    ap.add_argument("--suffix", default="_instances")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(args.glob) if not p.endswith(args.suffix + ".glb"))
    labels_by_path = parse_log(args.log)
    print(f"[split] {len(paths)} GLBs | labels parsed for {len(labels_by_path)} scenes")
    n_ok = 0
    for p in paths:
        try:
            out, n_inst, n_un = split_one(p, labels_by_path, args.suffix)
            print(f"[ok]   {out}  ({n_inst} instance nodes + {n_un:,} unassigned pts)")
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {p}: {e}")
    print(f"[done] {n_ok}/{len(paths)} split")


if __name__ == "__main__":
    main()
