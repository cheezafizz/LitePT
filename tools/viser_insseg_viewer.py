"""
Localhost viser viewer for the GT-free InsSeg results written by tools/infer_insseg.py
(`--save-npz`). Reads ONLY the structured .npz files (numpy + viser; no repo imports), so
it runs in the lightweight viewer venv created by tools/setup_viser_venv.sh.

Each .npz (viz/vggt_insseg/<model>/<scene>_pred.npz) holds:
  coord (N,3) f32 | rgb (N,3) u8 | inst_id (N,) i32 (-1 = unassigned)
  inst_class (P,) i32 | inst_score (P,) f32 | inst_color (P,3) u8
  class_names (str[]) | scene (str) | model (str)

GUI: scene + model dropdowns (model includes an "RGB (input)" entry), point-size and
score-threshold sliders, Show all / Hide all buttons, an "unassigned" toggle, and ONE
checkbox per predicted instance (labelled "#id class score") that toggles just that
instance on/off. Switching scene or model rebuilds the clouds and the checkbox list.

  .venv-viser/bin/python tools/viser_insseg_viewer.py \
      --results-dir viz/vggt_insseg --port 8080
  # then open http://localhost:8080
"""
import argparse
import glob
import os
import time

import numpy as np
import viser

RGB_MODE = "RGB (input)"
SEM_MODE = "Semantic (labels)"
VIEW_MODE = "View (camera)"
# one pseudo-model per real model: "Split-cause: <model>" shows that model's npz seam
SPLITCAUSE_PREFIX = "Split-cause: "
GRAY = np.array([160, 160, 160], dtype=np.uint8)
# preferred display order; any other model dirs are appended sorted
_MODEL_ORDER = ["normaug", "rigidaug", "softaug"]

# tab10-style categorical palette for coloring points by their source camera view
_VIEW_PALETTE = np.array([
    [ 31, 119, 180], [255, 127,  14], [ 44, 160,  44], [214,  39,  40],
    [148, 103, 189], [140,  86,  75], [227, 119, 194], [127, 127, 127],
    [188, 189,  34], [ 23, 190, 207], [174, 199, 232], [255, 187, 120],
], dtype=np.uint8)


def view_color(i):
    """Solid (3,) uint8 color for camera-view index i (cycles if > palette len)."""
    return _VIEW_PALETTE[int(i) % len(_VIEW_PALETTE)]


# preferred config-dropdown order for the --insseg-root layout (cluster_labeled_scenes.py);
# the model-inference (`-modelsem`) arm sits next to its labeled-data sibling for easy A/B.
_CONFIG_ORDER = [
    "geometric",
    "embed", "embed-modelsem",
    "embed-maskclust", "embed-maskclust-allviews",
    "embed-maskclust-strong", "embed-maskclust-strong-modelsem",
    "embed-maskclust-filter", "embed-maskclust-strong-filter",
]


def discover(results_dir):
    """-> (models[list[str]], scenes[sorted list], index[(model,scene)->npz path])."""
    index, models, scenes = {}, set(), set()
    for path in glob.glob(os.path.join(results_dir, "*", "*_pred.npz")):
        model = os.path.basename(os.path.dirname(path))
        scene = os.path.basename(path)[: -len("_pred.npz")]
        index[(model, scene)] = path
        models.add(model)
        scenes.add(scene)
    ordered = [m for m in _MODEL_ORDER if m in models]
    ordered += sorted(m for m in models if m not in _MODEL_ORDER)
    return ordered, sorted(scenes), index


def discover_insseg(root, subdirs="insseg"):
    """Discovery for the tools/cluster_labeled_scenes.py layout
    `<root>/<group>/<seq>/<subdir>/<config>.npz`: the "model" axis is the clustering
    config (filename stem) and the "scene" axis is `<group>/<seq>`. `subdirs` selects
    which per-scene output dir(s) to read (e.g. 'insseg' or 'insseg_conf30').

    With a SINGLE subdir the model axis is the bare config name (unchanged). With
    MULTIPLE subdirs (e.g. a labels arm and a model arm) each config key is tagged
    `"<config>  [<subdir>]"` so both arms coexist in one dropdown and sort adjacent
    per config for A/B. Returns the same (models, scenes, index) triple as
    discover(), so the rest of the viewer is reused unchanged."""
    if isinstance(subdirs, str):
        subdirs = [subdirs]
    tag = len(subdirs) > 1
    index, models, scenes = {}, set(), set()
    for subdir in subdirs:
        for path in glob.glob(os.path.join(root, "*", "*", subdir, "*.npz")):
            config = os.path.splitext(os.path.basename(path))[0]
            key = f"{config}  [{subdir}]" if tag else config
            seq_dir = os.path.dirname(os.path.dirname(path))  # <root>/<group>/<seq>
            scene = os.path.relpath(seq_dir, root)            # "<group>/<seq>"
            index[(key, scene)] = path
            models.add(key)
            scenes.add(scene)
    ordered = [m for m in _CONFIG_ORDER if m in models]
    ordered += sorted(m for m in models if m not in _CONFIG_ORDER)
    return ordered, sorted(scenes), index


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    d = {
        "coord": z["coord"].astype(np.float32),
        "rgb": z["rgb"].astype(np.uint8),
        "inst_id": z["inst_id"].astype(np.int64),
        "inst_class": z["inst_class"].astype(np.int64),
        "inst_score": z["inst_score"].astype(np.float32),
        "inst_color": z["inst_color"].astype(np.uint8),
        "class_names": [str(x) for x in z["class_names"]],
    }
    if "mask_view" in z.files:  # baked-in view index (future runs); else loaded from scene dir
        d["mask_view"] = z["mask_view"].astype(np.int64)
    if "segment" in z.files:  # baked-in per-point GIVEN semantic class (for SEM_MODE)
        d["segment"] = z["segment"].astype(np.int64)
        d["segment_names"] = ([str(x) for x in z["segment_names"]]
                              if "segment_names" in z.files else d["class_names"])
    if "split_cause" in z.files:  # per-point seam (cause-of-split) + fragment masks
        d["split_cause"] = z["split_cause"].astype(bool)
        d["split_fragment"] = (z["split_fragment"].astype(bool)
                               if "split_fragment" in z.files
                               else np.zeros(d["coord"].shape[0], bool))
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="viz/vggt_insseg",
                    help="dir with <model>/<scene>_pred.npz results")
    ap.add_argument("--insseg-root", default=None,
                    help="alternative layout from tools/cluster_labeled_scenes.py: "
                         "<root>/<group>/<seq>/<subdir>/<config>.npz (config=model axis, "
                         "scene=<group>/<seq>). Overrides --results-dir when set.")
    ap.add_argument("--insseg-subdir", nargs="+", default=["insseg"],
                    help="per-scene output subdir(s) read under --insseg-root "
                         "(default 'insseg'; e.g. 'insseg_conf30' for the thresholded run). "
                         "Pass MULTIPLE to overlay them in one Model dropdown, e.g. "
                         "'insseg_svdom_sv insseg_svdom_sv_model' -> each config appears "
                         "once per subdir, tagged '[<subdir>]' and sorted adjacent for A/B.")
    ap.add_argument("--scene-data-dir",
                    default="data/scannet-v1.1.1-2of3-vggt-mask-aligned/test",
                    help="dir with <scene>/mask_view.npy, used by the 'View (camera)' mode "
                         "to color points by source view (degrades gracefully if absent)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point-size", type=float, default=0.004,
                    help="initial point size in meters (2 mm voxel grid -> ~4 mm reads well)")
    args = ap.parse_args()

    if args.insseg_root:
        models, scenes, index = discover_insseg(args.insseg_root, args.insseg_subdir)
        subdir_disp = args.insseg_subdir[0] if len(args.insseg_subdir) == 1 \
            else "{" + ",".join(args.insseg_subdir) + "}"
        src = os.path.join(args.insseg_root, "*", "*", subdir_disp)
        if not index:
            raise SystemExit(f"no '<group>/<seq>/{subdir_disp}/<config>.npz' under "
                             f"{args.insseg_root!r}; run tools/cluster_labeled_scenes.py first")
    else:
        models, scenes, index = discover(args.results_dir)
        src = args.results_dir
        if not index:
            raise SystemExit(f"no '<model>/<scene>_pred.npz' under {args.results_dir!r}; "
                             f"run tools/run_vggt_insseg_viz.sh first")
    model_options = (models + [RGB_MODE, SEM_MODE, VIEW_MODE]
                     + [SPLITCAUSE_PREFIX + m for m in models])
    print(f"[viewer] {len(scenes)} scenes x {len(models)} models "
          f"({', '.join(models)}) from {src}")

    def load_view(scene):
        """Per-point camera-view index for `scene` from <scene-data-dir>/<scene>/mask_view.npy,
        or None if the dir/file is missing (View mode then shows a message)."""
        p = os.path.join(args.scene_data_dir, scene, "mask_view.npy")
        return np.load(p).astype(np.int64) if os.path.isfile(p) else None

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")

    # --- GUI controls --------------------------------------------------------
    gui_scene = server.gui.add_dropdown("Scene", scenes, initial_value=scenes[0])
    gui_model = server.gui.add_dropdown("Model", model_options,
                                        initial_value=model_options[0])
    gui_psize = server.gui.add_slider("Point size (m)", min=0.001, max=0.02,
                                      step=0.001, initial_value=args.point_size)
    gui_thresh = server.gui.add_slider("Score threshold", min=0.0, max=1.0,
                                       step=0.01, initial_value=0.0)
    gui_show = server.gui.add_button("Show all instances")
    gui_hide = server.gui.add_button("Hide all instances")
    gui_info = server.gui.add_markdown("")
    gui_folder = server.gui.add_folder("Instances")

    # mutable view state, kept in a dict so nested callbacks can mutate it
    st = {"clouds": [], "rows": [], "unassigned": None, "folder_handles": []}

    def _clear():
        for h in st["clouds"]:
            h.remove()
        if st["unassigned"] is not None:
            st["unassigned"].remove()
        for h in st["folder_handles"]:
            h.remove()
        st["clouds"], st["rows"], st["unassigned"], st["folder_handles"] = [], [], None, []

    def _add_cloud(name, pts, color_u8):
        # color_u8 is either (3,) for a solid color or (M,3) per-point
        col = np.asarray(color_u8, np.uint8)
        if col.ndim == 1:
            col = np.tile(col, (pts.shape[0], 1))
        return server.scene.add_point_cloud(
            name, points=pts.astype(np.float32), colors=col,
            point_size=gui_psize.value, point_shape="circle")

    def render():
        _clear()
        scene_name, model = gui_scene.value, gui_model.value

        if model == RGB_MODE:
            data = None
            for m in models:  # any model's npz carries the same input rgb/coord
                if (m, scene_name) in index:
                    data = load_npz(index[(m, scene_name)])
                    break
            if data is None:
                gui_info.content = f"_no result for {scene_name}_"
                return
            st["clouds"].append(_add_cloud("/scene/rgb", data["coord"], data["rgb"]))
            gui_info.content = f"**{scene_name}** — RGB input · {data['coord'].shape[0]:,} pts"
            return

        if model == SEM_MODE:
            data = None
            for m in models:  # any config's npz carries the same given per-point labels
                if (m, scene_name) in index:
                    data = load_npz(index[(m, scene_name)])
                    break
            if data is None:
                gui_info.content = f"_no result for {scene_name}_"
                return
            coord = data["coord"]
            seg = data.get("segment")
            if seg is None:
                gui_info.content = (f"_no `segment` in the npz for {scene_name}; re-run "
                                    f"tools/cluster_labeled_scenes.py to bake it_")
                return
            names = data.get("segment_names", data["class_names"])
            ids = np.unique(seg)
            for c in ids:
                pts = coord[seg == c]
                if pts.shape[0] == 0:
                    continue
                cname = names[int(c)] if 0 <= int(c) < len(names) else str(int(c))
                h = _add_cloud(f"/scene/sem/{int(c):02d}", pts, view_color(c))
                with gui_folder:
                    cb = server.gui.add_checkbox(
                        f"{cname} ({pts.shape[0]:,})", initial_value=True)
                cb.on_update(lambda _, hh=h, cc=cb: setattr(hh, "visible", cc.value))
                st["clouds"].append(h)
                st["rows"].append({"handle": h, "cb": cb, "score": 1.0})
                st["folder_handles"].append(cb)
            counts = ", ".join(
                f"{names[int(c)] if 0 <= int(c) < len(names) else int(c)}:"
                f"{int((seg == c).sum()):,}" for c in ids)
            gui_info.content = (f"**{scene_name}** — Semantic (labels) · {len(ids)} "
                                f"classes · {coord.shape[0]:,} pts ({counts})")
            return

        if model == VIEW_MODE:
            data = None
            for m in models:  # any model's npz carries the same input coord (+ maybe mask_view)
                if (m, scene_name) in index:
                    data = load_npz(index[(m, scene_name)])
                    break
            if data is None:
                gui_info.content = f"_no result for {scene_name}_"
                return
            coord = data["coord"]
            view = data.get("mask_view")  # baked into npz?
            if view is None:
                view = load_view(scene_name)  # else from <scene-data-dir>/<scene>/mask_view.npy
            if view is None:
                gui_info.content = (f"_no view data for {scene_name}: expected "
                                    f"`{args.scene_data_dir}/{scene_name}/mask_view.npy`_")
                return
            if view.shape[0] != coord.shape[0]:
                gui_info.content = (f"_view/coord mismatch for {scene_name}: "
                                    f"{view.shape[0]} views vs {coord.shape[0]} pts_")
                return
            ids = np.unique(view)
            for v in ids:
                pts = coord[view == v]
                if pts.shape[0] == 0:
                    continue
                h = _add_cloud(f"/scene/view/{int(v):02d}", pts, view_color(v))
                with gui_folder:
                    cb = server.gui.add_checkbox(
                        f"view {int(v)} ({pts.shape[0]:,})", initial_value=True)
                cb.on_update(lambda _, hh=h, cc=cb: setattr(hh, "visible", cc.value))
                st["clouds"].append(h)
                st["rows"].append({"handle": h, "cb": cb, "score": 1.0})
                st["folder_handles"].append(cb)
            counts = ", ".join(f"{int(v)}:{int((view == v).sum()):,}" for v in ids)
            gui_info.content = (f"**{scene_name}** — View (camera) · {len(ids)} views · "
                                f"{coord.shape[0]:,} pts ({counts})")
            return

        if model.startswith(SPLITCAUSE_PREFIX):
            src_model = model[len(SPLITCAUSE_PREFIX):]
            if (src_model, scene_name) not in index:
                gui_info.content = f"_no result for {src_model} / {scene_name}_"
                return
            data = load_npz(index[(src_model, scene_name)])
            if "split_cause" not in data:
                gui_info.content = (f"_no split-cause fields in {src_model}/{scene_name}; "
                                    f"re-run tools/infer_insseg.py with --highlight-split-cause_")
                return
            coord, inst_id = data["coord"], data["inst_id"]
            seam, frag = data["split_cause"], data["split_fragment"]
            ic = data["inst_color"]
            assigned = inst_id >= 0
            frag_m = frag & ~seam
            ctx_m = assigned & ~frag & ~seam
            un_m = (~assigned) & ~seam
            ctx_col = (ic[inst_id[ctx_m]].astype(np.float32) * 0.35
                       + 160 * 0.65).astype(np.uint8) if ctx_m.any() else GRAY
            # seam drawn on top (added last); context/unassigned default off so the
            # carved-off fragments + red seams read clearly, toggle the rest back on.
            groups = [
                ("context (dimmed)", coord[ctx_m], ctx_col, False),
                ("unassigned (gray)", coord[un_m], GRAY, False),
                ("fragment instances", coord[frag_m],
                 ic[inst_id[frag_m]] if frag_m.any() else GRAY, True),
                ("seam (cause · red)", coord[seam], np.array([255, 0, 0], np.uint8), True),
            ]
            for label, pts, col, default_on in groups:
                if pts.shape[0] == 0:
                    continue
                h = _add_cloud(f"/scene/splitcause/{label.split()[0]}", pts, col)
                h.visible = default_on
                with gui_folder:
                    cb = server.gui.add_checkbox(f"{label} ({pts.shape[0]:,})",
                                                 initial_value=default_on)
                cb.on_update(lambda _, hh=h, cc=cb: setattr(hh, "visible", cc.value))
                st["clouds"].append(h)
                st["rows"].append({"handle": h, "cb": cb, "score": 1.0})
                st["folder_handles"].append(cb)
            n_frag_inst = int(np.unique(inst_id[seam & assigned]).size)
            gui_info.content = (f"**{scene_name}** — Split-cause · {int(seam.sum()):,} seam "
                                f"pts (red), {n_frag_inst} fragment instances · "
                                f"{coord.shape[0]:,} pts")
            return

        if (model, scene_name) not in index:
            gui_info.content = f"_no result for {model} / {scene_name}_"
            return
        d = load_npz(index[(model, scene_name)])
        coord, inst_id = d["coord"], d["inst_id"]
        n_inst = d["inst_color"].shape[0]
        thresh = gui_thresh.value

        # unassigned (gray) cloud + its toggle
        un = coord[inst_id < 0]
        if un.shape[0]:
            st["unassigned"] = _add_cloud("/scene/unassigned", un, GRAY)
        with gui_folder:
            cb_un = server.gui.add_checkbox("unassigned (gray)", initial_value=True)
            st["folder_handles"].append(cb_un)
        if st["unassigned"] is not None:
            cb_un.on_update(
                lambda _, h=st["unassigned"], cb=cb_un: setattr(h, "visible", cb.value))

        # one cloud + one checkbox per instance
        for k in range(n_inst):
            pts = coord[inst_id == k]
            if pts.shape[0] == 0:
                continue
            cls = int(d["inst_class"][k])
            cname = d["class_names"][cls] if 0 <= cls < len(d["class_names"]) else str(cls)
            score = float(d["inst_score"][k])
            h = _add_cloud(f"/scene/inst/{k:03d}", pts, d["inst_color"][k])
            with gui_folder:
                cb = server.gui.add_checkbox(
                    f"#{k:02d} {cname} {score:.2f} ({pts.shape[0]:,})",
                    initial_value=(score >= thresh))
            h.visible = score >= thresh
            cb.on_update(lambda _, hh=h, cc=cb: setattr(hh, "visible", cc.value))
            st["clouds"].append(h)
            st["rows"].append({"handle": h, "cb": cb, "score": score})
            st["folder_handles"].append(cb)

        n_pts = coord.shape[0]
        gui_info.content = (f"**{scene_name}** — {model} · {n_inst} instances · "
                            f"{n_pts:,} pts ({(inst_id < 0).sum():,} unassigned)")

    # --- callbacks -----------------------------------------------------------
    @gui_scene.on_update
    def _(_):
        render()

    @gui_model.on_update
    def _(_):
        render()

    @gui_psize.on_update
    def _(_):
        for h in st["clouds"]:
            h.point_size = gui_psize.value
        if st["unassigned"] is not None:
            st["unassigned"].point_size = gui_psize.value

    @gui_thresh.on_update
    def _(_):
        for row in st["rows"]:
            vis = row["score"] >= gui_thresh.value
            row["cb"].value = vis
            row["handle"].visible = vis

    @gui_show.on_click
    def _(_):
        for row in st["rows"]:
            row["cb"].value = True
            row["handle"].visible = True

    @gui_hide.on_click
    def _(_):
        for row in st["rows"]:
            row["cb"].value = False
            row["handle"].visible = False

    render()
    print(f"[viewer] serving at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
