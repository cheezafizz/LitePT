"""
Localhost viser viewer for inspecting the PER-VIEW 2D-mask assignment of a VGGT scene
ONE VIEW AT A TIME, to diagnose mask-misassignment-driven over-segmentation.

Unlike tools/viser_multiview_mask_viewer.py (which fans all 7 views out side-by-side), this
viewer shows the SINGLE fused point cloud IN PLACE and lets you scroll through the 7 camera
views with a slider / ◀▶ buttons. For the selected view `s` every point is colored by the 2D
mask it lands in for that view (`mask_per_view[:, s]`, a fixed palette over the view-local mask
ids; -1 = occluded / no mask -> gray), and above the cloud float the view's RGB image and its 2D
COCO mask overlay, the overlay colored with the SAME palette so the 2D mask <-> 3D point coloring
line up exactly.

The point of the tool is the "misassignment" hypothesis: a point's mask in a NON-origin view is
chosen purely by reprojection (z-buffer occlusion + camera alignment), so a bad reprojection lands
it on the wrong object's 2D mask, and the multi-view cannot-link clustering then shatters the
object. The "Color by" modes make that visible:
  * Mask (this view)  -- the literal per-view assignment (palette over mask_per_view[:, s]).
  * Origin highlight  -- native (origin view == s) = GREEN, reprojected-with-mask
                         (origin != s, has a mask here) = ORANGE (the suspected misassignment set),
                         occluded / no mask here = dim gray.
  * Origin view       -- color every point by its origin view (mask_view), a tab10 palette.
  * RGB (input)       -- reference colors.
Plus an "Only native points (origin == this view)" filter to compare the trustworthy assignment
against the full cloud.

Reads ONLY numpy / json / viser (no repo imports, no torch/cv2/pycocotools) and REUSES the COCO /
palette / overlay helpers from tools/viser_multiview_mask_viewer.py, so it runs in the lightweight
viewer venv from tools/setup_viser_venv.sh:

  .venv-viser/bin/python tools/viser_perview_mask_viewer.py --port 8082
  # then open http://localhost:8082

Inputs per scene `<loc>_<frame>` (split on the FIRST underscore):
  --scene-data-dir/<scene>/{coord,color,mask_per_view,mask_view}.npy  (cloud + per-view + origin)
  --source-root/<loc>/<frame>/data.npz                                (rgb, input_order_list)
  --mask-root/<loc>/<frame>/coco_annotations.json                     (aligned 518x392 COCO masks)
Billboards degrade gracefully (skipped, noted in the info panel) if data.npz / coco is missing.
mask_per_view.npy / mask_view.npy come from tools/build_mask_labels_allviews.sh; the default
--scene-data-dir is that build's output (the same filtered+conf25 cloud as
data/scannet-vggt-default-filtered-conf25, with the per-view mask labels added).
"""
import argparse
import os
import sys
import time

import numpy as np
import viser

# Reuse the pure-numpy/json helpers from the side-by-side viewer (single source of truth for the
# COCO decode, the stable palette and the 2D-mask overlay, so colors match exactly).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viser_multiview_mask_viewer import (  # noqa: E402
    GRAY,
    MASK_MODE,
    RGB_MODE,
    _BILLBOARD_WXYZ,
    build_overlay,
    discover_scenes,
    load_scene as _load_scene_multiview,
)

ORIGIN_HL_MODE = "Origin highlight"
ORIGIN_VIEW_MODE = "Origin view"

DIM_GRAY = np.array([90, 90, 90], dtype=np.uint8)     # occluded / no mask in this view
NATIVE_GREEN = np.array([60, 200, 90], dtype=np.uint8)  # origin view == selected view
REPROJ_ORANGE = np.array([240, 140, 30], dtype=np.uint8)  # borrowed from another view (suspect)
# tab10 -- categorical palette for the "Origin view" mode (one color per camera view).
VIEW_PALETTE = np.array(
    [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189],
     [140, 86, 75], [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207]],
    dtype=np.uint8,
)


def load_scene(scene, args):
    """Reuse the multiview loader (cloud + mask_per_view + rgb/coco billboards), then add the
    origin-view arrays this viewer needs (mask_view, and mask_instance if present)."""
    d = _load_scene_multiview(scene, args)
    sdir = os.path.join(args.scene_data_dir, scene)
    n = d["coord"].shape[0]
    mv_path = os.path.join(sdir, "mask_view.npy")
    if os.path.isfile(mv_path):
        mask_view = np.load(mv_path).astype(np.int64)
    else:
        mask_view = np.full(n, -1, dtype=np.int64)
        d["notes"].append(f"no mask_view.npy ({mv_path}) -> origin-view modes unavailable")
    d["mask_view"] = mask_view
    return d


def compute_colors(d, s, mode):
    """(N,3) uint8 colors for the selected view `s` under the chosen `mode`."""
    coord, color, mpv, mask_view = d["coord"], d["color"], d["mask_per_view"], d["mask_view"]
    palette = d["palette"]
    col_view = mpv[:, s]
    has = col_view >= 0
    if mode == RGB_MODE:
        return color.astype(np.uint8)
    if mode == ORIGIN_VIEW_MODE:
        ov = np.where(mask_view >= 0, mask_view, 0)
        return VIEW_PALETTE[ov % len(VIEW_PALETTE)].astype(np.uint8)
    if mode == ORIGIN_HL_MODE:
        cols = np.tile(DIM_GRAY, (coord.shape[0], 1))
        cols[has & (mask_view == s)] = NATIVE_GREEN
        cols[has & (mask_view != s)] = REPROJ_ORANGE
        return cols.astype(np.uint8)
    # MASK_MODE (default): palette over the per-view mask id, gray where unlabeled.
    cols = np.tile(GRAY, (coord.shape[0], 1))
    if has.any():
        cols[has] = palette[np.clip(col_view[has], 0, len(palette) - 1)]
    return cols.astype(np.uint8)


def view_counts(d, s):
    """(n_labeled, n_native_with_mask, n_reproj_with_mask, n_occluded, n_origin) for view s."""
    mpv, mask_view = d["mask_per_view"], d["mask_view"]
    has = mpv[:, s] >= 0
    n_native = int((has & (mask_view == s)).sum())
    n_reproj = int((has & (mask_view != s)).sum())
    return (int(has.sum()), n_native, n_reproj, int((~has).sum()), int((mask_view == s).sum()))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-data-dir",
                    default="data/scannet-v1.1.1-2of3-vggt-mask-allviews/test",
                    help="dir with <scene>/{coord,color,mask_per_view,mask_view}.npy")
    ap.add_argument("--source-root",
                    default="/home/fai/workspace/jhp/dataset/val_sample_output",
                    help="dir with <loc>/<frame>/data.npz (rgb, input_order_list)")
    ap.add_argument("--mask-root",
                    default="/home/fai/workspace/jhp/dataset/val_sample_output_aligned",
                    help="dir with <loc>/<frame>/coco_annotations.json (aligned 518x392)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--point-size", type=float, default=0.004,
                    help="initial point size in meters (2 mm voxel grid -> ~4 mm reads well)")
    ap.add_argument("--overlay-alpha", type=float, default=0.6,
                    help="2D mask overlay blend over the RGB image [0..1]")
    ap.add_argument("--scene", default=None, help="initial scene (default: first found)")
    args = ap.parse_args()

    scenes = discover_scenes(args.scene_data_dir)
    if not scenes:
        raise SystemExit(f"no '<scene>/mask_per_view.npy' under {args.scene_data_dir!r}; "
                         f"run tools/build_mask_labels_allviews.sh first")
    init_scene = args.scene if (args.scene in scenes) else scenes[0]
    print(f"[viewer] {len(scenes)} scenes from {args.scene_data_dir}")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")

    # cache scenes (incl. the initial one, also used to size the view slider).
    cache = {init_scene: load_scene(init_scene, args)}
    max_views = max(1, cache[init_scene]["n_views"])

    # --- GUI controls --------------------------------------------------------
    gui_scene = server.gui.add_dropdown("Scene", scenes, initial_value=init_scene)
    gui_view = server.gui.add_slider("View index", min=0, max=max_views - 1, step=1,
                                     initial_value=0)
    gui_prev = server.gui.add_button("◀ prev view")
    gui_next = server.gui.add_button("▶ next view")
    gui_color = server.gui.add_dropdown(
        "Color by", [MASK_MODE, ORIGIN_HL_MODE, ORIGIN_VIEW_MODE, RGB_MODE],
        initial_value=MASK_MODE)
    gui_only_native = server.gui.add_checkbox("Only native points (origin == this view)",
                                              initial_value=False)
    gui_psize = server.gui.add_slider("Point size (m)", min=0.001, max=0.02, step=0.001,
                                      initial_value=args.point_size)
    gui_alpha = server.gui.add_slider("Overlay alpha", min=0.0, max=1.0, step=0.05,
                                      initial_value=float(np.clip(args.overlay_alpha, 0, 1)))
    gui_unlabeled = server.gui.add_checkbox("Show unlabeled / occluded", initial_value=True)
    gui_rgb_bb = server.gui.add_checkbox("Show RGB billboard", initial_value=True)
    gui_mask_bb = server.gui.add_checkbox("Show 2D mask billboard", initial_value=True)
    gui_info = server.gui.add_markdown("")

    st = {"handles": []}  # all per-render scene handles, removed on the next render

    def _clear():
        for h in st["handles"]:
            if h is not None:
                h.remove()
        st["handles"] = []

    def render():
        _clear()
        scene = gui_scene.value
        if scene not in cache:
            cache[scene] = load_scene(scene, args)
        d = cache[scene]
        coord, S = d["coord"], d["n_views"]
        s = int(np.clip(gui_view.value, 0, S - 1))
        mode = gui_color.value

        cols = compute_colors(d, s, mode)
        col_view = d["mask_per_view"][:, s]
        has = col_view >= 0
        base = (d["mask_view"] == s) if gui_only_native.value else np.ones(coord.shape[0], bool)
        labeled = base & has
        unlabeled = base & ~has
        pts = coord.astype(np.float32)

        if labeled.any():
            st["handles"].append(server.scene.add_point_cloud(
                "/cloud/labeled", points=pts[labeled], colors=cols[labeled],
                point_size=gui_psize.value, point_shape="circle"))
        if unlabeled.any():
            h = server.scene.add_point_cloud(
                "/cloud/unlabeled", points=pts[unlabeled], colors=cols[unlabeled],
                point_size=gui_psize.value, point_shape="circle")
            h.visible = gui_unlabeled.value
            st["handles"].append(h)

        # billboards for the SELECTED view, centered above the in-place cloud.
        rw = max(0.6, float(np.ptp(coord[:, 0])))
        rh = rw * 392.0 / 518.0
        cx, cy = float(coord[:, 0].mean()), float(coord[:, 1].mean())
        z_top = float(coord[:, 2].max())
        rgb_cz = z_top + 0.15 + rh / 2.0
        mask_cz = rgb_cz + rh + 0.1
        label_cz = mask_cz + rh / 2.0 + 0.1
        rgb = d["rgb"]
        if rgb is not None and s < rgb.shape[0]:
            h = server.scene.add_image("/cloud/rgb", image=rgb[s], render_width=rw,
                                       render_height=rh, format="jpeg", wxyz=_BILLBOARD_WXYZ,
                                       position=(cx, cy, rgb_cz))
            h.visible = gui_rgb_bb.value
            st["handles"].append(h)
            li = d["label_imgs"][s]
            if li is not None:
                ov = build_overlay(rgb[s], li, d["palette"], gui_alpha.value)
                hm = server.scene.add_image("/cloud/mask2d", image=ov, render_width=rw,
                                            render_height=rh, format="png", wxyz=_BILLBOARD_WXYZ,
                                            position=(cx, cy, mask_cz))
                hm.visible = gui_mask_bb.value
                st["handles"].append(hm)

        token = d["order"][s] if s < len(d["order"]) else f"view{s}"
        st["handles"].append(server.scene.add_label(
            "/cloud/label", text=f"view {s}: {token}", position=(cx, cy, label_cz)))

        n_lab, n_nat, n_rep, n_occ, n_org = view_counts(d, s)
        notes = ("  \n".join(f"_⚠ {n}_" for n in d["notes"]) + "  \n") if d["notes"] else ""
        gui_info.content = (
            f"**{scene}** — view **{s} ({token})** of {S} · {coord.shape[0]:,} pts · "
            f"{mode}  \n{notes}"
            f"labeled here: **{n_lab:,}**  \n"
            f"· native (origin==this view, with mask): **{n_nat:,}**  \n"
            f"· reprojected from another view (with mask): **{n_rep:,}**  ← misassignment suspects  \n"
            f"· occluded / no mask here: **{n_occ:,}**  \n"
            f"points whose origin view is this view: {n_org:,}")

    # --- callbacks -----------------------------------------------------------
    @gui_scene.on_update
    def _(_):
        render()

    @gui_view.on_update
    def _(_):
        render()

    @gui_color.on_update
    def _(_):
        render()

    @gui_only_native.on_update
    def _(_):
        render()

    @gui_alpha.on_update
    def _(_):
        render()  # rebuild overlay at the new alpha

    @gui_psize.on_update
    def _(_):
        for h in st["handles"]:
            if h is not None and hasattr(h, "point_size"):
                h.point_size = gui_psize.value

    @gui_unlabeled.on_update
    def _(_):
        render()

    @gui_rgb_bb.on_update
    def _(_):
        render()

    @gui_mask_bb.on_update
    def _(_):
        render()

    @gui_prev.on_click
    def _(_):
        gui_view.value = max(0, gui_view.value - 1)

    @gui_next.on_click
    def _(_):
        d = cache.get(gui_scene.value)
        hi = (d["n_views"] - 1) if d is not None else (max_views - 1)
        gui_view.value = min(hi, gui_view.value + 1)

    render()
    print(f"[viewer] serving at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
