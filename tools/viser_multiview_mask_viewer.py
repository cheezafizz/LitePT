"""
Localhost viser viewer for the ALL-VIEWS multiview mask labels written by
tools/build_mask_labels_allviews.sh (mask_per_view.npy, N x 7).

Lays out all 7 camera views of a scene SIDE BY SIDE so the cross-view consistency of the
2D masks can be eyeballed. View s is the SAME fused point cloud, offset along +x and
colored by that view's mask column `mask_per_view[:, s]` (a fixed palette over the
view-local mask ids; -1 = not visible/unlabeled -> gray). Above each cloud float two
upright billboards: the view's RGB camera image and its 2D COCO mask overlay, the overlay
colored with the SAME palette so the 2D mask <-> 3D point coloring line up exactly.

Reads ONLY numpy / json / viser (no repo imports, no torch/cv2/pycocotools), so it runs in
the lightweight viewer venv from tools/setup_viser_venv.sh:

  .venv-viser/bin/python tools/viser_multiview_mask_viewer.py --port 8081
  # then open http://localhost:8081

Inputs per scene `<loc>_<frame>` (split on the FIRST underscore):
  --scene-data-dir/<scene>/{coord,color,mask_per_view}.npy   (the fused cloud + labels)
  --source-root/<loc>/<frame>/data.npz                       (rgb, input_order_list)
  --mask-root/<loc>/<frame>/coco_annotations.json            (aligned 518x392 COCO masks)
Billboards degrade gracefully (skipped, noted in the info panel) if data.npz / coco is
missing or a segmentation is compressed RLE.
"""
import argparse
import colorsys
import glob
import json
import os
import time

import numpy as np
import viser

GRAY = np.array([150, 150, 150], dtype=np.uint8)
MASK_MODE = "Mask (per-view)"
RGB_MODE = "RGB (input)"
# upright billboard: +90 deg about x -> image vertical maps to world +z, normal to world -y.
_BILLBOARD_WXYZ = (0.70710678, 0.70710678, 0.0, 0.0)


# ---------------------------------------------------------------------------
# COCO -> per-view global-id label image. Copied VERBATIM (pure numpy/json) from
# tools/vggt_to_scene.py (_decode_rle / _view_token / _load_coco_per_instance /
# _view_label_image + the `offsets` cumsum), so the overlay ids EQUAL mask_per_view ids.
# ---------------------------------------------------------------------------
def _decode_rle(seg):
    """Decode a COCO segmentation (uncompressed list-RLE) to (H,W) uint8."""
    h, w = int(seg["size"][0]), int(seg["size"][1])
    counts = seg["counts"]
    if isinstance(counts, (list, tuple)):
        flat = np.zeros(h * w, dtype=np.uint8)
        idx = 0
        val = 0
        for c in counts:
            if val:
                flat[idx : idx + c] = 1
            idx += c
            val ^= 1
        return flat.reshape((w, h)).T  # column-major -> (H, W)
    import pycocotools.mask as mask_util  # compressed RLE (not present in this data)
    return mask_util.decode(seg).astype(np.uint8)


def _view_token(file_name):
    """Map a COCO image file_name to its camera view token (e.g. 'TB')."""
    base = os.path.splitext(os.path.basename(file_name))[0]
    first = base.split("_")[0]
    if "-" in first:
        first = first.split("-")[-1]
    return first


def _load_coco_per_instance(coco_path, input_order_list, original_size):
    """Per-view list of (S_i, H0, W0) uint8 masks (category_id != 0), in view order."""
    W0, H0 = int(original_size[0]), int(original_size[1])
    with open(coco_path, "r") as f:
        coco = json.load(f)
    token_to_id = {_view_token(img["file_name"]): img["id"] for img in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    per_view = []
    for view in input_order_list:
        chans = []
        img_id = token_to_id.get(str(view))
        if img_id is not None:
            for ann in anns_by_img.get(img_id, []):
                if ann.get("category_id", 0) == 0:
                    continue
                m = _decode_rle(ann["segmentation"])
                if m.shape != (H0, W0):
                    m = m[:H0, :W0]
                chans.append((m > 0).astype(np.uint8) * 255)
        per_view.append(
            np.stack(chans, axis=0) if chans else np.zeros((0, H0, W0), dtype=np.uint8)
        )
    return per_view


def _view_label_image(aligned_s, areas_s, offset_s, H, W):
    """Per-pixel global mask id for one view (smallest-area instance wins), -1 if none."""
    lab = np.full((H, W), -1, dtype=np.int32)
    if aligned_s.shape[0] == 0:
        return lab
    best_area = np.full((H, W), np.iinfo(np.int64).max, dtype=np.int64)
    for c in range(aligned_s.shape[0]):
        take = aligned_s[c] & (int(areas_s[c]) < best_area)
        lab[take] = int(offset_s + c)
        best_area[take] = int(areas_s[c])
    return lab


# ---------------------------------------------------------------------------
def make_palette(n):
    """(n,3) uint8 golden-ratio HSV palette; palette[id] is stable across cloud + overlay."""
    n = max(int(n), 1)
    cols = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.6180339887) % 1.0, 0.65, 1.0)
        cols[i] = (round(r * 255), round(g * 255), round(b * 255))
    return cols


def scene_to_loc_frame(scene):
    """`<loc>_<frame>` -> (loc, frame); split on the FIRST underscore (loc has no '_')."""
    loc, _, frame = scene.partition("_")
    return loc, frame


def discover_scenes(scene_data_dir):
    if not os.path.isdir(scene_data_dir):
        return []
    return sorted(
        d for d in os.listdir(scene_data_dir)
        if os.path.isfile(os.path.join(scene_data_dir, d, "mask_per_view.npy"))
    )


def load_scene(scene, args):
    """Load cloud + labels (required) and rgb/coco-derived billboards (optional).

    Returns a dict; `notes` lists why billboards were skipped (shown in the info panel).
    """
    sdir = os.path.join(args.scene_data_dir, scene)
    coord = np.load(os.path.join(sdir, "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(sdir, "color.npy")).astype(np.uint8)
    mask_per_view = np.load(os.path.join(sdir, "mask_per_view.npy")).astype(np.int32)
    n_views = mask_per_view.shape[1]

    notes = []
    rgb = None                       # (S,H,W,3) u8
    order = [f"view{s}" for s in range(n_views)]
    label_imgs = [None] * n_views    # (H,W) i32 global ids, or None
    offsets_max = int(mask_per_view.max()) + 1 if mask_per_view.size else 1

    loc, frame = scene_to_loc_frame(scene)
    npz_path = os.path.join(args.source_root, loc, frame, "data.npz")
    coco_path = os.path.join(args.mask_root, loc, frame, "coco_annotations.json")

    if os.path.isfile(npz_path):
        z = np.load(npz_path, allow_pickle=False)
        if "rgb" in z.files:
            rgb = z["rgb"]
            if rgb.dtype != np.uint8:  # float [0,1] -> u8
                rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
        if "input_order_list" in z.files:
            order = [str(v) for v in z["input_order_list"]]
    else:
        notes.append(f"no data.npz ({npz_path}) -> generic view labels, no RGB billboard")

    if os.path.isfile(coco_path) and rgb is not None:
        try:
            H, W = rgb.shape[1], rgb.shape[2]
            per_view_raw = _load_coco_per_instance(coco_path, order, (W, H))
            offsets = np.cumsum([0] + [m.shape[0] for m in per_view_raw]).astype(np.int64)
            offsets_max = max(offsets_max, int(offsets[-1]))
            for s in range(min(n_views, len(per_view_raw))):
                aligned_s = per_view_raw[s] > 0
                areas_s = aligned_s.reshape(aligned_s.shape[0], -1).sum(1).astype(np.int64)
                label_imgs[s] = _view_label_image(aligned_s, areas_s, int(offsets[s]), H, W)
        except Exception as e:  # e.g. compressed RLE needing pycocotools
            notes.append(f"coco decode failed ({type(e).__name__}: {e}) -> no 2D mask overlay")
    elif rgb is not None and not os.path.isfile(coco_path):
        notes.append(f"no coco ({coco_path}) -> no 2D mask overlay")

    palette = make_palette(offsets_max)
    labeled_counts = [int((mask_per_view[:, s] >= 0).sum()) for s in range(n_views)]
    return {
        "scene": scene, "coord": coord, "color": color, "mask_per_view": mask_per_view,
        "n_views": n_views, "rgb": rgb, "order": order, "label_imgs": label_imgs,
        "palette": palette, "labeled_counts": labeled_counts, "notes": notes,
    }


def build_overlay(rgb_s, label_img_s, palette, alpha):
    """RGB image with the 2D mask painted on labeled pixels at `alpha` (same palette)."""
    out = rgb_s.astype(np.float32)
    lab = label_img_s >= 0
    if lab.any():
        col = palette[label_img_s[lab]].astype(np.float32)
        out[lab] = (1.0 - alpha) * out[lab] + alpha * col
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-data-dir",
                    default="data/scannet-v1.1.1-2of3-vggt-mask-allviews/test",
                    help="dir with <scene>/{coord,color,mask_per_view}.npy")
    ap.add_argument("--source-root",
                    default="/home/fai/workspace/jhp/dataset/val_sample_output",
                    help="dir with <loc>/<frame>/data.npz (rgb, input_order_list)")
    ap.add_argument("--mask-root",
                    default="/home/fai/workspace/jhp/dataset/val_sample_output_aligned",
                    help="dir with <loc>/<frame>/coco_annotations.json (aligned 518x392)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--point-size", type=float, default=0.004,
                    help="initial point size in meters (2 mm voxel grid -> ~4 mm reads well)")
    ap.add_argument("--col-spacing", type=float, default=0.0,
                    help="x offset between views in meters (0 = auto from scene x-extent)")
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

    # --- GUI controls --------------------------------------------------------
    gui_scene = server.gui.add_dropdown("Scene", scenes, initial_value=init_scene)
    gui_color = server.gui.add_dropdown("Color by", [MASK_MODE, RGB_MODE],
                                        initial_value=MASK_MODE)
    gui_psize = server.gui.add_slider("Point size (m)", min=0.001, max=0.02,
                                      step=0.001, initial_value=args.point_size)
    gui_alpha = server.gui.add_slider("Overlay alpha", min=0.0, max=1.0, step=0.05,
                                      initial_value=float(np.clip(args.overlay_alpha, 0, 1)))
    gui_unlabeled = server.gui.add_checkbox("Show unlabeled (gray)", initial_value=True)
    gui_rgb_bb = server.gui.add_checkbox("Show RGB billboards", initial_value=True)
    gui_mask_bb = server.gui.add_checkbox("Show 2D mask billboards", initial_value=True)
    gui_show = server.gui.add_button("Show all views")
    gui_hide = server.gui.add_button("Hide all views")
    gui_info = server.gui.add_markdown("")
    gui_folder = server.gui.add_folder("Views")

    # mutable state so nested callbacks can reach the live handles
    st = {"frames": [], "labeled": [], "unlabeled": [], "rgb_img": [], "mask_img": [],
          "labels": [], "view_cb": [], "cache": {}}

    def _clear():
        for key in ("labeled", "unlabeled", "rgb_img", "mask_img", "labels", "frames",
                    "view_cb"):
            for h in st[key]:
                if h is not None:
                    h.remove()
            st[key] = []

    def render():
        _clear()
        scene = gui_scene.value
        if scene not in st["cache"]:
            st["cache"][scene] = load_scene(scene, args)
        d = st["cache"][scene]
        coord, color, mpv = d["coord"], d["color"], d["mask_per_view"]
        palette, S = d["palette"], d["n_views"]
        by_rgb = gui_color.value == RGB_MODE

        spacing = args.col_spacing if args.col_spacing > 0 else np.ptp(coord[:, 0]) * 1.25 + 0.1
        rw = max(0.6, float(np.ptp(coord[:, 0])))          # billboard width (m)
        rh = rw * 392.0 / 518.0                            # billboard height (m)
        cx, cy = float(coord[:, 0].mean()), float(coord[:, 1].mean())
        z_top = float(coord[:, 2].max())
        rgb_cz = z_top + 0.15 + rh / 2.0
        mask_cz = rgb_cz + rh + 0.1
        label_cz = mask_cz + rh / 2.0 + 0.1

        for s in range(S):
            off = np.array([s * spacing, 0.0, 0.0], dtype=np.float32)
            frame = server.scene.add_frame(f"/views/{s:02d}", show_axes=False)
            st["frames"].append(frame)

            sel = mpv[:, s] >= 0
            pts = (coord + off).astype(np.float32)
            # labeled points
            if sel.any():
                lcol = color[sel] if by_rgb else palette[mpv[sel, s]]
                st["labeled"].append(server.scene.add_point_cloud(
                    f"/views/{s:02d}/labeled", points=pts[sel],
                    colors=lcol.astype(np.uint8), point_size=gui_psize.value,
                    point_shape="circle"))
            else:
                st["labeled"].append(None)
            # unlabeled points (toggleable)
            un = ~sel
            if un.any():
                ucol = color[un] if by_rgb else np.tile(GRAY, (int(un.sum()), 1))
                h = server.scene.add_point_cloud(
                    f"/views/{s:02d}/unlabeled", points=pts[un],
                    colors=ucol.astype(np.uint8), point_size=gui_psize.value,
                    point_shape="circle")
                h.visible = gui_unlabeled.value
                st["unlabeled"].append(h)
            else:
                st["unlabeled"].append(None)

            # billboards (optional)
            rgb = d["rgb"]
            if rgb is not None and s < rgb.shape[0]:
                h = server.scene.add_image(
                    f"/views/{s:02d}/rgb", image=rgb[s], render_width=rw, render_height=rh,
                    format="jpeg", wxyz=_BILLBOARD_WXYZ, position=(cx + s * spacing, cy, rgb_cz))
                h.visible = gui_rgb_bb.value
                st["rgb_img"].append(h)
                li = d["label_imgs"][s]
                if li is not None:
                    ov = build_overlay(rgb[s], li, palette, gui_alpha.value)
                    hm = server.scene.add_image(
                        f"/views/{s:02d}/mask2d", image=ov, render_width=rw, render_height=rh,
                        format="png", wxyz=_BILLBOARD_WXYZ,
                        position=(cx + s * spacing, cy, mask_cz))
                    hm.visible = gui_mask_bb.value
                    st["mask_img"].append(hm)
                else:
                    st["mask_img"].append(None)
            else:
                st["rgb_img"].append(None)
                st["mask_img"].append(None)

            st["labels"].append(server.scene.add_label(
                f"/views/{s:02d}/label", text=str(d["order"][s]),
                position=(cx + s * spacing, cy, label_cz)))

            with gui_folder:
                cb = server.gui.add_checkbox(
                    f"{d['order'][s]} ({d['labeled_counts'][s]:,} labeled)",
                    initial_value=True)
            cb.on_update(lambda _, fr=frame, c=cb: setattr(fr, "visible", c.value))
            st["view_cb"].append(cb)

        notes = ("  \n".join(f"_⚠ {n}_" for n in d["notes"]) + "  \n") if d["notes"] else ""
        counts = ", ".join(f"{d['order'][s]}:{d['labeled_counts'][s]:,}" for s in range(S))
        gui_info.content = (
            f"**{scene}** — {S} views · {coord.shape[0]:,} pts · "
            f"{'RGB' if by_rgb else 'mask'} coloring  \n{notes}labeled/view: {counts}")

    # --- callbacks -----------------------------------------------------------
    @gui_scene.on_update
    def _(_):
        render()

    @gui_color.on_update
    def _(_):
        render()

    @gui_alpha.on_update
    def _(_):
        render()  # rebuild overlays at the new alpha (cloud/rgb reused from cache)

    @gui_psize.on_update
    def _(_):
        for h in st["labeled"] + st["unlabeled"]:
            if h is not None:
                h.point_size = gui_psize.value

    @gui_unlabeled.on_update
    def _(_):
        for h in st["unlabeled"]:
            if h is not None:
                h.visible = gui_unlabeled.value

    @gui_rgb_bb.on_update
    def _(_):
        for h in st["rgb_img"]:
            if h is not None:
                h.visible = gui_rgb_bb.value

    @gui_mask_bb.on_update
    def _(_):
        for h in st["mask_img"]:
            if h is not None:
                h.visible = gui_mask_bb.value

    @gui_show.on_click
    def _(_):
        for cb in st["view_cb"]:
            cb.value = True

    @gui_hide.on_click
    def _(_):
        for cb in st["view_cb"]:
            cb.value = False

    render()
    print(f"[viewer] serving at http://{args.host}:{args.port}  (Ctrl-C to stop)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
