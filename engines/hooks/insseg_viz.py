import os
import colorsys

import numpy as np
import trimesh


GRAY = np.array([160, 160, 160], dtype=np.uint8)


def voxel_downsample_indices(coord, grid_size):
    """Return indices of first occurrence per voxel cell.

    `coord` is an (N, 3) float array (any units; `grid_size` uses the same unit).
    Caller subsets all per-point arrays with the returned indices.
    """
    keys = np.floor(coord / grid_size).astype(np.int64)
    h = (
        keys[:, 0] * np.int64(73856093)
        ^ keys[:, 1] * np.int64(19349663)
        ^ keys[:, 2] * np.int64(83492791)
    )
    _, uniq = np.unique(h, return_index=True)
    uniq.sort()
    return uniq


def instance_palette(n, seed=0):
    """Deterministic distinct RGB palette via golden-ratio hue spread."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    hues = (np.arange(n) * 0.61803398875 + seed * 0.137) % 1.0
    rgb = np.array(
        [colorsys.hsv_to_rgb(h, 0.65, 0.95) for h in hues], dtype=np.float32
    )
    return (rgb * 255.0).astype(np.uint8)


def aabb_edges(min_xyz, max_xyz, color, radius):
    """Return a trimesh.Trimesh of 12 thin cylinders forming an AABB wireframe."""
    mn = np.asarray(min_xyz, dtype=np.float32)
    mx = np.asarray(max_xyz, dtype=np.float32)
    corners = np.array(
        [
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mx[2]],
            [mn[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    color4 = np.append(np.asarray(color, dtype=np.uint8), np.uint8(255))
    cyls = []
    for a, b in edges:
        seg = np.array([corners[a], corners[b]], dtype=np.float32)
        if np.linalg.norm(seg[1] - seg[0]) < 1e-9:
            continue
        cyl = trimesh.creation.cylinder(radius=radius, segment=seg, sections=4)
        cyl.visual.face_colors = np.tile(color4, (cyl.faces.shape[0], 1))
        cyls.append(cyl)
    if not cyls:
        return None
    return trimesh.util.concatenate(cyls)


def save_pointcloud_glb(path, coord, color_uint8, boxes=None, box_radius=0.01):
    """Save points (and optional AABB wireframes) as a GLB.

    `boxes` is an iterable of `(min_xyz, max_xyz, rgb_uint8)` tuples.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cloud = trimesh.PointCloud(
        vertices=np.asarray(coord, dtype=np.float32),
        colors=np.asarray(color_uint8, dtype=np.uint8),
    )
    if not boxes:
        cloud.export(path)
        return
    geometries = [cloud]
    for mn, mx, c in boxes:
        wire = aabb_edges(mn, mx, c, box_radius)
        if wire is not None:
            geometries.append(wire)
    trimesh.Scene(geometries).export(path)


def assign_instance_ids(num_points, pred_masks, pred_scores):
    """Winner-take-all instance id per point: the highest-scoring predicted
    instance that contains it. Unassigned points get -1.

    pred_masks: bool array (P, num_points)
    pred_scores: float array (P,)
    Returns: int32 array (num_points,) holding the proposal index, or -1.
    """
    inst_id = np.full(num_points, -1, dtype=np.int32)
    if pred_masks.shape[0] == 0:
        return inst_id
    order = np.argsort(-np.asarray(pred_scores))
    assigned = np.zeros(num_points, dtype=bool)
    for proposal_idx in order:
        mask = pred_masks[proposal_idx] & ~assigned
        if not mask.any():
            continue
        inst_id[mask] = proposal_idx
        assigned |= mask
    return inst_id


def colorize_predicted_instances(num_points, pred_masks, pred_scores):
    """Assign each point the color of the highest-scoring predicted instance
    that contains it; unassigned points stay gray.

    pred_masks: bool array (P, num_points)
    pred_scores: float array (P,)
    """
    out = np.tile(GRAY, (num_points, 1))
    if pred_masks.shape[0] == 0:
        return out
    inst_id = assign_instance_ids(num_points, pred_masks, pred_scores)
    palette = instance_palette(pred_masks.shape[0])
    assigned = inst_id >= 0
    out[assigned] = palette[inst_id[assigned]]
    return out


def colorize_gt_instances(instance_ids, ignore_value=-1):
    """Assign a deterministic color per unique GT instance id; ignored points
    (== ignore_value) stay gray."""
    n = instance_ids.shape[0]
    out = np.tile(GRAY, (n, 1))
    valid = instance_ids != ignore_value
    uniq = np.unique(instance_ids[valid])
    if uniq.size == 0:
        return out
    palette = instance_palette(uniq.size, seed=1)
    lookup = {int(inst): palette[k] for k, inst in enumerate(uniq)}
    for inst, color in lookup.items():
        out[instance_ids == inst] = color
    return out
