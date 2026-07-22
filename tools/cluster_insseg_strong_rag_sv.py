"""
Standalone, model-free instance clustering for a pre-filtered 'object'-class point
cloud, reproducing the *clustering* half of the ``embed-maskclust-strong-rag-sv`` config.

The caller passes in a point cloud that is ALREADY just the object-class points (no
neural network is run here); this module groups those points into instances with the
exact geometric + 2D-mask pipeline the config specifies, minus the parts that needed
the network:

    radius connected-components  ->  strict multi-view 2D-mask CANNOT-LINK split
                                 ->  RAG shared-view MERGE  ->  size filter

Dropped vs the full model pipeline (they required the network and are gone here):
  * the predicted per-point OFFSET (points used to be shifted toward instance
    centroids before grouping),
  * the learned EMBEDDING split,
  * the SEMANTIC head (all input points are already 'object', so class gating is moot).

Consequence: because there is no offset/embedding, the 2D masks are now the PRIMARY
mechanism that separates *touching* objects. Objects more than ~5 mm apart still split
geometrically; touching objects only split where ``mask_per_view`` says they belong to
different 2D masks. With ``mask_per_view=None`` the mask stages are skipped and the
result is plain geometric connected-components (touching objects merge) -- a warning is
printed.

PORTABILITY: this file depends ONLY on ``numpy`` and ``scipy`` -- no torch, no CUDA
``pointops``, no repo-internal imports. Drop it into any repository as-is.

Units: ``coord`` is (N, 3) in METERS. Radii params are in VOXEL units and become metric
as ``param * voxel_size`` (defaults: 2.5 * 0.002 m = 5 mm). ``mask_per_view`` is
(N, S) int: column s is the globally-unique 2D-mask id the point falls in for view s
(-1 = not visible / not covered there); mask ids are unique per (view, object).

CLI (numpy-only convenience / verification):
    python cluster_insseg_strong_rag_sv.py --scene-dir <dir_with_coord.npy[,mask_per_view.npy]>
    python cluster_insseg_strong_rag_sv.py --coord obj_coord.npy --mask-per-view mpv.npy --out inst.npy
"""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Verbatim embed-maskclust-strong-rag-sv values (the model-independent subset).
DEFAULTS = dict(
    voxel_size=0.002,
    cluster_thresh=2.5,            # geometric radius, voxel units -> 5 mm
    cluster_min_points=100,        # drop components / sub-components below this
    cluster_propose_points=300,    # final proposals must EXCEED this
    mask_split_radius=2.5,         # cannot-link adjacency radius, voxel units -> 5 mm
    mask_constraint_strict=True,
    rag_merge=True,
    rag_merge_thresh=0.3,          # MAX shared-view conflict ratio that still merges
    rag_adjacency_radius=2.5,      # RAG adjacency radius, voxel units -> 5 mm
)


# --- voxelization ---------------------------------------------------------------
def _voxel_downsample(coord, voxel_size):
    """Deterministic voxel grid downsample. Returns (grid_coord (G,3),
    origin_to_grid (N,) mapping each input point to its voxel's grid index,
    first_idx (G,) the input index of each voxel's representative -- its first-seen
    point). Per-point arrays (e.g. masks) remap to the grid via ``arr[first_idx]``."""
    keys = np.floor(coord / voxel_size).astype(np.int64)
    _, first_idx, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).ravel()          # numpy>=2 can return (N,1)
    return coord[first_idx], inverse, first_idx


# --- geometric grouping (replaces the CUDA ball-query + BFS) ---------------------
def _radius_components(n, pairs):
    """Connected-component labels (n,) for n nodes joined by undirected edges
    ``pairs`` ((M,2) indices). Isolated nodes are their own component."""
    if pairs.shape[0] == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(pairs.shape[0], dtype=np.uint8),
                    (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    return labels


def _geometric_components(coord, radius, min_points):
    """Radius connected-components on ``coord``; return the member index arrays of
    components with >= ``min_points`` points (all points are one class, so this is
    exactly the same-class BFS grouping the model's ball-query+BFS produced)."""
    n = coord.shape[0]
    pairs = cKDTree(coord).query_pairs(radius, output_type="ndarray")
    labels = _radius_components(n, pairs)
    members = []
    for r in np.unique(labels):
        idx = np.nonzero(labels == r)[0]
        if idx.size >= min_points:
            members.append(idx)
    return members


# --- strict transitive cannot-link refinement (ported verbatim) ------------------
def _strict_refine(base, pairs, mv):
    """Refine CC labels ``base`` (n,) into a TRANSITIVELY-CLOSED cannot-link partition
    over per-node per-view mask ids ``mv`` (n, S): no output component may hold two
    distinct valid mask ids in any single view. Ported from
    models/point_group/point_group_v1m2_custom_criteria.py:_strict_refine."""
    n = base.shape[0]
    S = mv.shape[1]
    bad_comps = set()
    for s in range(S):
        col = mv[:, s]
        valid = col >= 0
        if not valid.any():
            continue
        comp_v = base[valid]
        mask_v = col[valid]
        order = np.lexsort((mask_v, comp_v))
        cs = comp_v[order]
        ms = mask_v[order]
        change = np.zeros(cs.shape[0], dtype=bool)
        change[1:] = (cs[1:] == cs[:-1]) & (ms[1:] != ms[:-1])
        for c in np.unique(cs[change]):
            bad_comps.add(int(c))
    if not bad_comps:
        _, inv = np.unique(base, return_inverse=True)
        return np.asarray(inv).ravel().astype(np.int64)

    conflicted = np.isin(base, list(bad_comps))
    out = np.full(n, -1, dtype=np.int64)
    keep = ~conflicted
    next_label = 0
    if keep.any():
        _, inv = np.unique(base[keep], return_inverse=True)
        inv = np.asarray(inv).ravel()
        out[keep] = inv
        next_label = int(inv.max()) + 1

    conf_nodes = np.nonzero(conflicted)[0]
    adj = {int(i): [] for i in conf_nodes}
    if pairs.shape[0]:
        em = conflicted[pairs[:, 0]] & conflicted[pairs[:, 1]]
        for a, b in pairs[em]:
            a = int(a)
            b = int(b)
            adj[a].append(b)
            adj[b].append(a)
    seen = {}
    for seed in conf_nodes:
        seed = int(seed)
        if seed in seen:
            continue
        cluster_mask = {}
        for s in np.nonzero(mv[seed] >= 0)[0]:
            cluster_mask[int(s)] = int(mv[seed, s])
        seen[seed] = next_label
        stack = [seed]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if w in seen:
                    continue
                wv = mv[w]
                ok = True
                for s in np.nonzero(wv >= 0)[0]:
                    cur = cluster_mask.get(int(s))
                    if cur is not None and cur != int(wv[s]):
                        ok = False
                        break
                if not ok:
                    continue
                seen[w] = next_label
                for s in np.nonzero(wv >= 0)[0]:
                    cluster_mask[int(s)] = int(wv[s])
                stack.append(w)
        next_label += 1
    for node, lab in seen.items():
        out[node] = lab
    return out


def _split_by_mask_multiview(members, coord, mask_per_view, radius, strict, min_points):
    """Sub-divide each proposal so two points in DIFFERENT 2D masks of the SAME view
    are never grouped (cannot-link in ANY shared view). Ported from
    _split_proposals_by_mask_multiview."""
    new_members = []
    for idx in members:
        n = idx.shape[0]
        if n == 0:
            continue
        pts = coord[idx]
        mv = mask_per_view[idx]  # (n, S)
        all_pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
        pairs = all_pairs
        if all_pairs.shape[0]:
            va, vb = mv[all_pairs[:, 0]], mv[all_pairs[:, 1]]
            cannot = ((va >= 0) & (vb >= 0) & (va != vb)).any(axis=1)
            pairs = all_pairs[~cannot]
        labels = _radius_components(n, pairs)
        if strict:
            labels = _strict_refine(labels, pairs, mv)
        for r in np.unique(labels):
            member = labels == r
            if int(member.sum()) < min_points:
                continue
            new_members.append(idx[member])
    return new_members


# --- RAG shared-view merge (ported, shared_view metric only) ---------------------
def _merge_by_mask_histogram(members, coord, mask_per_view, radius, merge_thresh):
    """Heal over-segmentation by merging adjacent proposals whose co-visible points
    mostly agree on their 2D mask across shared views. Ported from
    _merge_proposals_by_mask_histogram with rag_sim_metric='shared_view' (the class
    check is dropped: every point is 'object')."""
    P = len(members)
    if P < 2:
        return members
    mv = mask_per_view
    max_id = int(mv.max()) + 1 if mv.size and mv.max() >= 0 else 0
    S = mv.shape[1] if mv.ndim == 2 else 0

    cnts = np.zeros((P, max_id), dtype=np.float64)   # count over global (view,mask) ids
    vc = np.zeros((P, S), dtype=np.float64)          # per-view visible counts
    for p, idx in enumerate(members):
        if idx.size == 0:
            continue
        mvp = mv[idx]                                # (n_p, S)
        vc[p] = (mvp >= 0).sum(axis=0)
        if max_id > 0:
            vals = mvp[mvp >= 0]
            if vals.size:
                cnts[p] = np.bincount(vals, minlength=max_id).astype(np.float64)

    # cross-cluster adjacency via one global tree
    owner = np.concatenate(
        [np.full(idx.size, p, dtype=np.int64) for p, idx in enumerate(members)])
    allpts = np.concatenate([coord[idx] for idx in members], axis=0)
    adj_pairs = cKDTree(allpts).query_pairs(radius, output_type="ndarray")
    adj = set()
    if adj_pairs.shape[0]:
        oa, ob = owner[adj_pairs[:, 0]], owner[adj_pairs[:, 1]]
        cross = oa != ob
        for a, b in zip(oa[cross].tolist(), ob[cross].tolist()):
            adj.add((a, b) if a < b else (b, a))

    ea, eb = [], []
    for a, b in adj:
        if S == 0:
            continue
        shared = (vc[a] > 0) & (vc[b] > 0)
        if not shared.any():
            continue                                 # no shared view -> no evidence
        den = float(vc[a] @ vc[b])
        if den <= 0:
            continue
        num = float(cnts[a] @ cnts[b])
        conflict = min(max(1.0 - num / den, 0.0), 1.0)
        if conflict <= merge_thresh:
            ea.append(a)
            eb.append(b)
    if not ea:
        return members

    g = coo_matrix((np.ones(len(ea), dtype=np.uint8), (ea, eb)), shape=(P, P))
    n_comp, comp = connected_components(g, directed=False)
    merged = []
    for c in range(n_comp):
        grp = np.nonzero(comp == c)[0]
        merged.append(np.concatenate([members[int(p)] for p in grp]))
    return merged


# --- public API -----------------------------------------------------------------
def cluster_object_points(coord, mask_per_view=None, *,
                          voxel_size=DEFAULTS["voxel_size"],
                          cluster_thresh=DEFAULTS["cluster_thresh"],
                          cluster_min_points=DEFAULTS["cluster_min_points"],
                          cluster_propose_points=DEFAULTS["cluster_propose_points"],
                          mask_split_radius=DEFAULTS["mask_split_radius"],
                          mask_constraint_strict=DEFAULTS["mask_constraint_strict"],
                          rag_merge=DEFAULTS["rag_merge"],
                          rag_merge_thresh=DEFAULTS["rag_merge_thresh"],
                          rag_adjacency_radius=DEFAULTS["rag_adjacency_radius"],
                          voxelize=True, verbose=True, return_details=False):
    """Cluster an object-class point cloud into instances (embed-maskclust-strong-rag-sv,
    model-free). See the module docstring for the contract.

    coord : (N, 3) float, meters -- already restricted to the 'object' class.
    mask_per_view : (N, S) int or None -- per-view 2D-mask id (-1 = absent). Enables the
        cannot-link split + RAG merge; None -> geometric-only (touching objects merge).
    voxelize : downsample at ``voxel_size`` before clustering (matches the config's
        point-count thresholds). Set False if ``coord`` is already a voxel grid.

    Returns (N,) int32 per-point instance id (-1 = unassigned), aligned with input coord.
    With return_details=True: dict(inst_id, n_instances, sizes).
    """
    coord = np.ascontiguousarray(coord, dtype=np.float64)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"coord must be (N,3); got {coord.shape}")
    n_input = coord.shape[0]

    if mask_per_view is not None:
        mask_per_view = np.ascontiguousarray(mask_per_view, dtype=np.int64)
        if mask_per_view.ndim != 2 or mask_per_view.shape[0] != n_input:
            raise ValueError(
                f"mask_per_view must be (N,S) with N={n_input}; got {mask_per_view.shape}")

    # 1. voxel downsample (grid cloud + input->grid map); masks remap via first_idx
    if voxelize:
        grid_coord, origin_to_grid, first_idx = _voxel_downsample(coord, voxel_size)
        grid_mpv = mask_per_view[first_idx] if mask_per_view is not None else None
    else:
        grid_coord = coord
        origin_to_grid = np.arange(n_input)
        grid_mpv = mask_per_view
    n_grid = grid_coord.shape[0]

    # 2. geometric connected components
    members = _geometric_components(
        grid_coord, radius=cluster_thresh * voxel_size, min_points=cluster_min_points)

    # 3 + 4. mask cannot-link split, then RAG merge (need mask_per_view)
    if grid_mpv is not None and len(members) > 0:
        members = _split_by_mask_multiview(
            members, grid_coord, grid_mpv,
            radius=mask_split_radius * voxel_size,
            strict=mask_constraint_strict, min_points=cluster_min_points)
        if rag_merge and len(members) > 1:
            members = _merge_by_mask_histogram(
                members, grid_coord, grid_mpv,
                radius=rag_adjacency_radius * voxel_size, merge_thresh=rag_merge_thresh)
    elif grid_mpv is None and verbose:
        print("[cluster] mask_per_view=None -> cannot-link & RAG merge SKIPPED; "
              "touching objects will merge into one instance (geometric only).")

    # 5. size filter (proposals must EXCEED cluster_propose_points)
    members = [m for m in members if m.size > cluster_propose_points]

    # 6. label grid points, remap onto the input cloud
    grid_label = np.full(n_grid, -1, dtype=np.int32)
    for i, idx in enumerate(members):
        grid_label[idx] = i
    inst_id = grid_label[origin_to_grid]

    if verbose:
        print(f"[cluster] {n_input:,} input pts -> {n_grid:,} grid pts | "
              f"{len(members)} instances | {int((inst_id >= 0).sum()):,} pts assigned")
    if return_details:
        sizes = np.array([int((inst_id == i).sum()) for i in range(len(members))],
                         dtype=np.int64)
        return dict(inst_id=inst_id, n_instances=len(members), sizes=sizes)
    return inst_id


# --- CLI ------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", default=None,
                    help="dir with coord.npy (object-only) [+ mask_per_view.npy]")
    ap.add_argument("--coord", default=None, help="object-only coord .npy (N,3)")
    ap.add_argument("--mask-per-view", default=None, help="mask_per_view .npy (N,S)")
    ap.add_argument("--out", default=None, help="output instance .npy (N,)")
    ap.add_argument("--no-voxelize", action="store_true",
                    help="coord is already a voxel grid; skip the internal downsample")
    ap.add_argument("--no-rag", action="store_true", help="disable the RAG merge")
    args = ap.parse_args()

    if args.scene_dir:
        coord_path = os.path.join(args.scene_dir, "coord.npy")
        mpv_path = os.path.join(args.scene_dir, "mask_per_view.npy")
        mpv_path = mpv_path if os.path.isfile(mpv_path) else None
        out_path = args.out or os.path.join(args.scene_dir, "instance.npy")
    else:
        if not args.coord:
            raise SystemExit("provide --scene-dir OR --coord")
        coord_path, mpv_path = args.coord, args.mask_per_view
        out_path = args.out or os.path.splitext(args.coord)[0] + "_instance.npy"

    coord = np.load(coord_path)
    mask_per_view = np.load(mpv_path) if mpv_path else None
    print(f"[scene] coord {coord.shape} | mask_per_view "
          f"{None if mask_per_view is None else mask_per_view.shape}")

    res = cluster_object_points(
        coord, mask_per_view, voxelize=not args.no_voxelize,
        rag_merge=not args.no_rag, return_details=True)
    order = np.argsort(-res["sizes"])
    for rank, pid in enumerate(order):
        print(f"   #{pid:03d} pts={int(res['sizes'][pid]):,}")
    np.save(out_path, res["inst_id"])
    print(f"[out] wrote {out_path}  ({res['inst_id'].shape[0]:,} per-point ids, "
          f"{res['n_instances']} instances, -1 = unassigned)")


if __name__ == "__main__":
    main()
