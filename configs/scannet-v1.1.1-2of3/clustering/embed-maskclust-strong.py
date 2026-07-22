# Inference-time CLUSTERING config for tools/infer_insseg.py (--cluster-config).
#
# Read SEPARATELY from the training config: the training config builds the backbone/heads
# and selects the checkpoint; this file decides how the per-point predictions are grouped
# into instances.
#
# Variant: GEOMETRIC + EMBEDDING SPLIT + 2D-MASK CANNOT-LINK (multi-view, STRONG).
#   * Embedding head IS used.                           -> instance_embedding = True
#   * Distinct-mask / same-view points must NOT connect -> cluster_mask_constraint = True
#   * Cannot-link fires in ANY shared view              -> mask_constraint_multiview = True
#   * Cannot-link enforced TRANSITIVELY (no detour leak)-> mask_constraint_strict = True
#
# Strict superset of clustering/embed-maskclust-allviews.py. The non-strict path enforces
# the cannot-link only as a LOCAL edge deletion and then takes (transitive) connected
# components, so two points in distinct 2D masks of the same view can still end up in one
# instance via a DETOUR of legal edges -- e.g. an unlabeled seam point (mask == -1 in the
# conflicting view) bridges the two touching objects, re-merging exactly what the split was
# meant to separate. With mask_constraint_strict=True, any component that leaked two masks
# is re-split by constrained region-growing, so NO predicted instance contains any
# same-view/distinct-mask pair, adjacent or not (the constraint is transitively closed).
#
# Output is IDENTICAL to -allviews on conflict-free proposals (a dense single-object or
# all-unlabeled slab takes the same fast scipy path); it only diverges -- producing more,
# smaller instances -- at touching-object seams where the leak actually fired. Clean A/B
# vs embed-maskclust-allviews.py.
#
# REQUIRES a checkpoint trained with instance_embedding=True. The constraint is a graceful
# no-op unless per-point per-view labels are supplied; tools/infer_insseg.py auto-feeds them
# from <scene-dir>/mask_per_view.npy (N, S) when present, and otherwise falls back to the
# single-view path (mask_instance/mask_view), which the strict split also supports.
#
# Effective grouping radius = cluster_thresh * voxel_size = 2.5 * 0.002 m = 5 mm.

cluster = dict(
    # --- geometric ball-query + BFS connected components (always active) ---
    voxel_size=0.002,
    cluster_thresh=2.5,           # 2.5 * 0.002 m = 5 mm physical radius
    cluster_closed_points=3000,
    cluster_propose_points=300,
    cluster_min_points=100,
    segment_ignore_index=(-1, 0, 1),

    # --- embedding-head split: ON ---
    instance_embedding=True,
    embed_bandwidth=1.5,
    embed_min_points=100,

    # --- 2D-mask cannot-link: ON (multi-view, strong) ---
    cluster_mask_constraint=True,
    mask_constraint_multiview=True,  # cannot-link in ANY shared view (needs mask_per_view.npy)
    mask_constraint_strict=True,     # transitively closed: detour paths can no longer re-merge
    mask_split_radius=2.5,           # adjacency radius (voxel units) for the constrained split; = cluster_thresh -> 5 mm
)
