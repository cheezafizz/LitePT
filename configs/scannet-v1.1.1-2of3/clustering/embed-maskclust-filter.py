# Inference-time CLUSTERING config for tools/infer_insseg.py (--cluster-config).
#
# Read SEPARATELY from the training config: the training config builds the backbone/heads
# and selects the checkpoint; this file decides how the per-point predictions are grouped
# into instances.
#
# Variant: GEOMETRIC + EMBEDDING SPLIT + 2D-MASK CANNOT-LINK (multi-view) + 2D-MASK FILTER.
#   * Embedding head IS used.                           -> instance_embedding = True
#   * Distinct-mask / same-view points must NOT connect -> cluster_mask_constraint = True
#   * Cannot-link fires in ANY shared view              -> mask_constraint_multiview = True
#   * Points in NO 2D mask are dropped BEFORE grouping  -> cluster_mask_filter = True
# Superset of clustering/embed-maskclust-allviews.py: the ONLY difference is
# cluster_mask_filter, which keeps only points falling inside at least one 2D object mask
# before clustering (and tools/infer_insseg.py removes the unmasked points from the output
# cloud / GLB / npz entirely). Clean A/B vs embed-maskclust-allviews (filter off).
#
# The filter is a graceful no-op unless per-point 2D-mask labels are supplied;
# tools/infer_insseg.py auto-feeds them from <scene-dir>/mask_per_view.npy (N, S) when
# present, and otherwise falls back to single-view (mask_instance/mask_view). Force-disable
# for A/B with --no-mask-filter.
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

    # --- 2D-mask cannot-link: ON (multi-view) ---
    cluster_mask_constraint=True,
    mask_constraint_multiview=True,  # cannot-link in ANY shared view (needs mask_per_view.npy)
    mask_split_radius=2.5,           # adjacency radius (voxel units) for the constrained split; = cluster_thresh -> 5 mm

    # --- 2D-mask pre-clustering filter: ON ---
    cluster_mask_filter=True,        # keep only points inside >=1 2D mask before clustering
)
