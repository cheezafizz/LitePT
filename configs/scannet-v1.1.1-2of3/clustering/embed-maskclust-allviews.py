# Inference-time CLUSTERING config for tools/infer_insseg.py (--cluster-config).
#
# Read SEPARATELY from the training config: the training config builds the backbone/heads
# and selects the checkpoint; this file decides how the per-point predictions are grouped
# into instances.
#
# Variant: GEOMETRIC + EMBEDDING SPLIT + 2D-MASK CANNOT-LINK (multi-view).
#   * Embedding head IS used.                           -> instance_embedding = True
#   * Distinct-mask / same-view points must NOT connect -> cluster_mask_constraint = True
#   * Cannot-link fires in ANY shared view              -> mask_constraint_multiview = True
# Mirrors insseg-litept-small-v1m2-2of3-embed-maskclust-allviews.py. Generalizes the
# single-view cannot-link: each point carries the 2D mask it lands in for EVERY view it is
# visible in (z-buffer-occluded), so two adjacent points are separated if they fall in
# different 2D masks in ANY view they share, not just their origin view. Points that share
# no labeled view, or agree in every shared view, still merge by adjacency.
#
# REQUIRES a checkpoint trained with instance_embedding=True. The multi-view cannot-link is
# a graceful no-op unless per-point per-view labels are supplied; tools/infer_insseg.py
# auto-feeds them from <scene-dir>/mask_per_view.npy (N, S) when present, and otherwise
# falls back to the single-view path (mask_instance/mask_view).
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
)
