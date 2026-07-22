# Inference-time CLUSTERING config for tools/infer_insseg.py (--cluster-config).
#
# Read SEPARATELY from the training config: the training config builds the backbone/heads
# and selects the checkpoint; this file decides how the per-point predictions are grouped
# into instances.
#
# Variant: GEOMETRIC + EMBEDDING SPLIT + 2D-MASK CANNOT-LINK (multi-view, STRONG) + 2D-MASK FILTER.
#   * Embedding head IS used.                           -> instance_embedding = True
#   * Distinct-mask / same-view points must NOT connect -> cluster_mask_constraint = True
#   * Cannot-link fires in ANY shared view              -> mask_constraint_multiview = True
#   * Cannot-link enforced TRANSITIVELY (no detour leak)-> mask_constraint_strict = True
#   * Points in NO 2D mask are dropped BEFORE grouping  -> cluster_mask_filter = True
#
# Superset of clustering/embed-maskclust-strong.py: the ONLY added knob is cluster_mask_filter.
# The "strong" config splits clusters along 2D-mask boundaries but KEEPS every point (all ~232k);
# this config additionally removes points that fall in NO 2D object mask BEFORE clustering, and
# tools/infer_insseg.py drops those unmasked points from the output cloud / GLB / npz entirely.
# Clean A/B vs embed-maskclust-strong.py (filter OFF): same strict cannot-link, but the output is
# restricted to the masked foreground -- background / unlabeled-seam points no longer appear and
# can no longer seed or pad an instance.
#
# REQUIRES a checkpoint trained with instance_embedding=True. Both the cannot-link and the filter
# are graceful no-ops unless per-point per-view labels are supplied; tools/infer_insseg.py
# auto-feeds them from <scene-dir>/mask_per_view.npy (N, S) when present, and otherwise falls back
# to the single-view path (mask_instance/mask_view). Force-disable the filter for A/B with
# --no-mask-filter (leaves the strict cannot-link intact -> reproduces embed-maskclust-strong.py).
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

    # --- 2D-mask pre-clustering filter: ON ---
    cluster_mask_filter=True,        # keep only points inside >=1 2D mask before clustering
)
