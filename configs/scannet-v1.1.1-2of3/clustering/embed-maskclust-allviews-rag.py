# Inference-time CLUSTERING config for tools/infer_insseg.py (--cluster-config).
#
# Read SEPARATELY from the training config: the training config builds the backbone/heads
# and selects the checkpoint; this file decides how the per-point predictions are grouped
# into instances.
#
# Variant: GEOMETRIC + EMBEDDING SPLIT + 2D-MASK CANNOT-LINK (multi-view) + RAG MASK-HISTOGRAM MERGE.
#   * Embedding head IS used.                           -> instance_embedding = True
#   * Distinct-mask / same-view points must NOT connect -> cluster_mask_constraint = True
#   * Cannot-link fires in ANY shared view              -> mask_constraint_multiview = True
#   * Over-segmentation is HEALED after the split       -> cluster_mask_rag_merge = True
#
# Superset of clustering/embed-maskclust-allviews.py: the ONLY added behavior is the post-clustering
# Region-Adjacency-Graph merge. The multi-view cannot-link OVER-segments -- a single noisy point
# (a 2D mask mislabeled in one view) can sever a real object into fragments. This config adds a
# final, inverse step:
#   1. METHOD  -- for each over-segmented cluster, aggregate all its points' per-view 2D-mask ids
#                 into a cluster-level histogram of masks.
#   2. FIX     -- build a Region Adjacency Graph (nodes = clusters, edges = spatial adjacency) and
#                 merge adjacent SAME-CLASS clusters whose mask histograms are highly similar.
# Because the histogram aggregates hundreds of points, the lone noisy point that originally broke
# the cluster is mathematically drowned out by the majority, so the two halves re-merge -- while
# two genuinely-distinct touching objects (different dominant masks) keep dissimilar histograms and
# stay apart. The merge runs BEFORE the cluster_propose_points size filter, so re-merged fragments
# can clear the size gate. See models/point_group/point_group_v1m2_custom_criteria.py
# (_merge_proposals_by_mask_histogram). Clean A/B vs clustering/embed-maskclust-allviews.py
# (RAG merge OFF): same multi-view cannot-link split, then the merge heals its over-segmentation.
#
# REQUIRES a checkpoint trained with instance_embedding=True. Both the cannot-link and the RAG
# merge are graceful no-ops unless per-point per-view labels are supplied; tools/infer_insseg.py
# auto-feeds them from <scene-dir>/mask_per_view.npy (N, S) when present.
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

    # --- RAG mask-histogram merge: ON (heals cannot-link over-segmentation) ---
    cluster_mask_rag_merge=True,
    rag_merge_thresh=0.5,            # merge adjacent same-class clusters when hist intersection >= 0.5
    rag_adjacency_radius=2.5,        # RAG adjacency radius (voxel units); = mask_split_radius -> 5 mm
    rag_sim_metric="intersection",   # histogram-similarity metric: "intersection" (default) or "cosine"
)
