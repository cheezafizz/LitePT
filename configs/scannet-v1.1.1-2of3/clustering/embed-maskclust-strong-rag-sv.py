# Inference-time CLUSTERING config for tools/infer_insseg.py (--cluster-config).
#
# Read SEPARATELY from the training config: the training config builds the backbone/heads
# and selects the checkpoint; this file decides how the per-point predictions are grouped
# into instances.
#
# Variant: GEOMETRIC + EMBEDDING SPLIT + 2D-MASK CANNOT-LINK (multi-view, STRONG) + RAG SHARED-VIEW MERGE.
#   * Embedding head IS used.                           -> instance_embedding = True
#   * Distinct-mask / same-view points must NOT connect -> cluster_mask_constraint = True
#   * Cannot-link fires in ANY shared view              -> mask_constraint_multiview = True
#   * Cannot-link enforced TRANSITIVELY (no detour leak)-> mask_constraint_strict = True
#   * Over-segmentation is HEALED after the split       -> cluster_mask_rag_merge = True
#
# Identical to clustering/embed-maskclust-strong-rag.py EXCEPT the RAG merge decision uses
# rag_sim_metric="shared_view" instead of the legacy global histogram intersection. The new measure is
# the SOFT cross-pair conflict ratio: over the views the two clusters SHARE, the fraction of
# co-visible point-pairs that land in DIFFERENT 2D masks (1 - (cnt_a . cnt_b)/(vc_a . vc_b)).
# Two adjacent same-class clusters are merged when that ratio is <= rag_merge_thresh. Unlike the
# legacy intersection/cosine metrics (where HIGHER thresh = stricter), here LOWER thresh = stricter
# (fewer, larger merges). The shared-view restriction fixes the intersection metric's flaw whereby
# two genuine fragments visible in DIFFERENT view sets look dissimilar (disjoint global histograms)
# even when they agree perfectly in the views they actually share. Adjacent same-class clusters that
# share NO view carry no mask evidence and are left unmerged. See
# models/point_group/point_group_v1m2_custom_criteria.py (_merge_proposals_by_mask_histogram).
#
# REQUIRES a checkpoint trained with instance_embedding=True. Both the cannot-link and the RAG merge
# are graceful no-ops unless per-point per-view labels are supplied; tools/infer_insseg.py auto-feeds
# them from <scene-dir>/mask_per_view.npy (N, S) when present.
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

    # --- RAG shared-view merge: ON (heals the strict cannot-link's over-segmentation) ---
    cluster_mask_rag_merge=True,
    rag_merge_thresh=0.3,            # MAX different-mask ratio that still merges (lower = stricter = fewer merges)
    rag_adjacency_radius=2.5,        # RAG adjacency radius (voxel units); = mask_split_radius -> 5 mm
    rag_sim_metric="shared_view",
)
