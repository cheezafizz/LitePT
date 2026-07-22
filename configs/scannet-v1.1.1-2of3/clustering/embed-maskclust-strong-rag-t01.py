# AUTO-GENERATED rag_merge_thresh sweep variant of clustering/embed-maskclust-strong-rag.py.
# Identical to that config EXCEPT rag_merge_thresh=0.1 (lower thresh -> more aggressive merge:
# fewer, larger instances). See embed-maskclust-strong-rag.py for the full method description.
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

    # --- RAG mask-histogram merge: ON (heals the strict cannot-link's over-segmentation) ---
    cluster_mask_rag_merge=True,
    rag_merge_thresh=0.1,            # merge adjacent same-class clusters when hist intersection >= 0.1
    rag_adjacency_radius=2.5,        # RAG adjacency radius (voxel units); = mask_split_radius -> 5 mm
    rag_sim_metric="intersection",   # histogram-similarity metric: "intersection" (default) or "cosine"
)
