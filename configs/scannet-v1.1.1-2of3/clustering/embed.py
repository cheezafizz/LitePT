# Inference-time CLUSTERING config for tools/infer_insseg.py (--cluster-config).
#
# Read SEPARATELY from the training config: the training config builds the backbone/heads
# and selects the checkpoint; this file decides how the per-point predictions are grouped
# into instances.
#
# Variant: GEOMETRIC + EMBEDDING SPLIT.
#   * Embedding head IS used.                       -> instance_embedding = True
#   * Distinct-mask / same-view points MAY connect  -> cluster_mask_constraint = False
# Mirrors insseg-litept-small-v1m2-2of3-embed.py. After the geometric ball-query+BFS pass,
# each proposal is sub-divided by mean-shift in embedding space, recovering touching
# same-class instances that offset+BFS merged.
#
# REQUIRES a checkpoint trained with instance_embedding=True (the embedding head must
# exist). tools/infer_insseg.py validates this and errors clearly otherwise.
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
    embed_bandwidth=1.5,          # mean-shift bandwidth (embedding units) for the split
    embed_min_points=100,         # drop embedding sub-clusters smaller than this

    # --- 2D-mask cannot-link: OFF ---
    cluster_mask_constraint=False,
)
