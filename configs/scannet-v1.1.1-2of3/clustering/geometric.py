# Inference-time CLUSTERING config for tools/infer_insseg.py (--cluster-config).
#
# This file carries ONLY the post-processing knobs that turn the network's per-point
# predictions (semantic logits + centroid offsets [+ optional embedding]) into instance
# proposals. It is read SEPARATELY from the training config: the training config builds
# the backbone/heads and selects the checkpoint; this file decides how those predictions
# are grouped. See configs/scannet-v1.1.1-2of3/clustering/README intent in the plan.
#
# Variant: GEOMETRIC ONLY (the universally-safe baseline).
#   * Embedding head is NOT used.                  -> instance_embedding = False
#   * Distinct-mask / same-view points MAY connect -> cluster_mask_constraint = False
# Equivalent to the base config insseg-litept-small-v1m2-2of3.py. Works with ANY
# checkpoint (an embedding head in the checkpoint, if present, is simply left unused).
#
# Values are the calibrated 2-of-3 2 mm operating point. The effective ball-query
# grouping radius is cluster_thresh * voxel_size = 2.5 * 0.002 m = 5 mm.

cluster = dict(
    # --- geometric ball-query + BFS connected components (always active) ---
    voxel_size=0.002,             # 2 mm grid; center_pred is put in voxel units before ball-query
    cluster_thresh=2.5,           # ball-query radius in voxel units; 2.5 * 0.002 m = 5 mm physical
    cluster_closed_points=3000,   # max neighbours returned per point by ball-query
    cluster_propose_points=300,   # drop final proposals with <= this many points
    cluster_min_points=100,       # BFS components smaller than this are not proposed
    segment_ignore_index=(-1, 0, 1),  # classes excluded from clustering (floor/board are stuff here)

    # --- embedding-head split: OFF ---
    # When False, touching same-class instances are NOT separated in embedding space;
    # the embedding head (if the checkpoint has one) is not consulted.
    instance_embedding=False,

    # --- 2D-mask cannot-link: OFF ---
    # When False, two points from the SAME camera view lying in DIFFERENT 2D instance
    # masks are allowed to merge by spatial adjacency (no cannot-link constraint).
    cluster_mask_constraint=False,
)
