# Inference-time CLUSTERING config for tools/infer_insseg.py (--cluster-config).
#
# Read SEPARATELY from the training config: the training config builds the backbone/heads
# and selects the checkpoint; this file decides how the per-point predictions are grouped
# into instances.
#
# Variant: GEOMETRIC + EMBEDDING SPLIT + 2D-MASK CANNOT-LINK (single-view).
#   * Embedding head IS used.                          -> instance_embedding = True
#   * Distinct-mask / same-view points must NOT connect -> cluster_mask_constraint = True
#   * Cannot-link uses each point's ORIGIN view only    -> mask_constraint_multiview = False
# Mirrors insseg-litept-small-v1m2-2of3-embed-maskclust.py. After the geometric and
# embedding splits, each proposal is cut so that two points coming from the SAME camera
# view but lying in DIFFERENT D-FINE-seg 2D instance masks never share an instance. Points
# from different views still merge by adjacency, so an object spanning views is not shattered.
#
# REQUIRES a checkpoint trained with instance_embedding=True. The cannot-link is a graceful
# no-op unless per-point mask labels are supplied; tools/infer_insseg.py auto-feeds them from
# <scene-dir>/mask_instance.npy (+ mask_view.npy) when present.
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

    # --- 2D-mask cannot-link: ON (single-view) ---
    cluster_mask_constraint=True,
    mask_constraint_multiview=False,  # origin-view cannot-link only (needs mask_instance/mask_view)
    mask_split_radius=2.5,            # adjacency radius (voxel units) for the constrained split; = cluster_thresh -> 5 mm
)
