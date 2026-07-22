# Anti-merge ablation C of insseg-litept-small-v1m2-2of3-query-realft-muon-nosem.py:
# BOTH Method 1 (pairwise query-mask repulsion, loss_overlap) AND Method 2 (boundary /
# hard-token weighting of the mask BCE) together, to see whether the query-pair penalty
# and the boundary emphasis are complementary. Hyperparameters mirror the isolated
# -overlap and -bweight configs. Everything else is identical to the parent.
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-realft-muon-nosem-overlap-bweight \
#     -n insseg-litept-small-v1m2-2of3-query-realft-muon-nosem-overlap-bweight -g 1
_base_ = ["./insseg-litept-small-v1m2-2of3-query-realft-muon-nosem.py"]

save_path = (
    "exp/scannet/insseg-litept-small-v1m2-2of3-query-realft-muon-nosem-overlap-bweight"
)

model = dict(
    loss_overlap_weight=1.0,
    use_boundary_weight=True,
    boundary_weight=5.0,
    boundary_radius=1,
)
