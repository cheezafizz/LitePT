# Anti-merge ablation B of insseg-litept-small-v1m2-2of3-query-realft-muon-nosem.py:
# Method 2 ONLY -- boundary / hard-token weighting of the mask BCE. A per-token weight
# is precomputed once per scene in _build_scene_gt: tokens within `boundary_radius`
# coarse cells of a token owned by a DIFFERENT GT instance (foreground<->foreground
# boundary) get `boundary_weight`x the BCE, so the loss concentrates on exactly the
# tokens where adjacent objects could merge. No query-pair interaction. loss_dice is
# left standard. Everything else is identical to the parent.
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-realft-muon-nosem-bweight \
#     -n insseg-litept-small-v1m2-2of3-query-realft-muon-nosem-bweight -g 1
_base_ = ["./insseg-litept-small-v1m2-2of3-query-realft-muon-nosem.py"]

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-realft-muon-nosem-bweight"

# Method 2: boundary-weighted mask BCE. boundary_radius is in coarse cells
# (context_grid_factor=1 -> 2 mm each), so radius=1 is a ~2 mm boundary band.
model = dict(
    use_boundary_weight=True,
    boundary_weight=5.0,
    boundary_radius=1,
)
