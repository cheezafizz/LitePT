# Anti-merge ablation A of insseg-litept-small-v1m2-2of3-query-realft-muon-nosem.py:
# Method 1 ONLY -- an explicit pairwise overlap/repulsion loss between matched-query
# masks. The plain token-averaged mask BCE barely penalizes a thin "bridge" of tokens
# shared by two adjacent objects (near-GT merge); loss_overlap adds a direct
# sum_{i!=j} sigma(m_i)*sigma(m_j) per-token penalty on the matched masks, so two
# different queries firing on the same token is costly. Everything else (nosem query
# losses, MuonAdamW, schedule, embed-checkpoint init) is identical to the parent.
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-realft-muon-nosem-overlap \
#     -n insseg-litept-small-v1m2-2of3-query-realft-muon-nosem-overlap -g 1
_base_ = ["./insseg-litept-small-v1m2-2of3-query-realft-muon-nosem.py"]

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-realft-muon-nosem-overlap"

# Method 1: pairwise query-mask repulsion. Sits alongside loss_mask_weight=5.0 /
# loss_dice_weight=5.0; start at 1.0 and tune (~0.5-2.0).
model = dict(loss_overlap_weight=1.0)
