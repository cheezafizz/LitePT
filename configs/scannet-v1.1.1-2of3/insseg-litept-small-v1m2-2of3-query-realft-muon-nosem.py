# Ablation of insseg-litept-small-v1m2-2of3-query-realft-muon.py that DISCARDS the
# auxiliary per-point semantic head (use_semantic_head=False in MQ-v1m1): no seg_head
# params, no CE+Lovasz seg_loss, and no unknown_bg not-object partial loss — training
# is driven purely by the Hungarian query losses (loss_ce / loss_mask / loss_dice).
# Everything else (data mix, MuonAdamW, schedule, embed-checkpoint init) is identical;
# the checkpoint's seg_head.* weights simply become unexpected keys under the
# non-strict CheckpointLoader.
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-realft-muon-nosem \
#     -n insseg-litept-small-v1m2-2of3-query-realft-muon-nosem -g 1
_base_ = ["./insseg-litept-small-v1m2-2of3-query-realft-muon.py"]

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-realft-muon-nosem"

model = dict(
    use_semantic_head=False,
    # unused with the head off; emptied so no CE/Lovasz criteria are even built
    criteria=[],
)
