# Muon-optimizer leg of the real-scene finetune insseg-litept-small-v1m2-2of3-query-realft.py.
# Same data mix (real-ssl-v1 + synthetic replay), partial-label losses, and 10-epoch
# schedule as -query-realft; ONLY optimizer/scheduler/init differ:
#   * MuonAdamW (utils/muon.py): 2D weight matrices train with Muon, biases/norms/
#     embeddings/spconv kernels keep AdamW. The shape-based split supersedes realft's
#     keyword-based backbone-block 10x LR reduction (flat 2-group LRs instead).
#   * Init from the embed (PG-v1m2) baseline checkpoint — CheckpointLoader is
#     strict=False, so the shared LitePT backbone loads and the MQ query decoder/heads
#     start randomly initialized.
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-realft-muon \
#     -n insseg-litept-small-v1m2-2of3-query-realft-muon -g 1
_base_ = ["./insseg-litept-small-v1m2-2of3-query-realft.py"]

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-realft-muon"

# Dense val curves (per user): eval + checkpoint every 100 steps instead of 1000.
# One eval (500-scene subset + scoring + save) is ~55s vs ~320s per 100 train steps,
# i.e. ~17% wall-clock overhead.
eval_step_interval = 100

# Init from the embed run's model_best.pth (epoch 21, AP50 0.822), transferring as
# much as possible (per user): backbone.* plus seg_head (same Linear(72->7) shape in
# PG-v1m2 and MQ-v1m1). PG's bias_head/embedding_head have no MQ counterpart and are
# skipped as unexpected keys by the non-strict CheckpointLoader; MQ's query decoder,
# input_proj, class/mask heads start randomly initialized (missing keys).
weight = (
    "exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed/model/model_best.pth"
)

# Finetune LRs = pretrain-muon values / 10 (mirrors realft's 10x-below-pretrain rule).
optimizer = dict(
    type="MuonAdamW",
    lr=0.00006,
    muon_lr=0.002,
    momentum=0.95,
    weight_decay=0.05,
)
# MuonAdamW's shape-based split supersedes realft's keyword-based block LR split.
param_dicts = None

# max_lr maps 1:1 onto MuonAdamW's two param groups: [muon (matrices), adamw (rest)].
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.00006],
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
