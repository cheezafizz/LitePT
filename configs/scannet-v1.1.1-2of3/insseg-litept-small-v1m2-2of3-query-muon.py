# Muon-optimizer race leg of insseg-litept-small-v1m2-2of3-query.py.
# ONLY the optimizer and scheduler.max_lr differ from -query: 2D weight matrices
# (backbone + MQ decoder attention/MLP/heads) are trained with Muon (orthogonalized
# momentum, utils/muon.py); biases/norms/embeddings/spconv kernels keep AdamW.
# -query stays on pure AdamW as the clean A/B leg vs the embed baseline; this config
# tests whether Muon's ~1.3x sample efficiency (LLM-scale evidence, unproven on sparse
# 3D transformers) shows up here. "Wins" = reaching -query's best AP50 in materially
# fewer global steps (compare val AP50-vs-global_step in wandb).
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-muon \
#     -n insseg-litept-small-v1m2-2of3-query-muon -g 1
_base_ = ["./insseg-litept-small-v1m2-2of3-query.py"]

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-muon"

# muon_lr=0.02 is the standard Muon starting point (internally rescaled per matrix by
# sqrt(max(1, rows/cols))); lr=6e-4 keeps the tuned AdamW LR for the non-matrix group
# (matches the base config's `block` group LR, which covered the same attention/MLP
# weights that now go to Muon).
optimizer = dict(
    type="MuonAdamW",
    lr=0.0006,
    muon_lr=0.02,
    momentum=0.95,
    weight_decay=0.05,
)
# The shape-based Muon/AdamW split supersedes the keyword-based `block` LR split.
param_dicts = None

# max_lr must map 1:1 onto MuonAdamW's two param groups: [muon (matrices), adamw (rest)].
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.02, 0.0006],
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
