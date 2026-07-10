# 4-GPU variant of insseg-litept-small-v1m2-2of3-query.py. LAUNCH WITH -g 4.
# batch_size (effective) stays 12; on 4 GPUs the per-GPU batch is 12/4 = 3.
# steps=2 -> micro-batches of [2, 1] scenes, max 2/micro-batch (~14.6 GB safe point).
# (steps=1 = 3 scenes/micro-batch ~22 GB likely fits but is untested vs large scenes.)
# Effective batch (12), total optimizer steps, and the LR schedule are unchanged.
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-4gpu \
#     -n insseg-litept-small-v1m2-2of3-query-4gpu -g 4
_base_ = ["./insseg-litept-small-v1m2-2of3-query.py"]

gradient_accumulation_steps = 2
save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-4gpu"
