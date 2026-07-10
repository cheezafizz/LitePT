# 2-GPU variant of insseg-litept-small-v1m2-2of3-query.py. LAUNCH WITH -g 2.
# batch_size (effective) stays 12; on 2 GPUs the per-GPU batch is 12/2 = 6.
# steps=3 -> 6/3 = 2 scenes/micro-batch (~14.6 GB, the validated safe point).
# Effective batch (12), total optimizer steps, and the LR schedule are unchanged.
#   bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-query-2gpu \
#     -n insseg-litept-small-v1m2-2of3-query-2gpu -g 2
_base_ = ["./insseg-litept-small-v1m2-2of3-query.py"]

gradient_accumulation_steps = 3
save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-2gpu"
