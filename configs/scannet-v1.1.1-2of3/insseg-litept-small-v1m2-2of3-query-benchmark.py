# Inference-cost benchmark for the query-based (MQ-v1m1) insseg model.
#
# Inherits the -query config (architecture identical to -query-muon, which differs
# only in optimizer), swaps hooks for [CheckpointLoader, RuntimeProfiler_inference].
#
# Run:
#   bash scripts/train.sh \
#     -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 \
#     -c insseg-litept-small-v1m2-2of3-query-benchmark \
#     -n insseg-litept-small-v1m2-2of3-query-benchmark \
#     -g 1

_base_ = ["./insseg-litept-small-v1m2-2of3-query.py"]

enable_wandb = False

weight = "exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-query/model/model_best.pth"

# The trained -query checkpoint predates the K=50 default and used 100 queries.
model = dict(num_queries=100)

# Profile the first 500 of 6,350 val scenes (override to None for the full split).
val_subset_size = 500

# In-process val transforms so preprocess time / host RSS are attributable.
num_worker_val = 0

save_path = "exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-query-benchmark"

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(
        type="RuntimeProfiler_inference",
        warm_up=5,
        interrupt=True,
        precisions=("fp32",),
        measure_end_to_end=True,
        csv_path="logs/insseg-2of3-query-inference-benchmark.csv",
    ),
]
