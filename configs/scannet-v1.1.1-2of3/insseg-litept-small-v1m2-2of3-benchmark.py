# Inference-cost benchmark for `insseg-litept-small-v1m2-2of3`.
#
# Inherits the target config unchanged (same model + same val transform pipeline), then
# swaps the hook list for [CheckpointLoader, RuntimeProfiler_inference]. The profiler runs
# in `before_train`, measures per-scene inference cost over the val loader (batch size 1 =>
# one scene per iteration), prints a report, and exits -- no training happens.
#
# Run:
#   bash scripts/train.sh \
#     -p /home/fai/miniconda3/envs/litept/bin/python \
#     -d scannet-v1.1.1-2of3 \
#     -c insseg-litept-small-v1m2-2of3-benchmark \
#     -n insseg-litept-small-v1m2-2of3-benchmark \
#     -g 1

_base_ = ["./insseg-litept-small-v1m2-2of3.py"]

enable_wandb = False

# This config's own model/ dir is empty; use the latest trained checkpoint with the
# identical architecture on the same 2of3 dataset (only augmentation differs).
weight = "exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-rigidaug/model/model_best.pth"

# Profile the first 500 of 6,350 val scenes (override to None for the full split).
val_subset_size = 500

# Run val transforms in-process so CPU preprocessing time and host RAM (RSS) are
# attributable to this process (worker RSS would otherwise be invisible, and prefetch
# would hide load time).
num_worker_val = 0

save_path = "exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-benchmark"

hooks = [
    # Must precede the profiler so `weight` is loaded before profiling.
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(
        type="RuntimeProfiler_inference",
        warm_up=5,
        interrupt=True,
        precisions=("fp32", "bf16"),
        measure_end_to_end=True,
        csv_path="logs/insseg-2of3-inference-benchmark.csv",
    ),
]
