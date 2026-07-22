_base_ = ["../_base_/default_runtime.py"]

# ============================================================================
# Phase-1 isolated remediation run #3: RECEPTIVE FIELD (architecture).
#
# Builds on `-2mm-fixclust` (2 mm input + rescaled clustering + pct_start=0.05) and
# additionally restores the network's *physical* receptive field at 2 mm. The
# backbone serializes / attends / applies RoPE in voxel-index units
# (models/utils/structure.py, models/litept/litept.py:59), so a 2.5x finer grid
# shrinks every metric span and inflates RoPE phase. Two minimal levers on the
# attention stages (enc3/enc4/dec3 — where long-range centroid context is formed):
#   * enc/dec patch_size 1024 -> 2048: each attention window covers ~2x more points
#     (~2x more physical area), restoring metric coverage for the bias/offset head.
#   * enc/dec rope_freq 100 -> 40 (x0.4 = 0.002/0.005): keeps the physical positional
#     wavelength the model saw at 5 mm and avoids RoPE phase aliasing over the now
#     100-voxel object spans.
# Isolates: "does the 2 mm network need a bigger receptive field?"
#
# >>> VRAM: patch_size=2048 roughly doubles attention memory/compute vs 1024. The
# host has a documented val-loader OOM history -- watch GPU memory on the first
# eval; if it OOMs, drop to enc/dec patch_size=1536 or reduce batch_size to 8
# before reaching instead for the (more invasive) extra-stride variant.
# ============================================================================

batch_size = 12
gradient_accumulation_steps = 1
num_worker = 12
mix_prob = 0.8
empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
evaluate = True
enable_wandb = True
wandb_log_interval = 50
wandb_project = "LitePT"
wandb_key = None

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-2mm-rf"

class_names = [
    "floor",
    "machine",
    "board",
    "tray",
    "paper",
    "table",
    "object",
]
num_classes = 7
segment_ignore_index = (-1, 0, 1)

model = dict(
    type="PG-v1m2",
    backbone=dict(
        type="LitePT",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(2048, 2048, 2048, 2048, 2048),  # was 1024 -- 2x attention coverage
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(40.0, 40.0, 40.0, 40.0, 40.0),   # was 100 -- x0.4 keeps physical wavelength at 2 mm
        dec_depths=(2, 2, 2, 2),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(2048, 2048, 2048, 2048),
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
        dec_rope_freq=(40.0, 40.0, 40.0, 40.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    backbone_out_channels=72,
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    # Same 2 mm-rescaled clustering as -2mm-fixclust (3 cm physical radius).
    voxel_size=0.002,
    cluster_thresh=15.0,
    cluster_closed_points=3000,
    cluster_propose_points=1200,
    cluster_min_points=300,
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=-1,
            weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
        ),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=-1,
        ),
    ],
)

epoch = 1600
eval_epoch = 1600
eval_step_interval = 1000
val_subset_size = 500
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

dataset_type = "ScanNetDataset"
data_root = "data/scannet-v1.1.1-2of3"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.1),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": 0.002}),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_centroid",
                    "bbox",
                    "grid_size"
                ),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.002,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "name",
                ),
                feat_keys=("color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict()
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
    ),
    dict(type="CheckpointSaver", save_freq=None),
]
