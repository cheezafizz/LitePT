_base_ = ["../_base_/default_runtime.py"]

# ============================================================================
# Phase-1 isolated remediation run #1: CLUSTERING + WARMUP only.
#
# Baseline for comparison: `insseg-litept-small-v1m2-2of3-rigidaug` collapsed to
# ~0.007 mAP at 2 mm. This config keeps the 2 mm network input and rigid-only aug
# but changes exactly two things vs rigidaug:
#   1. Rescales the PointGroup clustering post-processing for the ~6x denser 2 mm
#      cloud (see model.* below). The physical ball radius is held at 3 cm
#      (cluster_thresh * voxel_size = 15.0 * 0.002 = 0.03 m), matching the
#      hard-coded-0.02 default that both 5 mm and rigidaug effectively used.
#   2. Restores pct_start=0.05 (rigidaug used 0.01, which ramped LR to peak ~5x
#      sooner; the healthy 5 mm run used 0.05).
# Isolates: "how much of the collapse is post-processing + warmup?"
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

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-2mm-fixclust"

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

# model settings
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
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(True, True, True, False),
        dec_attn=(False, False, False, True),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
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
    # --- CLUSTERING RESCALED FOR 2 mm DENSITY -------------------------------
    # center_pred = (coord + bias) / voxel_size, then ball radius = cluster_thresh.
    # Make voxel_size explicit (= the 2 mm grid) and set cluster_thresh=15.0 so the
    # physical radius stays 3 cm. Count thresholds scaled ~6x (5mm->2mm surface
    # density) so spurious sub-object fragments no longer survive `propose_points`.
    voxel_size=0.002,
    cluster_thresh=15.0,        # 15.0 * 0.002 m = 0.03 m physical radius (was 1.5 * 0.02)
    cluster_closed_points=3000,  # was 600
    cluster_propose_points=1200,  # was 200
    cluster_min_points=300,      # was 50
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

# scheduler settings
epoch = 1600
eval_epoch = 1600
eval_step_interval = 1000
val_subset_size = 500
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,  # restored from rigidaug's 0.01 (matches the healthy 5 mm run)
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
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
            # Rigid-only (same as rigidaug): jitter/elastic intentionally omitted here
            # so this run isolates the clustering+warmup change. The `-2mm-aug` config
            # adds them back to test their contribution separately.
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
