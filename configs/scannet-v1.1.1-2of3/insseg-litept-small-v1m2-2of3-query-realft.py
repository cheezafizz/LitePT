_base_ = ["../_base_/default_runtime.py"]

# Real-scene FINETUNE of insseg-litept-small-v1m2-2of3-query.py.
#
# Starts from the synthetic v1.1.1-2of3 -query checkpoint (`weight` below) and adapts
# to real scenes labeled by VisionSCO's save_instance_labels.py pipeline (converted by
# tools/convert_ssl_scenes.py into data/real-ssl-v1). Real labels are PARTIAL:
#   segment 6 ("object")   — detected object points (seg_id >= 0 or -2)
#   segment 7 (unknown_bg) — everything else; class unknown but definitely NOT object
#   instance >= 0 only for matcher-grouped objects; seg_id=-2 object points keep
#   instance=-1 and are token-ignored in the Hungarian mask/dice loss.
# The semantic head is guided by (a) full CE+Lovasz on object points, (b) the
# not-object partial loss on unknown_bg points, and (c) SYNTHETIC REPLAY: training
# mixes the full synthetic set (ConcatDataset, ~80k real : ~50.8k syn ≈ 1.6:1) so the
# six non-object classes and full-supervision instance behavior don't collapse.
# Full finetune at 10x-reduced LR; backbone blocks a further 10x lower (param_dicts).

# misc custom setting
batch_size = 12
gradient_accumulation_steps = 6
num_worker = 12
mix_prob = 0.8
empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
evaluate = True
enable_wandb = True
wandb_log_interval = 10  # ~3s/step -> a train_batch/* point every ~30s
wandb_project = "LitePT"
wandb_key = None

save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query-realft"

# Init from the embed run's model_best.pth (epoch 21, AP50 0.822), same as the
# -muon leg: backbone.* plus seg_head transfer (non-strict CheckpointLoader); MQ's
# query decoder, input_proj, class/mask heads start randomly initialized.
weight = (
    "exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed/model/model_best.pth"
)
resume = False

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
unknown_bg_index = 7  # real-scene sentinel: known not-object, class unknown

# model settings — identical to -query plus the real-scene partial-label knobs
model = dict(
    type="MQ-v1m1",
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
    num_queries=50,
    dec_dim=256,
    dec_layers=6,
    dec_num_head=8,
    dec_ffn_dim=1024,
    context_grid_factor=1,
    mask_threshold=0.5,
    cost_class=2.0,
    cost_mask=5.0,
    cost_dice=5.0,
    loss_class_weight=2.0,
    loss_mask_weight=5.0,
    loss_dice_weight=5.0,
    eos_coef=0.1,
    class_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
    # --- real-scene partial labels ---
    unknown_bg_index=unknown_bg_index,
    object_class_index=6,
    not_object_loss_weight=1.0,
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

# scheduler settings — finetune: 10 epochs over the ~131k-scene mixed set
# (~10.9k steps/epoch at bs=12), LR 10x below pretrain, backbone blocks 100x below.
epoch = 10
eval_epoch = 10
# Dense val curves (matches -muon): eval + checkpoint every 100 steps.
eval_step_interval = 100
val_subset_size = 500
optimizer = dict(type="AdamW", lr=0.0006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0006, 0.00006],
    pct_start=0.01,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.00006)]

# dataset settings
real_data_root = "data/real-ssl-v1"
syn_data_root = "data/scannet-v1.1.1-2of3"

# Real branch: -query train pipeline WITHOUT the sim->real normal randomization
# (NormalJitter/NormalDropout) — real normals are computed on VGGT-style
# reconstructions and are already noisy.
real_train_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.002, clip=0.006),
    dict(type="ElasticDistortion", distortion_params=[[0.1, 0.004], [0.4, 0.008]]),
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
            "grid_size",
        ),
        feat_keys=("color", "normal"),
    ),
]

# Synthetic replay branch: identical to the -query train pipeline (incl. the
# NormalJitter/NormalDropout sim->real randomization).
syn_train_transform = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.002, clip=0.006),
    dict(type="ElasticDistortion", distortion_params=[[0.1, 0.004], [0.4, 0.008]]),
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
    dict(type="NormalJitter", sigma=[0.05, 0.30], clip=0.5, corrupt_ratio=0.05, p=0.9),
    dict(type="NormalDropout", dropout_application_ratio=0.1),
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
            "grid_size",
        ),
        feat_keys=("color", "normal"),
    ),
]

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type="RealSSLDataset",
                split="train",
                data_root=real_data_root,
                transform=real_train_transform,
                test_mode=False,
                loop=1,
            ),
            dict(
                type="ScanNetDataset",
                split="train",
                data_root=syn_data_root,
                transform=syn_train_transform,
                test_mode=False,
                loop=1,
            ),
        ],
    ),
    # validate on REAL scenes (GT instances all class `object`)
    val=dict(
        type="RealSSLDataset",
        split="val",
        data_root=real_data_root,
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
    test=dict(),
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
        # Real-ssl scene names are <chunk>_<machine>_<seq>_<frame>; break val
        # metrics out per machine (val/mAP_001, val/mAP_004, ...) in addition
        # to the overall val/mAP etc. over the whole val set.
        group_pattern=r"^[^_]+_([^_]+)_",
    ),
    dict(type="CheckpointSaver", save_freq=None),
]
