_base_ = ["../_base_/default_runtime.py"]

# Query-based (Mask3D/SPFormer-style) variant of insseg-litept-small-v1m2-2of3-embed.py.
# Instead of recovering instances by offset regression + ball-query/BFS clustering
# (the PG-v1m2 path), a set of learnable queries attend to the backbone's per-point
# features through a masked transformer decoder (MQ-v1m1); each query directly emits
# one instance (class + soft mask). There are NO cluster_*/voxel_size/embedding knobs
# -- instance formation is learned, not a geometric heuristic. An auxiliary per-point
# semantic head (CE+Lovasz, identical to -embed) is kept to stabilize training.
# Backbone, dataset, augmentations, schedule, and hooks match -embed so this is a
# clean A/B against the clustering path.

# misc custom setting
batch_size = 12 # bs: total (effective) bs in all gpus -- matches the base -2of3 config
# context_grid_factor=1 (full 2mm resolution) makes each scene's ~150-230k points
# attention tokens; processing many scenes in one backward OOMs a 32GB GPU. Gradient
# accumulation splits the EFFECTIVE batch (batch_size) into `steps` scene-contiguous
# micro-batches, backward each, and takes ONE optimizer step -> peak memory = one
# micro-batch, effective batch preserved. Here 12 / 6 = 2 scenes/micro-batch -> ~14.6GB
# peak (measured). NOTE this trainer uses "divide" semantics: effective batch == batch_size
# (NOT batch_size * steps). (This knob was previously a no-op; engines/train.py now honors it.)
gradient_accumulation_steps = 6
num_worker = 12
mix_prob = 0.8
empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
evaluate = True
enable_wandb = True
wandb_log_interval = 50  # throttle per-step wandb logging on this ~169k-step run
wandb_project = "LitePT"  # wandb project name
wandb_key = None # wandb token, default is None. If None, login with `wandb login` in your terminal


save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-query"

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
    # --- query decoder ---
    # 50 queries (was 100): decoder cost (cross-attn K x Nc, mask logits, Hungarian
    # matching) is ~linear in K and dominates (~75%) of step time at factor=1, so this
    # is ~1.6x faster/step. K must cover the MIXED-sample instance count: mix_prob=0.8
    # merges scene pairs at collate, so GT instances/sample go up to ~46 (single-scene
    # max 23, p95 18, mean 10.9 over 500 train scenes). 25 was tried and rejected --
    # ~1/3 of mixed samples would exceed it, leaving GTs Hungarian-unmatched.
    num_queries=50,
    dec_dim=256,
    dec_layers=6,
    dec_num_head=8,
    dec_ffn_dim=1024,
    context_grid_factor=1,   # 1 * 0.002 m grid = 2 mm tokens = FULL point resolution
    #   Fineness A/B (overfit, per-GT mean mask-IoU): factor=5(10mm) 0.30 -> 3(6mm) 0.80;
    #   factor=1 gives the sharpest masks but needs the gradient accumulation above to fit
    #   in memory (and is ~3x slower/step). Set to 3 with gradient_accumulation_steps=1 for
    #   a ~3x faster run if 6mm masks suffice.
    mask_threshold=0.5,
    # --- set-prediction (Hungarian) matching + loss weights ---
    cost_class=2.0,
    cost_mask=5.0,
    cost_dice=5.0,
    loss_class_weight=2.0,
    loss_mask_weight=5.0,
    loss_dice_weight=5.0,
    eos_coef=0.1,
    # class order: floor, machine, board, tray, paper, table, object
    # `object` (idx 6) is the primary target -- weighted 10x in the query classifier too
    class_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0],
    # --- retained auxiliary per-point semantic head ---
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
# 40 epochs (was 200): the embed baseline hit its never-beaten best AP50 at epoch 21
# of 200, so 40 epochs (~2x that) is the realistic convergence budget. OneCycleLR's
# total_steps spans the configured epochs, so shrinking `epoch` (rather than stopping
# a 200-epoch schedule early) lets the LR actually anneal within the run.
epoch = 40
eval_epoch = 40
# train=50,803 scenes, bs=12 -> ~4,233 steps/epoch -> ~169,300 total steps.
# eval every 1,000 steps -> ~169 evals (val split has 6,350 scenes, ~35x the s4
# val set, so evaluating more often would dominate wall-clock).
eval_step_interval = 1000
val_subset_size = 500  # periodic eval uses 500 evenly-spaced val scenes (full val has 6,350)
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.01,
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
