_base_ = ["../_base_/default_runtime.py"]

# Instance-embedding variant of insseg-litept-small-v1m2-2of3-normaug.py.
# Identical to normaug EXCEPT it enables the optional discriminative
# instance-embedding head on the PG-v1m2 model (instance_embedding=True). The head
# learns a per-point embedding with a pull/push loss; at inference each geometric
# (offset+BFS) proposal is sub-divided in embedding space, separating touching
# same-class objects that ball-query+BFS would otherwise merge into one instance.
# Everything else (backbone, dataset, augs, schedule, clustering radius) matches
# normaug so this is a clean A/B against insseg-litept-small-v1m2-2of3-normaug.py.

# misc custom setting
batch_size = 6 # bs: total bs in all gpus
gradient_accumulation_steps = 2
num_worker = 12
mix_prob = 0.8
empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
evaluate = True
enable_wandb = True
wandb_log_interval = 50  # throttle per-step wandb logging on this ~847k-step run
wandb_project = "LitePT"  # wandb project name
wandb_key = None # wandb token, default is None. If None, login with `wandb login` in your terminal


save_path = "exp/scannet/insseg-litept-small-v1m2-2of3-embed"

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
    voxel_size=0.002,
    cluster_thresh=2.5,           # 2.5 * 0.002 m = 5 mm physical radius
    cluster_closed_points=3000,
    cluster_propose_points=300,
    cluster_min_points=100,
    # --- instance-embedding head (the only functional difference vs normaug) ---
    instance_embedding=True,
    embedding_dim=5,              # per-point embedding dimensionality
    embed_delta_v=0.5,           # pull margin: points kept within delta_v of their mean
    embed_delta_d=1.5,           # push margin: instance means kept >= 2*delta_d apart
    embed_loss_weight=1.0,       # weight of the discriminative loss in the total loss
    embed_reg_weight=0.001,      # small regularizer pulling means toward the origin
    embed_bandwidth=1.5,         # mean-shift bandwidth (embedding units) for the split
    embed_min_points=100,        # drop embedding sub-clusters smaller than this (matches cluster_min_points)
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=-1,
            # class order: floor, machine, board, tray, paper, table, object
            # `object` (idx 6) is the primary target — weighted 10x
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
epoch = 200
eval_epoch = 200
# train=50,803 scenes, bs=12 -> ~4,234 steps/epoch -> ~846,800 total steps.
# eval every 5,000 steps -> ~169 evals (val split has 6,350 scenes, ~35x the s4
# val set, so evaluating more often would dominate wall-clock).
eval_step_interval = 1000
val_subset_size = 500  # periodic eval uses only the first 500 val scenes (full val has 6,350)
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
            # Inference-calibrated non-rigid aug (vs rigidaug, which omits both).
            # Dataset: 2 mm voxel grid, ~1 m crop, objects 14-40 cm with median
            # 6-18 cm centroid spacing but as little as ~8 mm in packed scenes.
            # Inference applies NEITHER transform and voxelizes at 2 mm, so the aug
            # must (a) survive the 2 mm grid and (b) keep worst-case displacement
            # under the ~8 mm packed gap so instance boundaries never merge.
            #   * RandomJitter sigma=2 mm (=1 voxel) models real depth-sensor noise;
            #     clip=6 mm caps the tail below the packed inter-object gap.
            #   * ElasticDistortion keeps the 2of3 granularity (10/40 cm ~ object
            #     size -> genuine non-rigid deform) but amplitude cut to 4/8 mm, well
            #     under the median 6-18 cm gap (vs the merging 5-20 cm 5mm-run values).
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
            # --- Normal-domain randomization (sim->real for VGGT inference) ---
            # Training normals are clean PCA normals on dense synthetic geometry
            # (local-consistency median ~0 deg). VGGT-reconstructed clouds, run
            # through the SAME compute_normals, are far noisier (median ~16 deg,
            # p95 ~58 deg, heavy gross-error tail) — see tools/analyze_normal_gap.py.
            # NormalJitter randomizes per-scene noise over sigma=[0.05,0.30] (median
            # ~5 deg .. ~29 deg, bracketing VGGT's 16 deg) and corrupt_ratio=0.05
            # injects gross errors to match VGGT's tail; p=0.9 leaves 10% of scenes
            # clean to preserve clean-data accuracy. NormalDropout zeros the whole
            # normal channel on 10% of scenes so the model still works when
            # reconstructed normals are unusable. MUST run before ToTensor (numpy).
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
