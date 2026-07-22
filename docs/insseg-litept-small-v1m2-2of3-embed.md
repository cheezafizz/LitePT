# InsSeg LitePT-small PG-v1m2 "embed" — 2of3 Industrial Dataset

**Config:** `configs/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed.py`
**Model:** PointGroup-v1m2 + LitePT-small backbone + discriminative instance-embedding head
**Best checkpoint:** `exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed/model/model_best.pth`

## 1. Overview

This is the instance-embedding variant of the PG-v1m2 instance-segmentation model for the 2of3 industrial dataset. It is identical to the `-normaug` config **except** that it enables the optional discriminative instance-embedding head (`instance_embedding=True`), making it a clean A/B against normaug.

**What the embed head adds:** the model learns a 5-dimensional per-point embedding trained with a pull/push (discriminative) loss — points of the same instance are pulled within `delta_v=0.5` of their instance mean, and instance means are pushed at least `2 × delta_d=1.5` apart. At inference, each geometric proposal produced by offset-shifted ball-query + BFS clustering is **sub-divided in embedding space** via mean-shift (bandwidth 1.5). This separates touching same-class objects that pure geometric clustering would merge into a single instance — the dominant failure mode on this densely packed dataset.

## 2. Dataset — `scannet-v1.1.1-2of3`

A synthetic 7-class industrial point-cloud dataset (ScanNet-format on disk, **not** ScanNet content). Scenes are small-scale industrial scans (~2–3 m extent) with objects 14–40 cm in size and centroid spacing typically 6–18 cm, down to ~8 mm in packed scenes.

| Property | Value |
|---|---|
| Classes (7) | floor, machine, board, tray, paper, table, object |
| Instance eval classes | all except ignored `(-1, 0, 1)` = floor, machine (semantic-only) |
| Splits (seed=42) | train 50,803 / val 6,350 / test 6,351 scenes |
| Voxel resolution | **2 mm** (re-voxelized from the former 5 mm set) |
| Crop | AABB x∈[−0.4, 0.6], y∈[−0.5, 0.7], z∈[−0.2, 0.7] m |
| Points per scene | ~327k (~252k voxels after 2 mm GridSample on val) |
| Disk size | ~645 GB |
| Location | `data/scannet-v1.1.1-2of3` (regenerable from raw `dataset/v1.1.1`) |

**Why 2 mm:** the previous 5 mm resolution under-resolved the 6–18 cm inter-object gaps. 1 mm was infeasible (~2.42 TB; raw depth caps density at ~4.3 M pts/scene anyway); 2 mm (~5.8× denser than 5 mm) fits on disk and resolves object gaps across ~30 voxels.

## 3. Preprocessing

### Offline (raw → training format)

`datasets/preprocessing/v1_0_1/preprocess_v1_0_1.py` converts the raw v1.1.1 dataset into per-scene `coord / color / normal / segment / instance` `.npy` files:

```bash
python datasets/preprocessing/v1_0_1/preprocess_v1_0_1.py \
  --dataset_root /home/fai/workspace/jhp/dataset/v1.1.1 \
  --output_root data/scannet-v1.1.1-2of3 \
  --voxel_size 0.002 \
  --crop_min=-0.4,-0.5,-0.2 --crop_max=0.6,0.7,0.7
```

Notes: pass `--crop_min`/`--crop_max` with `=` (leading `-` otherwise trips argparse); the script is resumable (skips scenes whose `instance.npy` already exists).

### Online training transforms (key augmentations)

| Transform | Setting | Rationale |
|---|---|---|
| RandomDropout | ratio 0.2, p 0.5 | density robustness |
| RandomRotate | z full, x/y ±1/64 π, p 0.5 | pose |
| RandomScale / RandomFlip | 0.9–1.1 / p 0.5 | rigid |
| RandomJitter | σ = 2 mm (1 voxel), clip 6 mm | models depth-sensor noise; tail kept under the ~8 mm packed inter-object gap |
| ElasticDistortion | (10 cm, 4 mm), (40 cm, 8 mm) | non-rigid deform at object granularity; amplitude well under the 6–18 cm gaps so instances never merge |
| Chromatic (AutoContrast/Translation/Jitter) | p 0.2 / 0.95 / 0.95 | color |
| GridSample | **2 mm** | matches dataset & inference resolution |
| SphereCrop | sample_rate 0.8, random | VRAM |
| NormalJitter | σ ∈ [0.05, 0.30], corrupt 5%, p 0.9 | sim→real: training normals are clean PCA normals (median ~0°) but VGGT-reconstructed clouds are noisy (median ~16°, p95 ~58°); jitter brackets that range |
| NormalDropout | 10% of scenes | model still works when reconstructed normals are unusable |

Validation uses no augmentation: CenterShift → 2 mm GridSample → NormalizeColor → InstanceParser, with predictions back-projected to original points for eval.

## 4. Model & training setup

### Architecture

| Component | Setting |
|---|---|
| Backbone | LitePT-small, in_channels 6 (color+normal), enc (36→504 ch, depths 2/2/2/6/2), dec (72/72/144/252), conv on early stages + attention on deep stages, serialized orders z / z-trans / hilbert / hilbert-trans, out 72 ch |
| Head | PG-v1m2: semantic head (7 cls) + offset (bias) head + instance-embedding head |
| Clustering | voxel_size 0.002, cluster_thresh 2.5 → **5 mm physical radius**, closed 3000 / propose 300 / min 100 pts |
| Embedding head | dim 5, δ_v 0.5, δ_d 1.5, loss weight 1.0, reg 0.001, mean-shift bandwidth 1.5, min sub-cluster 100 pts |
| Semantic losses | CrossEntropy (class-weighted: `object` ×10) + Lovász |

> The 5 mm clustering radius is critical: a 3 cm radius on this dataset merges sub-mm-apart objects and collapses mAP to ~0.008 on the same checkpoint.

### Training

| Setting | Value |
|---|---|
| Batch | 6 (× grad-accum 2 micro-batches; effective batch = 6) |
| Precision | bf16 AMP (training only — inference must be fp32) |
| Optimizer | AdamW, lr 6e-3 (blocks 6e-4), wd 0.05 |
| Scheduler | OneCycleLR, cos anneal, pct_start 0.01 |
| Epochs | 200 planned, ~8,467 steps/epoch |
| Eval | every 1,000 steps on the first **500** val scenes (full val = 6,350); best-checkpoint selection by AP50 |
| Mix3D | mix_prob 0.8 |
| Hardware | 1× RTX 5090 |

```bash
bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python \
  -d scannet-v1.1.1-2of3 -c insseg-litept-small-v1m2-2of3-embed \
  -n insseg-litept-small-v1m2-2of3-embed -g 1
```

## 5. Best checkpoint result

Validation on the 500-scene val subset (classes: board, tray, paper, table, object):

| Metric | model_best (epoch 21) | Later subset peak (not saved) |
|---|---|---|
| mAP | **0.7255** | 0.7540 |
| AP@50 | **0.8218** | 0.8551 |
| AP@25 | **0.8491** | 0.8728 |

- `model_best.pth` corresponds to the **epoch-21** eval (2026-06-20); best-checkpoint tracking snapshots at epoch boundaries, so the higher mid-epoch subset evals (right column) were observed but not saved.
- The run was stopped early at **epoch 55 / 200** (2026-06-22) — subset AP50 had plateaued in the 0.75–0.85 band with noise, and the epoch-21 best was already reached; subsequent configs budget ~40 epochs based on this.
- Log: `exp/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-embed/train.log`

## 6. Inference speed & memory

Measured on **RTX 5090, fp32**, 500 val scenes (~252k voxels / ~311k original points per scene) with the base 2of3 PG-v1m2 model (same backbone, dataset, and clustering; the embed head adds only a lightweight mean-shift split on top).

| Metric | Value |
|---|---|
| Forward + clustering | mean **262 ms** (median 248, p95 435, max 566) ≈ 3.8 scenes/s |
| — with warm spconv autotune cache | mean **130 ms** (~2× faster; cache-state dependent) |
| End-to-end per scene | ~**380 ms** = preprocess (2 mm GridSample) 69 + H2D 2 + forward 262 + back-projection 47 |
| GPU activation peak / scene | mean **1.4 GB**, p95 2.4 GB, max 3.0 GB |
| True GPU working set | ~4.5 GB (activation peak + CUDA context); reserved/`nvidia-smi` beyond that is reclaimable allocator cache |
| Host RAM | ~2.1 GB fixed working set (per-scene delta ≈ 0) |

Caveats:

- **Latency and memory are content-driven, not point-count-driven** (corr with point count ≈ −0.1 to −0.2): cost is dominated by CPU BFS instance clustering and spconv sparse structure, so it stays flat as scenes grow.
- **fp32 only for inference** — bf16/AMP crashes the spconv autotuner in eval mode (training in bf16 is fine).
- **Re-measure with:** `configs/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-benchmark.py` (loads a trained checkpoint and hooks `RuntimeProfiler_inference` for per-scene latency/memory CSV; set `val_subset_size=None` for the full 6,350-scene val split).
