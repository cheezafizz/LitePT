# LitePT — Operating Rules for Agents

Current-state rules only. Rationale/history lives in `docs/cleanup-adr.md` and git log.

## Environment (non-negotiable)

- **Python**: always use the conda env interpreter
  `/home/fai/miniconda3/envs/litept/bin/python`. The system python will not work
  (spconv/pointops/pointgroup_ops are built against this env).
  Training: `bash scripts/train.sh -p /home/fai/miniconda3/envs/litept/bin/python ...`
- **Env setup / reproduction**: `scripts/setup_litept_env.sh` +
  `scripts/requirements-litept-lock.txt` are the canonical env definition.
  Root `requirements.txt` is only a pointer.
- **Long trainings must be detached**: launch with `setsid nohup ... &` and log to a file.
  Claude-Code-managed background tasks die when the session exits (a run was lost this way).
- **Eval/inference is fp32 only**: bf16/AMP inference crashes the spconv autotuner.
  Training in bf16 is fine.
- **This host (fai-5090) has a history of silent hardware/power crashes** — a dead training
  with no OOM/traceback may be a machine crash, not a code bug. Check `journalctl --list-boots`.

## Data / scale facts

- `scannet-v1.1.1-2of3` is a **small-scale industrial dataset** (scenes ~2–3 m, objects
  6–18 cm apart, 7 classes), NOT room-scale ScanNet. Scale geometric augmentations in cm.
- 2of3 data is **2 mm voxels** (config `grid_size=0.002`), cropped.
- Instance clustering radius must match voxel scale: 5 mm radius
  (`voxel_size=0.002, cluster_thresh=2.5`); a 3 cm radius merges adjacent objects and
  destroys mAP.
- D-FINE pseudo-label `sem_seg` class scheme ≠ training scheme — always remap
  (`tools/build_pseudo_label_dataset.py`).

## Layout

- `configs/scannet-v1.1.1-2of3/` — the experiment grid (one config per ablation).
- `exp/` — training outputs (ignored). `eval/`, `logs/` — generated (ignored).
- `tools/` — analysis/inference/one-off scripts. `scratch/` — throwaways (ignored).
- Viser visualization uses a separate venv: `.venv-viser` (`tools/setup_viser_venv.sh`).

## Git

- Work happens on feature branches (`feat/query` currently); commit experiment configs and
  tools scripts — they are the reproducibility record.
- Cleanup/maintenance decisions: append to `docs/cleanup-adr.md` (append-only ledger;
  never delete entries, mark superseded).
