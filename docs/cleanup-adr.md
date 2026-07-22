# Cleanup ADR Ledger

Append-only decision ledger for repo hygiene / environment cleanup.

**Rules:**
- Entry bodies are immutable once written. New information → new entry.
- Only the `Status` line of an existing entry may be edited, and only to advance it
  (e.g. `proposed → accepted → done`, or `→ superseded-by-#NNN`).
- If entry N negates entry K, do NOT delete K: append N, then mark K's status
  `superseded-by-#N`.
- Monotonic entry numbers, one decision per entry, no nesting.

**Statuses:** `proposed` | `accepted` | `done` | `superseded-by-#NNN` | `rejected`

---

## #001 — Establish this ledger as the tracked record of cleanup decisions
- Date: 2026-07-22
- Status: done
- Context: Repo had zero tracked agent/maintenance docs; cleanup rationale lived only in
  private Claude session memory. Project history shows conclusions get revised
  (e.g. the 2mm "insseg collapse" was later re-diagnosed as an eval clustering-radius
  artifact), so preserving the wrong-then-corrected chain matters.
- Decision: Append-only ledger at `docs/cleanup-adr.md`, statuses mutable, bodies immutable.
- Consequences: `docs/` becomes a tracked directory. Current-state guidance (which python,
  how to launch training) belongs in a future `CLAUDE.md`, not here — this file is the *why*-log.

## #002 — Expand .gitignore to cover build/log/agent-state artifacts
- Date: 2026-07-22
- Status: done
- Context: `git status` shows 68 untracked entries. Un-ignored noise includes `logs/` (190MB),
  `eval/` (9.5GB, 218 epoch dirs), `.omc/`, `libs/*/build/` (~94MB), `*.egg-info/`,
  223 `__pycache__` dirs, `.claude/settings.local.json`.
- Decision: Add `logs/`, `eval/`, `.omc/`, `libs/**/build/`, `*.egg-info/`, `__pycache__/`,
  `.claude/settings.local.json` to `.gitignore`. Ignore rather than relocate, because training
  scripts and hooks hardcode these output paths.
- Consequences: `git status` becomes a usable signal again. Eval outputs remain invisible to
  git; archive strategy (if any) is a separate decision.

## #003 — Remove the global `*.json` ignore rule
- Date: 2026-07-22
- Status: done
- Context: `.gitignore` contains a bare `*.json`, which silently ignores any config, manifest,
  or camera/extrinsics JSON anywhere in the tree. In a config-driven ML repo this will
  eventually swallow a file that was meant to be committed.
- Decision: Delete the global `*.json` rule; replace with targeted path-scoped rules only where
  large generated JSON actually accumulates (identify offenders before writing the rules).
- Consequences: Some previously-hidden JSON may appear as untracked and need triage — that is
  the point.

## #004 — Track `docs/` content
- Date: 2026-07-22
- Status: done
- Context: `docs/` holds three substantive clustering write-ups
  (`clustering-embed-maskclust-strong-rag-sv.md`, `clustering_sv_svdom.md`,
  `insseg-litept-small-v1m2-2of3-embed.md`) that exist only on this machine.
- Decision: Commit all current `docs/*.md`.
- Consequences: Analysis docs survive machine loss and are visible to teammates/agents.

## #005 — Triage the untracked configs and tools scripts (commit, don't scratch)
- Date: 2026-07-22
- Status: proposed
- Context: ~43 untracked configs under `configs/scannet-v1.1.1-2of3/` and ~30 untracked
  `tools/` scripts are real work products — they are the reproducibility record of the
  ablation grid and the VGGT/clustering pipelines. 9 tracked files also carry +1,374
  uncommitted lines on `feat/query`.
- Decision: Default disposition is COMMIT (grouped, coherent commits on `feat/query`).
  True one-off throwaways move to an ignored `scratch/` dir instead of repo root
  (e.g. `scratch_wandb.py`).
- Consequences: Branch history reflects the actual experiment record; repo root stays clean.

## #006 — Add a repo `CLAUDE.md` with environment non-negotiables
- Date: 2026-07-22
- Status: done
- Context: Critical operational facts exist only in private session memory: must use
  `/home/fai/miniconda3/envs/litept/bin/python`; long trainings need `setsid nohup`
  (CC background tasks die on session exit); eval must be fp32 (spconv autotuner breaks
  under bf16); dataset is cm-scale industrial, not room-scale ScanNet; 2of3 is 2mm voxels.
- Decision: Write `CLAUDE.md` (~40 lines) capturing current-state operating rules, linking
  here for rationale.
- Consequences: Fresh agent sessions and teammates start with the footgun list instead of
  rediscovering it.

## #007 — Sync `requirements.txt` with the locked env
- Date: 2026-07-22
- Status: done
- Context: `scripts/setup_litept_env.sh` + `scripts/requirements-litept-lock.txt` are the real
  env definition; root `requirements.txt` is unpinned/stale and misleads anyone who finds it first.
- Decision: Add a header comment to `requirements.txt` pointing to the installer + lockfile
  (or replace its contents with that pointer).
- Consequences: One canonical env path; no accidental unpinned installs.

## #008 — Relocate `eval/` output root out of the source tree (low priority)
- Date: 2026-07-22
- Status: proposed
- Context: 9.5GB of per-epoch eval output lives at repo root. Ignoring it (#002) hides the
  noise but the disk weight and backup/rsync friction remain.
- Decision: Deferred option — point eval output at `exp/` (already ignored) or an external
  path via config. Do not do this mid-training-campaign; path changes risk breaking
  resume/eval hooks.
- Consequences: If done, #002's `eval/` ignore rule becomes vestigial but stays (append a
  superseding entry then).

## #009 — Ignore `viz/` (generated visualization output)
- Date: 2026-07-22
- Status: done
- Context: `viz/` holds 4.1GB of generated PNG/demo output across 19 experiment dirs; no
  source code lives there (whole dir was untracked).
- Decision: Add `viz/` to `.gitignore`. Viewer/plotting *code* lives in `tools/` and is tracked.
- Consequences: If a viz subdir ever needs sharing, copy it out or negate the rule for that path.
