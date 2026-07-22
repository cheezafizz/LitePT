# Research Question Ledger

Sibling of `docs/cleanup-adr.md`. Append-only ledger of research questions, the runs that
probe them, and their (always-provisional) conclusions.

**Rules:**
- One entry per research question (hypothesis), numbered `#Q001…`. Not per run — runs are
  fields inside an entry.
- Entry bodies are immutable once written. New evidence that *changes a conclusion* → new
  entry, then mark the old one `superseded-by-#QNNN`. Only the `Status` line may be edited.
- **Add the entry at launch time** (Status: `running`), not after results — the ledger is the
  plan, not just the record.
- Primary keys for evidence are **config name + wandb run id**; local ckpt/eval paths are
  convenience only (they move/expire).
- Durable write-ups promoted from agent state or notebooks land in `docs/` and are linked
  from the entry (see cleanup ADR #010).

**Statuses:** `open` (question posed, no run yet) | `running` | `answered` (provisional) |
`superseded-by-#QNNN` | `abandoned`

---

## #Q001 — Why did 2mm re-voxelization collapse insseg mAP (0.008)?
- Date: 2026-06 (retro-seeded 2026-07-22)
- Status: answered
- Hypothesis (initial): receptive-field/scale mismatch of the network at 2mm.
- Runs: 2mm ablation grid (`-2mm-rf`, `-2mm-net5mm`, `-2mm-aug`, `-2mm-full`, `-2mm-fixclust`)
- Conclusion: NOT an RF/scale problem. Eval-time clustering radius artifact: the default
  `voxel_size=0.02` gave a 3cm grouping radius that merged sub-mm-apart objects. Explicit
  `voxel_size=0.002, cluster_thresh=2.5` (5mm radius) → mAP 0.008 → 0.73 on the same ckpt.
- Implication: clustering hyper-params must scale with voxel size; fix committed to the base
  2of3 config. The initial RF hypothesis consumed several ablation runs — evidence of why
  this ledger exists.

## #Q002 — What clustering strategy segments touching (<1cm) objects on VGGT clouds?
- Date: 2026-06/07 (retro-seeded)
- Status: answered
- Hypothesis: offset-embedding clustering alone suffices.
- Runs: `-embed`, `-embed-maskclust*` family + `configs/.../clustering/` sweep
- Conclusion: Goldilocks result — plain embed UNDER-segments touching objects; strict
  cannot-link OVER-segments; intermediate (single-view / non-strict) cannot-link is best.
  Keep 5mm radius throughout.
- Implication: `embed-maskclust-strong-rag-sv` became the production clustering config.

## #Q003 — Can post-clustering merging heal cannot-link over-segmentation?
- Date: 2026-07 (retro-seeded)
- Status: answered
- Hypothesis: per-cluster mask-histogram + region-adjacency merge recovers split objects.
- Runs: `-embed-maskclust-allviews-rag` (A/B vs non-RAG)
- Conclusion: yes — RAG histogram merge improves over-seg cases (see
  `_merge_proposals_by_mask_histogram`, A/B in memory/docs).

## #Q004 — Is clustering the bottleneck for GT-free VGGT insseg?
- Date: 2026-07 (retro-seeded)
- Status: answered
- Hypothesis: better clustering is the highest-leverage fix.
- Runs: `tools/oracle_insseg_diagnostic.py` (oracle-input ceiling study, no training)
- Conclusion: NO. Oracle-input ceiling is 0.84 mAP — semantic head quality + domain gap
  dominate. Data/pseudo-labels first; query head second-order.
- Implication: redirected effort from clustering tweaks to pseudo-label pipeline + query model.

## #Q005 — Do synthetic-trained normals transfer to VGGT point clouds?
- Date: 2026-06/07 (retro-seeded)
- Status: answered
- Hypothesis: normals are a free win on real data.
- Runs: `-normaug`, `-coloronly`; `tools/analyze_normal_gap.py`
- Conclusion: large gap — synthetic median error 0°, VGGT median 16°. Mitigation:
  NormalJitter+NormalDropout aug; coloronly as fallback ablation.

## #Q006 — Does filtered_valid_mask improve VGGT surface quality enough to keep?
- Date: 2026-07 (retro-seeded)
- Status: answered
- Conclusion: yes (now default in `vggt_to_scene.py`) — halves surface noise at ~42%
  retention. Caveat: fixed-k normal metrics are density-confounded; use fixed-radius.

## #Q007 — Does a learnable-query head (MQ-v1m1) beat PG-v1m2 offset clustering?
- Date: 2026-07 (retro-seeded)
- Status: running
- Hypothesis: Mask3D/SPFormer-style queries remove CPU clustering and its radius
  sensitivity, at equal or better mAP.
- Runs: `-query`, `-query-muon` (race), `-query-realft*`; benchmark configs for cost.
- Evidence so far: inference cost acceptable (fwd 287ms vs PG 130-262ms; +1GB activation;
  no CPU clustering). Mask resolution matters: overfit mIoU 0.30@10mm → 0.80@6mm →
  context_grid_factor=1 via grad-accum.
- Conclusion: pending scaled A/B vs embed-maskclust baseline.

## #Q008 — Is the aux semantic head (and not-object loss) necessary for MQ-v1m1?
- Date: 2026-07-15
- Status: running
- Runs: `-query-realft-muon-nosem` (launched 2026-07-15)
- Conclusion: pending.

## #Q009 — How does num_queries affect cost and what is the floor?
- Date: 2026-07 (retro-seeded)
- Status: answered
- Conclusion: K barely affects step time (context tokens dominate: 6.4s @K=50 vs
  6.99s @K=100). Floor K=50 forced by Mix3D pair-merging (~46 GT inst/sample max).
  Trained ckpts need `num_queries` override at inference.

## #Q010 — Do pseudo-labels from D-FINE need class remapping?
- Date: 2026-07 (retro-seeded)
- Status: answered
- Conclusion: YES, mandatory — D-FINE scheme ≠ training scheme (object 5→6, oob→object);
  without remap every object becomes "table". `tools/build_pseudo_label_dataset.py` handles it.

## #Q011 — Does MuonAdamW beat AdamW for the query model at equal budget?
- Date: 2026-07 (retro-seeded)
- Status: running
- Runs: `-query-muon` vs `-query` race (epoch=40 budget from embed's epoch-21 best).
- Conclusion: pending.
