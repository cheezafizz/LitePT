# How `embed-maskclust-strong-rag-sv` and `-svdom` Cluster a Point Cloud into Instances

## TL;DR

- Both are **inference-time clustering configs** for `tools/infer_insseg.py --cluster-config`. They do **not** define the network — they decide how the network's per-point predictions get grouped into object instances.
- Both run the **same 5-stage pipeline**: geometric grouping → embedding split → 2D-mask cannot-link split → RAG merge → size filter & scoring.
- They are **byte-for-byte identical except one knob**: the RAG-merge similarity metric.
  - `sv` → `rag_sim_metric = "shared_view"` — a **soft, point-pair-weighted** conflict ratio.
  - `svdom` → `rag_sim_metric = "shared_view_dominant"` — a **coarse, per-view majority-mask** vote.
- For both, `rag_merge_thresh = 0.3` is the **max conflict allowed to still merge** → **lower = stricter = fewer, larger instances** (opposite of the legacy similarity metrics).

## Inputs

- **Per-point network outputs** (from the backbone + heads, selected by the *training* config, not this file):
  - `semantic logits` → each point's predicted class (`argmax` of softmax).
  - `offset / bias` → a vector pointing each point toward its instance centroid.
  - `embedding` → a discriminative per-point vector (same-instance points pulled together, different instances pushed apart).
- **`mask_per_view.npy` — shape `(N, S)`** — for each point `N` and each camera view `S`, the globally-unique 2D-mask id that point lands in when reprojected into that view (`-1` = not visible / not covered there). Produced by the 2D segmentation (D-FINE-seg) reprojection.
  - `tools/infer_insseg.py` **auto-feeds** this file when it exists next to the scene.
  - The cannot-link split (Stage 3) and the RAG merge (Stage 4) **consume it**. Without it, both stages are **graceful no-ops** and you fall back to geometry + embedding only.

## Pipeline (shared by both configs)

### Stage 1 — Geometric grouping (ball-query + BFS)

- **Always active.** Params: `voxel_size = 0.002`, `cluster_thresh = 2.5` → physical grouping radius = `2.5 × 0.002 m` = **5 mm**.
- Shift each point by its predicted offset toward its centroid: `center = coord + bias`, then convert to voxel units.
- Drop points whose predicted class is in `segment_ignore_index = (-1, 0, 1)` (background-type classes).
- **Ball-query + BFS connected components**, gated by predicted class (only same-class neighbors within 5 mm connect) → **geometric proposals**.
- Drop proposals smaller than `cluster_min_points = 100`.

### Stage 2 — Embedding split

- Enabled by `instance_embedding = True`; `embed_bandwidth = 1.5`, `embed_min_points = 100`.
- Within each geometric proposal, run **mean-shift clustering on the per-point embeddings** to pull apart touching **same-class** objects that ball-query + BFS wrongly fused into one blob.
- Sub-clusters smaller than `embed_min_points` are dropped as noise.
- Requires a checkpoint trained with `instance_embedding = True`.

### Stage 3 — 2D-mask cannot-link split (multi-view, strict)

- Enabled by `cluster_mask_constraint = True`, `mask_constraint_multiview = True`, `mask_constraint_strict = True`; `mask_split_radius = 2.5` → **5 mm** (same scale as BFS).
- Within each proposal, build a radius graph on the points, then **cut** any edge whose two endpoints fall in **different 2D masks of ANY shared view** (a cannot-link — they cannot be the same object).
- `mask_constraint_strict = True` → **transitively closed**: if two distinct masks re-merge through a detour of otherwise-legal edges, that component is re-split so the leak is closed.
- Take connected components of what survives → sub-instances (each < `cluster_min_points` dropped).
- This is the **most aggressive splitter**. A single noisy 2D-mask point can shatter a real object into fragments at touching-object seams — which is exactly what Stage 4 exists to heal.

### Stage 4 — RAG merge (the healer) — **where `sv` and `svdom` diverge**

- Enabled by `cluster_mask_rag_merge = True`; `rag_adjacency_radius = 2.5` → **5 mm**, `rag_merge_thresh = 0.3`.
- Build a **Region Adjacency Graph (RAG)**: nodes = the sub-instance proposals; an edge joins two proposals that have points **within 5 mm** of each other (found with one global cKDTree over all member points, keeping cross-owner pairs).
- An edge becomes a **merge edge** only if **both** hold:
  - the two proposals share the **same predicted class**, AND
  - they pass the **`rag_sim_metric` agreement test** (this is the sv/svdom knob — see next section).
- **Union** the connected components of the merge edges (proposal masks are disjoint, so a clamped sum = the union); the merged instance **inherits the class of its largest fragment**.
- Runs **before** the size gate (Stage 5), so re-merged fragments can grow past the size threshold.
- Intuition: because each proposal's mask evidence is aggregated over hundreds of points, one lone noisy point that severed a real object is **drowned out by the majority** → the halves re-merge, while two genuinely distinct touching objects keep dissimilar mask evidence and **stay apart**.

### Stage 5 — Size filter & scoring

- Keep only proposals with more than `cluster_propose_points = 300` points.
- **Score** = mean softmax confidence of the proposal's predicted class over its member points.
- **Class** = the proposal's predicted class. Output = `pred_masks`, `pred_scores`, `pred_classes`.

## The one difference: `sv` vs `svdom`

Both metrics answer the same question — *"do these two adjacent same-class fragments belong to the same object?"* — using only the camera views the two fragments **share**, and both **merge when `conflict ≤ rag_merge_thresh` (0.3)**.

- **Shared by both:**
  - Evidence is restricted to **shared views only** (views where *both* fragments have visible points). This fixes the legacy flaw where two real fragments seen in *different* view sets look dissimilar even though they agree perfectly where they overlap.
  - Adjacent same-class fragments that **share no view** carry no mask evidence → **left unmerged**.
  - Direction is **inverted** vs the legacy `intersection` / `cosine` metrics: there, higher threshold = stricter; here, **lower threshold = stricter = fewer, larger merges**.

- **`sv` = `shared_view` — soft, point-pair-weighted:**
  - Counts, over the shared views, the fraction of **co-visible point-pairs** that land in **different** 2D masks.
  - `conflict = 1 − (cnt_a · cnt_b) / (vc_a · vc_b)`
    - `cnt` = per-fragment count histogram over global `(view, mask)` ids.
    - `vc[s]` = per-fragment count of points visible in view `s`.
  - Sensitive to **how much** of the overlap disagrees — a sizeable minority of conflicting point-pairs raises the conflict and can block the merge.

- **`svdom` = `shared_view_dominant` — coarse, per-view majority vote:**
  - `conflict = (number of shared views whose DOMINANT (modal) mask differs) / (number of shared views)`.
  - Each shared view contributes a single yes/no vote based only on each fragment's **most common** mask in that view.
  - **Ignores minority within-view disagreement** — a few stray points in the "wrong" mask don't count as long as the majority mask agrees.

- **When they differ in practice:**
  - `svdom` is **more tolerant of within-view mask noise / bleed** (only the majority mask votes) → tends to merge more, giving larger instances.
  - `sv` is **more precise** but is more easily pushed over the threshold by a substantial minority of conflicting pairs → tends to keep more splits.
  - If your 2D masks are clean, the two behave similarly; the gap widens as mask noise increases.

## Config knobs quick reference

Shared by both `sv` and `svdom` (values in `cluster = dict(...)`):

- **Geometric:** `voxel_size = 0.002`, `cluster_thresh = 2.5` (→ 5 mm), `cluster_closed_points = 3000`, `cluster_propose_points = 300`, `cluster_min_points = 100`, `segment_ignore_index = (-1, 0, 1)`.
- **Embedding split:** `instance_embedding = True`, `embed_bandwidth = 1.5`, `embed_min_points = 100`.
- **Cannot-link:** `cluster_mask_constraint = True`, `mask_constraint_multiview = True`, `mask_constraint_strict = True`, `mask_split_radius = 2.5` (→ 5 mm).
- **RAG merge:** `cluster_mask_rag_merge = True`, `rag_merge_thresh = 0.3`, `rag_adjacency_radius = 2.5` (→ 5 mm).
- **The only differing line:**
  - `sv` → `rag_sim_metric = "shared_view"`
  - `svdom` → `rag_sim_metric = "shared_view_dominant"`

## How to run

- `python tools/infer_insseg.py ... --cluster-config embed-maskclust-strong-rag-sv` (or `...-svdom`).
- `mask_per_view.npy` is auto-fed when present next to the scene; without it the cannot-link and RAG stages are skipped.
- Threshold-sweep siblings exist for both (`...-sv-t01 … -t05` and `...-svdom-t01 … -t05`) that only vary `rag_merge_thresh`.

## Where this lives in code

- Configs: `configs/scannet-v1.1.1-2of3/clustering/embed-maskclust-strong-rag-sv.py` and `...-svdom.py`.
- Pipeline: `models/point_group/point_group_v1m2_custom_criteria.py`
  - `_cluster` — orchestrates all 5 stages.
  - `_split_proposals_by_embedding` / `_meanshift_cluster` — Stage 2.
  - `_split_proposals_by_mask_multiview` / `_strict_refine` — Stage 3.
  - `_merge_proposals_by_mask_histogram` — Stage 4 (the `sv` vs `svdom` branch lives here).
