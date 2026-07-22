# 🧩 Instance Clustering: `embed-maskclust-strong-rag-sv`

> 💡 **TL;DR** — An *inference-time* recipe that turns LitePT's per-point predictions (semantics, offsets, embeddings) into instance masks: group points geometrically → **split** using the learned embedding and 2D D-FINE-seg masks (strict multi-view *cannot-link*) → **re-merge** fragments the strict split broke, using mask agreement in the views two fragments actually *share*.

- **Config**: `configs/scannet-v1.1.1-2of3/clustering/embed-maskclust-strong-rag-sv.py`
- **Implementation**: `PointGroup._cluster` and helpers in `models/point_group/point_group_v1m2_custom_criteria.py`
- **Entry point**: `tools/infer_insseg.py --cluster-config …`
    - Clustering config is read **separately** from the training config — training config builds the backbone/heads and picks the checkpoint; this file only decides how points are grouped
    - Precedence: model defaults < `--cluster-config` < individual CLI overrides

```
per-point predictions
        │
        ▼
① Geometric clustering        offset-shifted points, ball-query + BFS, 5 mm radius
        │                     (touching objects still merged)
        ▼
② Embedding split             mean-shift in learned embedding space
        │                     (splits touching same-class objects)
        ▼
③ Strict cannot-link split    2D-mask disagreement in ANY shared view cuts the graph,
        │                     enforced transitively (no detour leak)
        ▼                     (over-segments: one noisy point can shatter an object)
④ RAG shared-view merge       re-merge adjacent same-class fragments whose masks
        │                     AGREE in the views they share
        ▼
⑤ Size filter + scoring       keep proposals > 300 points, score by mean class confidence
```

---

## 📥 Inputs

| Input | Source | Required? |
|---|---|---|
| Semantic logits (per point) | LitePT semantic head | ✅ always |
| Offset vectors (per point) | offset/bias head — points shift toward their instance centroid | ✅ always |
| Instance embedding (per point) | embedding head — **checkpoint must be trained with `instance_embedding=True`** | ✅ for stage ② |
| `mask_per_view.npy` — `(N, S)` int array | 2D D-FINE-seg masks reprojected onto the point cloud (`tools/vggt_to_scene.py --mask-all-views`) | for stages ③–④ |

- `mask_per_view.npy` semantics:
    - Column *s* = the **globally-unique 2D-mask id** point *i* lands in when reprojected into view *s* (z-buffer occlusion aware)
    - **−1** = not visible / not covered in that view
    - Mask ids are unique per (view, object) → "two points in different masks of the same view" ⇔ some column holds two distinct non-negative values
    - Auto-loaded by `tools/infer_insseg.py` from the scene directory when present

> 💡 Stages ③ and ④ are **graceful no-ops** when the file is missing — the config degrades to plain geometric + embedding clustering. Training/validation are never affected.

---

## ① Geometric clustering — ball-query + BFS

*Code: `_cluster` (point_group_v1m2_custom_criteria.py:225)*

- Shift every point by its predicted offset: `center = coord + bias_pred` → points of one instance collapse toward a common centroid
- Drop points whose predicted class is in `segment_ignore_index = (-1, 0, 1)` (background/wall/floor-type classes)
- Group the shifted points:
    - **Ball query** with radius `cluster_thresh × voxel_size = 2.5 × 2 mm = 5 mm`, capped at `cluster_closed_points = 3000` neighbors
    - **BFS connected components**, gated so only points of the *same predicted class* connect
    - Discard components smaller than `cluster_min_points = 100`

> ⚠️ **Why 5 mm matters** — this is a small-scale industrial dataset (~2–3 m scenes, objects 6–18 cm apart, some sub-millimeter gaps). The ScanNet-style 3 cm radius merged nearly everything into one blob (mAP 0.008); shrinking to 5 mm on the *same checkpoint* recovered mAP 0.73. Keep it at 5 mm.

- **What's left unsolved**: two same-class objects that physically *touch* end up in one geometric proposal → the next two stages split them

---

## ② Embedding split — mean-shift

*Code: `_split_proposals_by_embedding` + `_meanshift_cluster`*

- Sub-divide each geometric proposal by clustering its points in the **learned embedding space**
- Algorithm: greedy seeded mean-shift, `embed_bandwidth = 1.5`, ≤ 10 iterations per seed
    - Training pulls points within *δ_v* of their instance mean and pushes instance means ≥ 2 *δ_d* apart → mean-shift with bandwidth ≈ *δ_d* converges to per-instance modes regardless of seed choice
- Single-instance proposal → one sub-cluster → passes through unchanged
- Sub-clusters smaller than `embed_min_points = 100` are dropped as noise

> 💡 Empirically this still **under-segments** touching (< 1 cm apart) objects — embeddings of adjacent instances blur into each other. That's why the 2D masks come in next.

---

## ③ Strict multi-view cannot-link split

*Code: `_split_proposals_by_mask_multiview` + `_strict_refine`*

- **Constraint injected**: two points in *different* 2D masks of the *same* camera view must never share a 3D instance
- Per proposal:
    - Build a **radius graph** over the points — physical radius `mask_split_radius × voxel_size` = 2.5 × 2 mm = **5 mm** (same scale BFS used to merge)
    - **Delete** every edge (u, v) where u and v carry distinct valid mask ids in **any shared view** (`mask_constraint_multiview = True` — one disagreeing view is enough)
    - Take connected components of the surviving graph (scipy's C `connected_components` — fast even for dense, million-edge proposals)
    - **Strict refinement** (`mask_constraint_strict = True`):
        - Plain edge-deletion has a *detour leak* — two points in different masks can re-merge through a chain of legal edges (e.g. an unlabeled `mask = −1` seam point bridging them)
        - `_strict_refine` closes the constraint **transitively**: any component holding two distinct valid mask ids in a single view is re-split by constrained region-growing — a node joins a growing sub-cluster only if it conflicts with *no* existing member (the sub-cluster's `view → mask id` map stays single-valued)
        - Already mask-consistent components skip this entirely (vectorized fast path)
    - Drop components smaller than `cluster_min_points = 100`; sub-instances inherit the parent's class
- Points sharing **no** labeled view, or agreeing in every shared view, still group by adjacency → an object spanning several camera views is *not* shattered by the constraint itself

> ⚠️ **The cost of strictness** — with transitive enforcement, a *single* noisy point mislabeled into a neighboring mask can sever a real object into fragments. The strict variant systematically **over-segments** — exactly the failure mode stage ④ heals.

---

## ④ RAG shared-view merge — the `sv` in the name

*Code: `_merge_proposals_by_mask_histogram` (point_group_v1m2_custom_criteria.py:817), `rag_sim_metric = "shared_view"`*

- **Idea**: a post-clustering **Region Adjacency Graph** re-merges fragments the strict split broke apart, using *aggregate* mask statistics of whole clusters — hundreds of points drown out the lone noisy point that caused the cut
- **Adjacency**: one global cKDTree over all proposal points; two proposals are RAG-adjacent if any cross-proposal point pair lies within `rag_adjacency_radius × voxel_size` = **5 mm**
- **Merge test** per adjacent pair (a, b) — only when they share the **same predicted class**:
    - With `rag_sim_metric = "shared_view"`, compute the **soft cross-pair conflict ratio** over the views the two clusters *share*:

        ```
        conflict = 1 − (cnt_a · cnt_b) / (vc_a · vc_b)
        ```

    - `cnt_p` = cluster *p*'s count histogram over global (view, mask) ids; `vc_p[s]` = number of *p*'s points visible in view *s*
    - Denominator = all co-visible cross-pairs over shared views; numerator = cross-pairs landing in the **same** 2D mask → `conflict` = fraction of co-visible point-pairs falling in *different* masks
    - **Merge when `conflict ≤ rag_merge_thresh = 0.3`**
- **Union**: connected components over merge edges are unioned (proposal masks are disjoint → clamped sum = union); merged instance inherits the class of its **largest** fragment; proposals in no merge edge pass through unchanged

> ⚠️ **Threshold direction is inverted vs. legacy metrics** — for `"intersection"`/`"cosine"` a *higher* threshold is stricter (minimum similarity); for `"shared_view"` a **lower** threshold is stricter (*maximum allowed conflict*) — fewer, larger merges as it decreases. Don't copy threshold values between metric variants.

> 💡 **Why shared-view instead of the legacy global histogram** — the legacy intersection metric compares *global* mask histograms across all views. Two genuine fragments of one object visible in **different view sets** get disjoint histograms → similarity ≈ 0 → never merged, even when they agree perfectly in every view they actually share. The shared-view metric restricts evidence to shared views, fixing this. Adjacent same-class pairs sharing **no** view carry no mask evidence and are conservatively left unmerged.

---

## ⑤ Final filter & scoring

- Size gate `cluster_propose_points = 300` runs **after** the RAG merge, deliberately — re-merged fragments that were individually too small can clear it together
- Confidence per proposal = **mean softmax probability of its predicted class** over its member points
- Output: `pred_masks`, `pred_classes`, `pred_scores`

---

## ⚙️ Parameter reference

| Parameter | Value | Physical meaning | Effect of increasing |
|---|---|---|---|
| `voxel_size` | 0.002 | 2 mm grid — must match the dataset's voxelization | — |
| `cluster_thresh` | 2.5 | geometric grouping radius = 2.5 × 2 mm = **5 mm** | merges more aggressively |
| `cluster_closed_points` | 3000 | ball-query neighbor cap | — |
| `cluster_min_points` | 100 | min component size in stages ① and ③ | drops more small fragments |
| `cluster_propose_points` | 300 | final proposal size gate (after merge) | drops more small instances |
| `segment_ignore_index` | (−1, 0, 1) | classes excluded from instancing | — |
| `instance_embedding` | True | enable stage ② (needs matching checkpoint) | — |
| `embed_bandwidth` | 1.5 | mean-shift bandwidth in embedding space | fewer, larger sub-clusters |
| `embed_min_points` | 100 | min embedding sub-cluster size | drops more sub-clusters |
| `cluster_mask_constraint` | True | enable stage ③ | — |
| `mask_constraint_multiview` | True | cannot-link fires in **any** shared view | — |
| `mask_constraint_strict` | True | transitively-closed constraint (no detour leak) | — |
| `mask_split_radius` | 2.5 | split-graph adjacency radius (**5 mm**, = `cluster_thresh`) | — |
| `cluster_mask_rag_merge` | True | enable stage ④ | — |
| `rag_merge_thresh` | 0.3 | **max** conflict ratio that still merges | merges more aggressively (**lower = stricter**) |
| `rag_adjacency_radius` | 2.5 | RAG adjacency radius (**5 mm**, = `mask_split_radius`) | more pairs considered for merging |
| `rag_sim_metric` | `"shared_view"` | soft cross-pair conflict over shared views | — |

---

## 🗺️ Where this config sits

- The `clustering/` directory is a strictness spectrum over the same trained model:
    - **`embed`** — stages ① + ② only → under-segments touching (< 1 cm) objects
    - **`embed-maskclust*`** — add the cannot-link split ③ (single-view / non-strict / multi-view-strict variants) → the strict variant over-segments
    - **`embed-maskclust-strong-rag`** — strict split + RAG merge with the legacy global histogram intersection
    - **`embed-maskclust-strong-rag-sv`** *(this config)* — same, but the merge uses the shared-view conflict ratio, fixing the disjoint-view-set flaw
- A standalone, verified port of this exact pipeline (for use outside the model) lives in `tools/cluster_insseg_strong_rag_sv.py`
