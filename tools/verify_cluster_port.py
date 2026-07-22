"""
Verify tools/cluster_insseg_strong_rag_sv.py is a FAITHFUL PORT of the
``embed-maskclust-strong-rag-sv`` config clustering process.

The standalone drops the model's offset, embedding split, and semantic head, so it can
only equal the config EXACTLY when those are neutralized. We do exactly that and then
assert the standalone produces IDENTICAL instance partitions on IDENTICAL inputs.

Part A -- stage-level differential (CPU, no GPU / no checkpoint):
  Feed identical inputs to the standalone's ported numpy functions AND the model's bound
  criteria methods, for each shared stage, and assert the induced partitions are identical:
    * cannot-link  : STA._split_by_mask_multiview   vs PointGroup._split_proposals_by_mask_multiview
    * strict refine: STA._strict_refine             vs PointGroup._strict_refine
    * RAG merge    : STA._merge_by_mask_histogram   vs PointGroup._merge_proposals_by_mask_histogram
  Inputs: many randomized scenes + structured edge cases (unlabeled-seam detour, no shared
  view, single/zero proposals).

Part B -- real-scene integration (GPU + pointgroup_ops, fp32):
  Neutralize the config path (offset=0 via run_backbone(zero_offset=True); embedding OFF via
  model.instance_embedding=False; UNIFORM one-hot semantics so bfs is class-agnostic like the
  standalone) and compare, per val_sample scene, the config _cluster partition vs the standalone
  on the SAME grid (voxelize=False), restricted to the object subset. This also exercises the
  reimplemented geometric connected-components (scipy) vs the config's CUDA ball-query+BFS.

Run with the litept env, fp32 (bf16 breaks the spconv autotuner in eval):
  /home/fai/miniconda3/envs/litept/bin/python tools/verify_cluster_port.py --part A
  /home/fai/miniconda3/envs/litept/bin/python tools/verify_cluster_port.py --part all
"""
import argparse
import os
import sys
import traceback

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.join(_REPO_ROOT, "tools")
for _p in (_REPO_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cluster_insseg_strong_rag_sv as STA  # the standalone port under test  # noqa: E402

# strong-rag-sv model-independent clustering params (config == STA.DEFAULTS).
PARAMS = dict(
    voxel_size=0.002,
    cluster_thresh=2.5,
    cluster_min_points=100,
    cluster_propose_points=300,
    mask_split_radius=2.5,
    mask_constraint_strict=True,
    rag_merge_thresh=0.3,
    rag_adjacency_radius=2.5,
    rag_sim_metric="shared_view",
)


# --------------------------------------------------------------------------- #
# partition equality (exact, label-permutation invariant, INCLUDES the -1 group)
# --------------------------------------------------------------------------- #
def labels_from_members(n, members):
    """List of index arrays -> per-point label (n,), -1 = unassigned."""
    lab = np.full(n, -1, dtype=np.int64)
    for i, idx in enumerate(members):
        lab[np.asarray(idx, dtype=np.int64)] = i
    return lab


def labels_from_masks(masks_2d):
    """(P, n) 0/1 (disjoint) -> per-point label (n,), -1 = unassigned."""
    masks_2d = np.asarray(masks_2d)
    n = masks_2d.shape[1] if masks_2d.ndim == 2 else 0
    lab = np.full(n, -1, dtype=np.int64)
    for p in range(masks_2d.shape[0]):
        lab[masks_2d[p].astype(bool)] = p
    return lab


def partitions_equal(a, b):
    """(ok, diag). ok iff a and b induce the SAME partition of indices (a bijection
    between the two label sets exists), counting the -1/unassigned group as a label so
    the two must agree on which points are assigned. Exact -- no float tolerance."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        return False, dict(reason="shape_mismatch", shape_a=a.shape, shape_b=b.shape)
    pairs = set(zip(a.tolist(), b.tolist()))
    na, nb, npair = len(set(a.tolist())), len(set(b.tolist())), len(pairs)
    ok = na == nb == npair
    return ok, dict(
        n_labels_a=na, n_labels_b=nb, n_pairs=npair,
        n_assigned_a=int((a >= 0).sum()), n_assigned_b=int((b >= 0).sum()),
        n_instances_a=int(len(set(a[a >= 0].tolist()))),
        n_instances_b=int(len(set(b[b >= 0].tolist()))),
    )


# --------------------------------------------------------------------------- #
# reference (model) side -- bare PointGroup carrying only the clustering attrs
# --------------------------------------------------------------------------- #
def make_reference():
    from models.point_group.point_group_v1m2_custom_criteria import PointGroup
    obj = PointGroup.__new__(PointGroup)  # skip __init__: no backbone / weights needed
    obj.voxel_size = PARAMS["voxel_size"]
    obj.mask_split_radius = PARAMS["mask_split_radius"]
    obj.mask_constraint_strict = PARAMS["mask_constraint_strict"]
    obj.cluster_min_points = PARAMS["cluster_min_points"]
    obj.rag_adjacency_radius = PARAMS["rag_adjacency_radius"]
    obj.rag_merge_thresh = PARAMS["rag_merge_thresh"]
    obj.rag_sim_metric = PARAMS["rag_sim_metric"]
    obj.record_split_cause = False
    return obj


def _members_to_proposals(torch, members, n):
    P = len(members)
    proposals = torch.zeros((P, n), dtype=torch.int)
    for p, idx in enumerate(members):
        if len(idx):
            proposals[p, torch.from_numpy(np.asarray(idx)).long()] = 1
    instance_pred = torch.full((P,), 5, dtype=torch.long)  # uniform class
    return proposals, instance_pred


def ref_split(obj, members, coord, mv, n):
    import torch
    proposals, instance_pred = _members_to_proposals(torch, members, n)
    nm, _ = obj._split_proposals_by_mask_multiview(
        proposals, instance_pred,
        torch.from_numpy(coord).float(), torch.from_numpy(mv).long())
    return labels_from_masks(nm.detach().cpu().numpy())


def ref_merge(obj, members, coord, mv, n):
    import torch
    proposals, instance_pred = _members_to_proposals(torch, members, n)
    nm, _ = obj._merge_proposals_by_mask_histogram(
        proposals, instance_pred,
        torch.from_numpy(coord).float(), torch.from_numpy(mv).long())
    return labels_from_masks(nm.detach().cpu().numpy())


def ref_strict(base, pairs, mv):
    from models.point_group.point_group_v1m2_custom_criteria import PointGroup
    return PointGroup._strict_refine(base.copy(), pairs.copy(), mv.copy())


# --------------------------------------------------------------------------- #
# standalone side
# --------------------------------------------------------------------------- #
def sta_split(members, coord, mv, n):
    radius = PARAMS["mask_split_radius"] * PARAMS["voxel_size"]
    new_members = STA._split_by_mask_multiview(
        members, coord, mv, radius,
        strict=PARAMS["mask_constraint_strict"], min_points=PARAMS["cluster_min_points"])
    return labels_from_members(n, new_members)


def sta_merge(members, coord, mv, n):
    radius = PARAMS["rag_adjacency_radius"] * PARAMS["voxel_size"]
    merged = STA._merge_by_mask_histogram(
        members, coord, mv, radius, PARAMS["rag_merge_thresh"])
    return labels_from_members(n, merged)


def sta_strict(base, pairs, mv):
    return STA._strict_refine(base.copy(), pairs.copy(), mv.copy())


# --------------------------------------------------------------------------- #
# synthetic scene / case generation
# --------------------------------------------------------------------------- #
def random_scene(rng):
    """A small point cloud of K touching gaussian blobs, each blob given a distinct
    globally-unique 2D-mask id in a random subset of the S views (with some -1 noise).
    Returns coord (N,3) float m, mv (N,S) int64."""
    vs = PARAMS["voxel_size"]
    K = int(rng.integers(2, 6))
    S = int(rng.integers(1, 6))
    spacing = 3.0 * vs  # blob centers ~3 voxels apart -> touch within the 5mm radius
    std = 1.0 * vs
    coord_parts, blob_of = [], []
    mask_id = 0
    # per (blob, view) global mask id table
    blob_view_mask = np.full((K, S), -1, dtype=np.int64)
    for k in range(K):
        npts = int(rng.integers(30, 130))
        center = np.array([k * spacing, 0.0, 0.0]) + rng.normal(0, 0.5 * vs, 3)
        pts = center + rng.normal(0, std, size=(npts, 3))
        coord_parts.append(pts)
        blob_of.append(np.full(npts, k))
        for s in range(S):
            if rng.random() < 0.7:  # blob visible & labeled in this view
                blob_view_mask[k, s] = mask_id
                mask_id += 1
    coord = np.concatenate(coord_parts, 0).astype(np.float64)
    blob_of = np.concatenate(blob_of, 0)
    N = coord.shape[0]
    mv = np.full((N, S), -1, dtype=np.int64)
    for s in range(S):
        for k in range(K):
            sel = blob_of == k
            mv[sel, s] = blob_view_mask[k, s]
    # -1 label noise: randomly blank some entries (unlabeled seam points)
    noise = rng.random((N, S)) < 0.15
    mv[noise] = -1
    return coord, mv


def merge_firing_cases(rng):
    """Structured RAG-merge cases where the merge path MUST actually fire (or must not),
    so the comparison of STA._merge_by_mask_histogram vs the model method is NOT vacuous.

    Returns list of (members, coord, mv, should_merge). Each case is ONE gaussian blob
    split into two adjacent halves by median-x; the two halves are same-class and within
    rag_adjacency_radius, so the merge decision is driven purely by mask agreement:
      * same mask ids per view  -> conflict 0            -> MERGE
      * disjoint mask ids        -> conflict 1            -> keep apart
      * mixed                    -> conflict ~0.5 (> 0.3) -> keep apart
    """
    vs = PARAMS["voxel_size"]
    std = 2.0 * vs
    cases = []
    for kind in ("same", "disjoint", "mixed"):
        npts = int(rng.integers(200, 320))
        coord = rng.normal(0, std, size=(npts, 3)).astype(np.float64)
        S = 2
        mv = np.full((npts, S), -1, dtype=np.int64)
        half = coord[:, 0] >= np.median(coord[:, 0])  # split into two adjacent members
        members = [np.nonzero(~half)[0], np.nonzero(half)[0]]
        if kind == "same":
            mv[:, 0] = 100
            mv[:, 1] = 101  # both halves share the object's per-view mask ids
            should = True
        elif kind == "disjoint":
            mv[~half, 0] = 100
            mv[~half, 1] = 101
            mv[half, 0] = 200  # the other half sits in DIFFERENT masks
            mv[half, 1] = 201
            should = False
        else:  # mixed: half of the 2nd member disagrees -> conflict ~0.5 > thresh
            mv[:, 0] = 100
            mv[:, 1] = 101
            hidx = members[1]
            flip = hidx[: hidx.size // 2]
            mv[flip, 0] = 300
            mv[flip, 1] = 301
            should = False
        # small -1 noise
        mv[rng.random((npts, S)) < 0.1] = -1
        cases.append((members, coord, mv, should))
    return cases


def detour_case():
    """Structured strict-refine case: two mask groups (A,B) in ONE view bridged by a
    chain of UNLABELED (-1) points. Non-strict CC keeps them one component (legal -1
    edges bridge); strict MUST split into 2. Verifies the transitive-closure logic."""
    vs = PARAMS["voxel_size"]
    # chain of 9 points along x, ends labeled A / B, middle unlabeled
    n = 9
    coord = np.stack([np.arange(n) * 1.5 * vs, np.zeros(n), np.zeros(n)], 1).astype(np.float64)
    mv = np.full((n, 1), -1, dtype=np.int64)
    mv[0, 0] = 10  # A
    mv[1, 0] = 10
    mv[n - 1, 0] = 20  # B
    mv[n - 2, 0] = 20
    pairs = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)  # legal chain
    base = np.zeros(n, dtype=np.int64)  # all one component
    return base, pairs, mv


# --------------------------------------------------------------------------- #
# Part A
# --------------------------------------------------------------------------- #
def part_a(args):
    print("=" * 78)
    print("PART A -- stage-level differential (CPU): standalone ports vs model methods")
    print("=" * 78)
    obj = make_reference()
    rng = np.random.default_rng(args.seed)
    n_cases = args.n_cases

    results = []  # (stage, case, ok, diag)
    n_split_events = n_merge_events = 0

    geo_radius = PARAMS["cluster_thresh"] * PARAMS["voxel_size"]

    for c in range(n_cases):
        coord, mv = random_scene(rng)
        n = coord.shape[0]

        # ---- cannot-link split: identical input members fed to both ----
        # min_points=1 keeps every geometric component so the split logic (not the size
        # gate) is what we compare.
        members = STA._geometric_components(coord, geo_radius, min_points=1)
        la = ref_split(obj, members, coord, mv, n)
        lb = sta_split(members, coord, mv, n)
        ok, diag = partitions_equal(la, lb)
        results.append(("cannot-link", c, ok, diag))
        # did the split actually fire? (input 1 blob-ish -> >1 sub-instances)
        if diag.get("n_instances_a", 0) > max(1, len(members)):
            n_split_events += 1

        # ---- RAG merge: feed the (agreed) over-segmentation to both ----
        merge_in = STA._split_by_mask_multiview(
            members, coord, mv, PARAMS["mask_split_radius"] * PARAMS["voxel_size"],
            strict=True, min_points=1)
        if len(merge_in) >= 2:
            ma = ref_merge(obj, merge_in, coord, mv, n)
            mb = sta_merge(merge_in, coord, mv, n)
            ok2, diag2 = partitions_equal(ma, mb)
            results.append(("rag-merge", c, ok2, diag2))
            if diag2.get("n_instances_a", 0) < len(merge_in):
                n_merge_events += 1

        # ---- strict refine on the largest proposal's local graph ----
        if members:
            idx = max(members, key=len)
            pts = coord[idx]
            mvl = mv[idx]
            from scipy.spatial import cKDTree
            allp = cKDTree(pts).query_pairs(
                PARAMS["mask_split_radius"] * PARAMS["voxel_size"], output_type="ndarray")
            if allp.shape[0]:
                va, vb = mvl[allp[:, 0]], mvl[allp[:, 1]]
                cannot = ((va >= 0) & (vb >= 0) & (va != vb)).any(axis=1)
                pl = allp[~cannot]
            else:
                pl = allp
            base = STA._radius_components(len(idx), pl)
            sa = ref_strict(base, pl, mvl)
            sb = sta_strict(base, pl, mvl)
            ok3, diag3 = partitions_equal(sa, sb)
            results.append(("strict-refine", c, ok3, diag3))

    # ---- structured merge-firing cases (make the RAG-merge comparison non-vacuous) ----
    merge_struct_ok = True
    merge_fired_struct = 0
    merge_expect_match = True
    for mi, (members, coord, mv, should) in enumerate(merge_firing_cases(rng)):
        n = coord.shape[0]
        ma = ref_merge(obj, members, coord, mv, n)
        mb = sta_merge(members, coord, mv, n)
        ok_m, diag_m = partitions_equal(ma, mb)
        results.append(("rag-merge-struct", mi, ok_m, diag_m))
        merge_struct_ok = merge_struct_ok and ok_m
        merged_here = diag_m.get("n_instances_b", 99) < len(members)
        if merged_here:
            merge_fired_struct += 1
        # sanity: the port should also MAKE the physically-expected decision
        if should != merged_here:
            merge_expect_match = False

    # ---- structured detour case for strict refine ----
    base, pairs, mv = detour_case()
    sa = ref_strict(base, pairs, mv)
    sb = sta_strict(base, pairs, mv)
    ok_d, diag_d = partitions_equal(sa, sb)
    results.append(("strict-detour", "struct", ok_d, diag_d))
    n_groups = len(set(sb[sb >= 0].tolist()))
    detour_split = n_groups == 2  # strict must yield exactly 2 groups here

    # ---- report ----
    by_stage = {}
    for stage, _, ok, _ in results:
        s = by_stage.setdefault(stage, [0, 0])
        s[0] += int(ok)
        s[1] += 1
    print(f"\n  cases={n_cases}  (split fired in {n_split_events} random cases)  |  "
          f"strict-detour split-into-2 = {detour_split}")
    print(f"  merge-struct: fired in {merge_fired_struct}/3 cases, "
          f"physical-decision match = {merge_expect_match}")
    print("  stage                 pass/total")
    all_ok = True
    for stage, (p, t) in by_stage.items():
        print(f"    {stage:20s}  {p}/{t}")
        all_ok = all_ok and p == t
    fails = [(stage, case, diag) for stage, case, ok, diag in results if not ok]
    if fails:
        print(f"\n  !! {len(fails)} MISMATCH(es):")
        for stage, case, diag in fails[:10]:
            print(f"     [{stage} case={case}] {diag}")
    verdict = (all_ok and detour_split and merge_struct_ok
               and merge_fired_struct >= 1 and merge_expect_match)
    print(f"\n  PART A VERDICT: {'PASS' if verdict else 'FAIL'}")
    return verdict


# --------------------------------------------------------------------------- #
# Part B
# --------------------------------------------------------------------------- #
def part_b(args):
    print("=" * 78)
    print("PART B -- real-scene integration (GPU, fp32): config _cluster (neutralized) "
          "vs standalone")
    print("=" * 78)
    import torch
    import cluster_labeled_scenes as CLS
    import infer_insseg as ifs
    from engines.hooks.insseg_viz import assign_instance_ids
    from utils.config import Config
    from utils.env import set_seed
    from models import build_model

    set_seed(0)
    cfg = Config.fromfile(args.config_file)
    model = build_model(cfg.model)
    model = ifs.load_weight(model, args.weight)
    model = model.cuda().eval()  # fp32
    base_voxel = float(model.voxel_size)
    transform = ifs.build_infer_transform(grid_size=base_voxel)
    fa = CLS.fuse_args(base_voxel, conf_percentile=0.0, conf_threshold=0.0,
                       valid_mask="filtered")
    cluster_cfg = os.path.join(args.clustering_dir, "embed-maskclust-strong-rag-sv.py")

    scenes = args.scenes or CLS.discover_scenes(args.root)
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    print(f"  {len(scenes)} scenes | config={os.path.basename(cluster_cfg)}\n")

    rows = []
    for rel in scenes:
        seq_dir = os.path.join(args.root, rel)
        aligned_seq_dir = os.path.join(args.aligned_root, rel)
        try:
            scene = CLS.build_labeled_scene(seq_dir, aligned_seq_dir, fa)
            cached = CLS.run_backbone(model, scene, transform, zero_offset=True,
                                      semantic_source="labels")
            # neutralize: apply config, then embedding OFF + uniform one-hot semantics
            ifs.apply_cluster_config(model, cluster_cfg)
            model.instance_embedding = False

            seg_grid = cached["seg_grid"].numpy()  # given per-grid label
            ignore = list(model.segment_ignore_index)
            subset = ~np.isin(seg_grid, ignore)
            sub_idx = np.nonzero(subset)[0]

            n_grid = int(cached["coord"].shape[0])
            n_cls = model.seg_head.out_features
            dev = cached["logit"].device
            dt = cached["logit"].dtype
            uni = torch.full((n_grid, n_cls), -20.0, device=dev, dtype=dt)
            sub_t = torch.from_numpy(subset).to(dev)
            uni[sub_t, 5] = 20.0   # object (foreground, not ignored) for the subset
            uni[~sub_t, 0] = 20.0  # background (ignored) for the rest
            cached["logit"] = uni

            out = CLS.cluster_one(model, cached)
            pm = out["pred_masks"]
            if pm.shape[0] > 0:
                pm_np = pm.bool().numpy()
                ps_np = out["pred_scores"].numpy()
            else:
                pm_np = np.zeros((0, n_grid), dtype=bool)
                ps_np = np.zeros(0, dtype=np.float32)
            cfg_inst = assign_instance_ids(n_grid, pm_np, ps_np)

            # standalone on the SAME grid, object subset, voxelize=False
            grid_coord = cached["coord"].detach().cpu().numpy()
            mpv_t = cached["point_mask_views"]
            grid_mpv = mpv_t.numpy() if mpv_t is not None else None
            sta = STA.cluster_object_points(
                grid_coord[sub_idx],
                grid_mpv[sub_idx] if grid_mpv is not None else None,
                voxelize=False, verbose=False,
                voxel_size=model.voxel_size,
                cluster_thresh=model.cluster_thresh,
                cluster_min_points=model.cluster_min_points,
                cluster_propose_points=model.cluster_propose_points,
                mask_split_radius=model.mask_split_radius,
                mask_constraint_strict=model.mask_constraint_strict,
                rag_merge=bool(model.cluster_mask_rag_merge),
                rag_merge_thresh=model.rag_merge_thresh,
                rag_adjacency_radius=model.rag_adjacency_radius)
            sta_full = np.full(n_grid, -1, dtype=np.int64)
            sta_full[sub_idx] = sta

            ok, diag = partitions_equal(cfg_inst[sub_idx], sta_full[sub_idx])
            has_mask = grid_mpv is not None
            rows.append((rel, ok, diag, has_mask))
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {rel:38s} obj={sub_idx.size:6d} "
                  f"inst cfg={diag['n_instances_a']:3d}/sta={diag['n_instances_b']:3d} "
                  f"{'' if has_mask else '(geom-only, no masks)'}"
                  f"{'' if ok else '  <-- ' + str(diag)}")
        except Exception as e:  # noqa: BLE001
            rows.append((rel, False, {"error": f"{type(e).__name__}: {e}"}, None))
            print(f"  [ERR ] {rel}: {type(e).__name__}: {e}")
            traceback.print_exc()

    n_pass = sum(1 for _, ok, _, _ in rows if ok)
    print(f"\n  PART B: {n_pass}/{len(rows)} scenes partition-equal")
    verdict = n_pass == len(rows) and len(rows) > 0
    print(f"  PART B VERDICT: {'PASS' if verdict else 'FAIL'}")
    return verdict


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["A", "B", "all"], default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-cases", type=int, default=60, help="Part A random cases")
    ap.add_argument("--root", default=CLS_ROOT)
    ap.add_argument("--aligned-root", default=CLS_ALIGNED)
    ap.add_argument("--config-file", default=CLS_CONFIG)
    ap.add_argument("--weight", default=CLS_WEIGHT)
    ap.add_argument("--clustering-dir", default=CLS_CLUSTERING_DIR)
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--limit-scenes", type=int, default=0)
    args = ap.parse_args()

    verdicts = {}
    if args.part in ("A", "all"):
        verdicts["A"] = part_a(args)
    if args.part in ("B", "all"):
        verdicts["B"] = part_b(args)

    print("\n" + "=" * 78)
    print("OVERALL:", "  ".join(f"Part {k}={'PASS' if v else 'FAIL'}"
                                 for k, v in verdicts.items()))
    print("=" * 78)
    sys.exit(0 if all(verdicts.values()) else 1)


# defaults mirrored from cluster_labeled_scenes (import kept lazy for Part-A-only runs)
CLS_ROOT = "/home/fai/workspace/jhp/dataset/val_sample_output"
CLS_ALIGNED = "/home/fai/workspace/jhp/dataset/val_sample_output_aligned"
CLS_CONFIG = os.path.join(
    _REPO_ROOT, "exp", "scannet-v1.1.1-2of3",
    "insseg-litept-small-v1m2-2of3-embed", "config.py")
CLS_WEIGHT = os.path.join(
    _REPO_ROOT, "exp", "scannet-v1.1.1-2of3",
    "insseg-litept-small-v1m2-2of3-embed", "model", "model_best.pth")
CLS_CLUSTERING_DIR = os.path.join(
    _REPO_ROOT, "configs", "scannet-v1.1.1-2of3", "clustering")


if __name__ == "__main__":
    main()
