from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pointgroup_ops import ballquery_batch_p, bfs_cluster
except ImportError:
    ballquery_batch_p, bfs_cluster = None, None

from models.utils import offset2batch, batch2offset
from models.utils.structure import Point

from models.builder import MODELS, build_model
from models.losses import build_criteria


@MODELS.register_module("PG-v1m2")
class PointGroup(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        cluster_thresh=1.5,
        cluster_closed_points=300,
        cluster_propose_points=100,
        cluster_min_points=50,
        voxel_size=0.02,
        instance_embedding=False,
        embedding_dim=5,
        embed_delta_v=0.5,
        embed_delta_d=1.5,
        embed_loss_weight=1.0,
        embed_reg_weight=0.001,
        embed_bandwidth=1.5,
        embed_min_points=50,
        cluster_mask_constraint=False,
        mask_split_radius=2.5,
        mask_constraint_multiview=False,
        mask_constraint_strict=False,
        cluster_mask_filter=False,
        cluster_mask_rag_merge=False,
        rag_merge_thresh=0.5,
        rag_adjacency_radius=2.5,
        rag_sim_metric="intersection",
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.semantic_num_classes = semantic_num_classes
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index
        self.cluster_thresh = cluster_thresh
        self.cluster_closed_points = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points = cluster_min_points
        self.voxel_size = voxel_size
        self.backbone = build_model(backbone)
        self.bias_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 3),
        )
        self.seg_head = nn.Linear(backbone_out_channels, semantic_num_classes)
        # Optional instance-embedding head (disabled by default so existing configs
        # train bit-identically). When enabled it adds a per-point embedding trained
        # with a discriminative pull/push loss to separate touching same-class
        # instances that offset+BFS clustering would otherwise merge.
        self.instance_embedding = instance_embedding
        self.embedding_dim = embedding_dim
        self.embed_delta_v = embed_delta_v
        self.embed_delta_d = embed_delta_d
        self.embed_loss_weight = embed_loss_weight
        self.embed_reg_weight = embed_reg_weight
        self.embed_bandwidth = embed_bandwidth
        self.embed_min_points = embed_min_points
        # Optional inference-only 2D-mask cannot-link constraint. When per-point mask
        # labels are supplied to _cluster (the GT-free VGGT path, tools/infer_insseg.py),
        # each proposal is additionally cut so two points from the SAME view but in
        # DIFFERENT 2D masks never share an instance. Graceful no-op without labels.
        self.cluster_mask_constraint = cluster_mask_constraint
        self.mask_split_radius = mask_split_radius
        # When enabled (and per-point per-view labels are supplied to _cluster as
        # point_mask_views (N, S)), the cannot-link is generalized across ALL views:
        # two points are separated if they fall in different 2D masks in ANY shared
        # view (see _split_proposals_by_mask_multiview), not just their single origin
        # view. Falls back to the single-view path when only point_view/point_mask
        # (1-D) are supplied. Graceful no-op without any labels.
        self.mask_constraint_multiview = mask_constraint_multiview
        # When enabled, the 2D-mask cannot-link is enforced TRANSITIVELY (a "strong" split):
        # the default path only deletes the DIRECT edge between two distinct-mask/same-view
        # neighbours and then takes transitive connected components, so two such points can
        # still re-merge through a detour of legal edges (e.g. an unlabeled seam point that
        # is mask==-1 in the conflicting view bridges them). With this on, each leaked
        # component is re-split by constrained region-growing so NO final instance contains
        # any same-view/distinct-mask pair, adjacent or not. Inference-only; no-op without
        # cluster_mask_constraint and per-point mask labels. See _strict_refine.
        self.mask_constraint_strict = mask_constraint_strict
        # Optional inference-only pre-clustering filter. When enabled and per-point 2D-mask
        # labels are supplied to _cluster, points that fall in NO 2D object mask are dropped
        # BEFORE grouping, so only points inside some 2D mask can form instances. Separate
        # from cluster_mask_constraint (cannot-link split) -- they compose but gate
        # independently. Graceful no-op without labels (train/val).
        self.cluster_mask_filter = cluster_mask_filter
        # Optional inference-only POST-clustering merge that HEALS the over-segmentation the
        # 2D-mask cannot-link split can introduce. After the splits, adjacent same-class clusters
        # are re-merged via a region-adjacency graph when their 2D-mask labels agree well enough.
        # The agreement test is selected by rag_sim_metric: legacy GLOBAL histogram similarity
        # ("intersection"/"cosine"), or the shared-view "different-mask" conflict ratio
        # ("shared_view"/"shared_view_dominant") which only compares the views the two clusters
        # share. Because the test aggregates hundreds of points, a single noisy point that
        # originally severed a real object is drowned out by the majority, so the two halves
        # re-merge. Gated by cluster_mask_rag_merge AND per-point mask labels; graceful no-op
        # otherwise (train/val). See _merge_proposals_by_mask_histogram.
        self.cluster_mask_rag_merge = cluster_mask_rag_merge
        # rag_merge_thresh: for legacy "intersection"/"cosine" it is the MIN histogram
        # similarity to merge (higher = stricter); for the shared-view conflict metrics
        # ("shared_view", "shared_view_dominant") it is the MAX different-mask ratio to
        # merge (lower = stricter) -- opposite direction. See _merge_proposals_by_mask_histogram.
        self.rag_merge_thresh = rag_merge_thresh
        self.rag_adjacency_radius = rag_adjacency_radius  # RAG adjacency radius (voxel units)
        # "intersection" (default) | "cosine" | "shared_view" | "shared_view_dominant"
        self.rag_sim_metric = rag_sim_metric
        if self.instance_embedding:
            self.embedding_head = nn.Sequential(
                nn.Linear(backbone_out_channels, backbone_out_channels),
                norm_fn(backbone_out_channels),
                nn.ReLU(),
                nn.Linear(backbone_out_channels, embedding_dim),
            )
        self.seg_criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, data_dict, return_point=False):
        if return_point:
            return dict(point=self.backbone(data_dict))
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        instance_centroid = data_dict["instance_centroid"]
        offset = data_dict["offset"]

        feat = self._backbone_feat(data_dict)
        bias_pred = self.bias_head(feat)
        logit_pred = self.seg_head(feat)
        embedding = self.embedding_head(feat) if self.instance_embedding else None

        # compute loss
        seg_loss = self.seg_criteria(logit_pred, segment)

        mask = (instance != self.instance_ignore_index).float()
        bias_gt = instance_centroid - coord
        bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
        bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

        bias_pred_norm = bias_pred / (
            torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
        )
        bias_gt_norm = bias_gt / (torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8)
        cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
        bias_cosine_loss = torch.sum(cosine_similarity * mask) / (
            torch.sum(mask) + 1e-8
        )

        loss = seg_loss + bias_l1_loss + bias_cosine_loss
        return_dict = dict(
            seg_loss=seg_loss,
            bias_l1_loss=bias_l1_loss,
            bias_cosine_loss=bias_cosine_loss,
        )
        if self.instance_embedding:
            embed_loss, embed_var_loss, embed_dist_loss = self._embedding_loss(
                embedding, instance, offset
            )
            loss = loss + self.embed_loss_weight * embed_loss
            return_dict["embed_loss"] = embed_loss
            return_dict["embed_var_loss"] = embed_var_loss
            return_dict["embed_dist_loss"] = embed_dist_loss
        return_dict["loss"] = loss

        if not self.training:
            return_dict.update(
                self._cluster(coord, bias_pred, logit_pred, offset, embedding)
            )
        return return_dict

    def _backbone_feat(self, data_dict):
        """Run the backbone and return per-point features at full resolution,
        un-pooling parent/child features back up the hierarchy as needed."""
        point = self.backbone(data_dict)
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        return feat

    def _forward_backbone(self, data_dict):
        """Run the backbone + bias/seg heads, returning (bias_pred, logit_pred).

        Split out from forward() so callers (e.g. tools/reeval_insseg_cluster_sweep.py)
        can cache these per-point predictions once per scene and then re-run only the
        post-processing in self._cluster() across many clustering settings.
        """
        feat = self._backbone_feat(data_dict)
        return self.bias_head(feat), self.seg_head(feat)

    @torch.no_grad()
    def _cluster(
        self,
        coord,
        bias_pred,
        logit_pred,
        offset,
        embedding=None,
        point_view=None,
        point_mask=None,
        point_mask_views=None,
    ):
        """Inference-only post-processing: shift each point by its predicted offset
        toward its instance centroid, group the shifted points with a ball-query +
        BFS connected-component pass (gated by predicted semantic class), then filter
        proposals by size.

        Entirely governed by self.voxel_size, self.cluster_thresh, and
        self.cluster_{closed,min,propose}_points. The effective physical grouping
        radius is cluster_thresh * voxel_size (center_pred is put into voxel units
        before ball-query). Returns dict(pred_scores, pred_masks, pred_classes).
        """
        # Optional, inference-only: collect the grid indices of points whose 2D-mask
        # cannot-link edge actually severed two final sub-instances (the seam between
        # touching objects). Flag-gated and default off, so train/eval and other tools
        # are unaffected; filled by the _split_proposals_by_mask* methods below.
        if getattr(self, "record_split_cause", False):
            self._split_cause_idx = []
        center_pred = coord + bias_pred
        center_pred = center_pred / self.voxel_size
        logit_pred = F.softmax(logit_pred, dim=-1)
        segment_pred = torch.max(logit_pred, 1)[1]  # [n]
        # cluster
        mask = (
            ~torch.concat(
                [
                    (segment_pred == index).unsqueeze(-1)
                    for index in self.segment_ignore_index
                ],
                dim=1,
            )
            .sum(-1)
            .bool()
        )

        # Inference-only pre-clustering filter: when enabled and per-point 2D-mask labels
        # are supplied, drop points that fall in NO 2D object mask before grouping, so only
        # points inside some 2D mask can form instances. Works with either label format
        # (multi-view point_mask_views (N,S) or single-view point_mask (N,)); independent of
        # the cannot-link constraint. No-op without labels (train/val, cluster-sweep tool).
        if self.cluster_mask_filter:
            has_label = None
            if point_mask_views is not None:
                has_label = (point_mask_views >= 0).any(dim=1)
            elif point_mask is not None:
                has_label = point_mask >= 0
            if has_label is not None:
                mask = mask & has_label.to(mask.device)

        if mask.sum() == 0:
            proposals_idx = torch.zeros((0, 2)).int()
            proposals_offset = torch.zeros(1).int()
        else:
            center_pred_ = center_pred[mask]
            segment_pred_ = segment_pred[mask]

            batch_ = offset2batch(offset)[mask]
            offset_ = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
            idx, start_len = ballquery_batch_p(
                center_pred_,
                batch_.int(),
                offset_.int(),
                self.cluster_thresh,
                self.cluster_closed_points,
            )
            proposals_idx, proposals_offset = bfs_cluster(
                segment_pred_.int().cpu(),
                idx.cpu(),
                start_len.cpu(),
                self.cluster_min_points,
            )
            proposals_idx[:, 1] = (
                mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
            )

        # get proposal
        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, center_pred.shape[0]), dtype=torch.int
        )
        proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
        instance_pred = segment_pred[
            proposals_idx[:, 1][proposals_offset[:-1].long()].long()
        ]
        # Sub-divide each geometric proposal in embedding space to recover touching
        # same-class instances that ball-query+BFS merged. No-op when the embedding
        # head is disabled or embedding is not supplied (e.g. the cluster-sweep tool).
        if (
            self.instance_embedding
            and embedding is not None
            and proposals_pred.shape[0] > 0
        ):
            proposals_pred, instance_pred = self._split_proposals_by_embedding(
                proposals_pred, instance_pred, embedding
            )
        # Further sub-divide each proposal so that points falling in DIFFERENT 2D masks
        # of the SAME view are separated (cannot-link). Composes after the embedding
        # split. No-op unless enabled and per-point mask labels are supplied.
        #
        # Multi-view path (preferred when enabled): point_mask_views (N, S) carries the
        # mask each point lands in for EVERY view it is visible in, so the cannot-link
        # fires whenever two points disagree in ANY shared view. Single-view path
        # (legacy): point_view/point_mask (N,) only knows each point's origin view.
        if (
            self.cluster_mask_constraint
            and self.mask_constraint_multiview
            and point_mask_views is not None
            and proposals_pred.shape[0] > 0
        ):
            proposals_pred, instance_pred = self._split_proposals_by_mask_multiview(
                proposals_pred, instance_pred, coord, point_mask_views
            )
        elif (
            self.cluster_mask_constraint
            and point_mask is not None
            and point_view is not None
            and proposals_pred.shape[0] > 0
        ):
            proposals_pred, instance_pred = self._split_proposals_by_mask(
                proposals_pred, instance_pred, coord, point_view, point_mask
            )
        # Post-clustering RAG merge: re-merge adjacent same-class sub-instances whose 2D-mask
        # histograms are highly similar, healing the over-segmentation the cannot-link split can
        # introduce (a single noisy point can otherwise sever a real object). Runs BEFORE the
        # cluster_propose_points size filter below so re-merged fragments can clear the size gate.
        # Needs the multi-view labels (point_mask_views, N x S); no-op otherwise.
        if (
            self.cluster_mask_rag_merge
            and point_mask_views is not None
            and proposals_pred.shape[0] > 1
        ):
            proposals_pred, instance_pred = self._merge_proposals_by_mask_histogram(
                proposals_pred, instance_pred, coord, point_mask_views
            )
        proposals_point_num = proposals_pred.sum(1)
        proposals_mask = proposals_point_num > self.cluster_propose_points
        proposals_pred = proposals_pred[proposals_mask]
        instance_pred = instance_pred[proposals_mask]

        pred_scores = []
        pred_classes = []
        pred_masks = proposals_pred.detach().cpu()
        for proposal_id in range(len(proposals_pred)):
            segment_ = proposals_pred[proposal_id]
            confidence_ = logit_pred[
                segment_.bool(), instance_pred[proposal_id]
            ].mean()
            object_ = instance_pred[proposal_id]
            pred_scores.append(confidence_)
            pred_classes.append(object_)
        if len(pred_scores) > 0:
            pred_scores = torch.stack(pred_scores).cpu()
            pred_classes = torch.stack(pred_classes).cpu()
        else:
            pred_scores = torch.tensor([])
            pred_classes = torch.tensor([])

        return dict(
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            pred_classes=pred_classes,
        )

    def _embedding_loss(self, embedding, instance, offset):
        """Discriminative (pull/push/reg) loss of De Brabandere et al., per scene.

        Learns a per-point embedding where points of the same instance collapse
        toward a shared mean (pull, within delta_v) and distinct instance means
        stay >= 2*delta_d apart (push). The push term is exactly the pairwise
        separability signal the offset+BFS grouping lacks for touching same-class
        objects. Computed per-scene (split by ``offset``, since instance ids
        restart per scene) over foreground points (instance != ignore index).

        Returns (total, var_loss, dist_loss) so the pull/push terms can be logged.
        """
        delta_v = self.embed_delta_v
        delta_d = self.embed_delta_d
        var_terms, dist_terms, reg_terms = [], [], []
        start = 0
        for b in range(offset.shape[0]):
            end = int(offset[b])
            emb = embedding[start:end]
            inst = instance[start:end]
            start = end
            valid = inst != self.instance_ignore_index
            emb = emb[valid]
            inst = inst[valid]
            if emb.shape[0] == 0:
                continue
            uids = torch.unique(inst)
            means = []
            var = emb.new_zeros(())
            for uid in uids:
                e = emb[inst == uid]
                mu = e.mean(0)
                means.append(mu)
                d = torch.norm(e - mu, p=2, dim=1)
                var = var + torch.clamp(d - delta_v, min=0).pow(2).mean()
            num_inst = len(means)
            var = var / num_inst
            means = torch.stack(means, 0)
            reg = torch.norm(means, p=2, dim=1).mean()
            if num_inst > 1:
                pair = torch.cdist(means, means)
                margin = torch.clamp(2 * delta_d - pair, min=0).pow(2)
                # zero the diagonal (self-pairs would otherwise add (2*delta_d)^2)
                margin = margin * (
                    1.0 - torch.eye(num_inst, device=margin.device, dtype=margin.dtype)
                )
                dist = margin.sum() / (num_inst * (num_inst - 1))
            else:
                dist = emb.new_zeros(())
            var_terms.append(var)
            dist_terms.append(dist)
            reg_terms.append(reg)
        if len(var_terms) == 0:
            # keep the embedding head in the autograd graph even with no valid instances
            z = embedding.sum() * 0.0
            return z, z, z
        var_loss = torch.stack(var_terms).mean()
        dist_loss = torch.stack(dist_terms).mean()
        reg_loss = torch.stack(reg_terms).mean()
        total = var_loss + dist_loss + self.embed_reg_weight * reg_loss
        return total, var_loss, dist_loss

    @torch.no_grad()
    def _meanshift_cluster(self, emb, bandwidth, max_iter=10):
        """Greedy mode-seeking clustering of points in embedding space.

        Training keeps points within delta_v of their instance mean and means
        >= 2*delta_d apart, so mean-shift with bandwidth ~ delta_d converges from
        any seed to its own instance mode -- the arbitrary seed choice is robust.
        Returns a (M,) long tensor of sub-cluster ids.
        """
        num_pts = emb.shape[0]
        device = emb.device
        labels = torch.full((num_pts,), -1, dtype=torch.long, device=device)
        unassigned = torch.ones(num_pts, dtype=torch.bool, device=device)
        cluster_id = 0
        while bool(unassigned.any()) and cluster_id < num_pts:
            seed_idx = int(torch.nonzero(unassigned, as_tuple=False).view(-1)[0])
            center = emb[seed_idx]
            for _ in range(max_iter):
                within = (torch.norm(emb - center, dim=1) < bandwidth) & unassigned
                if not bool(within.any()):
                    break
                new_center = emb[within].mean(0)
                if float(torch.norm(new_center - center)) < 1e-4:
                    center = new_center
                    break
                center = new_center
            member = (torch.norm(emb - center, dim=1) < bandwidth) & unassigned
            if not bool(member.any()):
                member = torch.zeros(num_pts, dtype=torch.bool, device=device)
                member[seed_idx] = True
            labels[member] = cluster_id
            unassigned = unassigned & (~member)
            cluster_id += 1
        return labels

    @torch.no_grad()
    def _split_proposals_by_embedding(self, proposals_pred, instance_pred, embedding):
        """Sub-divide each geometric proposal by clustering its points in embedding
        space, separating touching same-class instances merged by offset+BFS. A
        single-instance proposal yields one sub-cluster and is returned unchanged.
        Sub-clusters smaller than embed_min_points are dropped as noise.
        """
        num_points = proposals_pred.shape[1]
        new_masks, new_classes = [], []
        for p in range(proposals_pred.shape[0]):
            idx = torch.nonzero(proposals_pred[p], as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            sub = self._meanshift_cluster(
                embedding[idx.to(embedding.device)], self.embed_bandwidth
            ).cpu()
            for c in torch.unique(sub):
                m = sub == c
                if int(m.sum()) < self.embed_min_points:
                    continue
                full = torch.zeros(num_points, dtype=proposals_pred.dtype)
                full[idx[m]] = 1
                new_masks.append(full)
                new_classes.append(instance_pred[p])
        if len(new_masks) == 0:
            return (
                torch.zeros((0, num_points), dtype=proposals_pred.dtype),
                instance_pred[:0],
            )
        return torch.stack(new_masks, 0), torch.stack(new_classes, 0)

    @torch.no_grad()
    def _split_proposals_by_mask(
        self, proposals_pred, instance_pred, coord, point_view, point_mask
    ):
        """Sub-divide each geometric proposal so that two points coming from the SAME
        camera view but lying in DIFFERENT 2D instance masks are never grouped together
        (a cannot-link constraint from the D-FINE-seg 2D masks). Points from different
        views, or with no/identical mask, may still group by spatial adjacency, so an
        object spanning several views is not shattered.

        Within each proposal we build a radius graph on ``coord`` (physical radius
        ``mask_split_radius * voxel_size`` -- the same scale BFS used to merge), drop any
        adjacent pair (u, v) with ``point_view[u] == point_view[v] and
        point_mask[u] != point_mask[v]`` (both labels valid, i.e. >= 0), and take the
        connected components of what remains. Components >= ``cluster_min_points`` become
        sub-instances inheriting the parent's class; a proposal with no cut is returned
        unchanged. No-op (returns inputs) if SciPy is unavailable.

        Component labelling uses scipy's C ``connected_components`` rather than a Python
        union-find loop, so a single dense proposal yielding millions of adjacency edges
        (e.g. a flat table/floor slab) stays sub-second instead of stalling for minutes.
        """
        try:
            from scipy.spatial import cKDTree
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
        except ImportError:
            return proposals_pred, instance_pred

        num_points = proposals_pred.shape[1]
        coord_np = coord.detach().cpu().numpy()
        view_np = point_view.detach().cpu().numpy()
        mask_np = point_mask.detach().cpu().numpy()
        radius = float(self.mask_split_radius * self.voxel_size)

        new_masks, new_classes = [], []
        for p in range(proposals_pred.shape[0]):
            idx = torch.nonzero(proposals_pred[p], as_tuple=False).view(-1).cpu().numpy()
            n = idx.shape[0]
            if n == 0:
                continue
            pts = coord_np[idx]
            v = view_np[idx]
            m = mask_np[idx]
            all_pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")  # (M,2)
            cannot_pairs = all_pairs[:0]
            pairs = all_pairs
            if all_pairs.shape[0]:
                a, b = all_pairs[:, 0], all_pairs[:, 1]
                # cannot-link: same view, different (valid) mask -> drop that edge
                cannot = (v[a] == v[b]) & (m[a] != m[b]) & (m[a] >= 0) & (m[b] >= 0)
                cannot_pairs = all_pairs[cannot]
                pairs = all_pairs[~cannot]
            labels = self._radius_components(n, pairs, connected_components, coo_matrix)
            if self.mask_constraint_strict:
                # transitively-closed cannot-link: re-split any component that re-merged
                # two distinct same-view masks through a detour of legal edges
                labels = self._strict_refine(labels, pairs, self._views_to_mv(v, m))
            # record the seam: cannot-link endpoints that ended in DIFFERENT final
            # sub-instances (grid indices). No-op unless recording is enabled.
            if getattr(self, "record_split_cause", False) and cannot_pairs.shape[0]:
                ca, cb = cannot_pairs[:, 0], cannot_pairs[:, 1]
                cut = labels[ca] != labels[cb]
                if cut.any():
                    self._split_cause_idx.append(
                        idx[np.unique(np.concatenate([ca[cut], cb[cut]]))])
            for r in np.unique(labels):
                member = labels == r
                if int(member.sum()) < self.cluster_min_points:
                    continue
                full = torch.zeros(num_points, dtype=proposals_pred.dtype)
                full[torch.from_numpy(idx[member]).long()] = 1
                new_masks.append(full)
                new_classes.append(instance_pred[p])
        if len(new_masks) == 0:
            return (
                torch.zeros((0, num_points), dtype=proposals_pred.dtype),
                instance_pred[:0],
            )
        return torch.stack(new_masks, 0), torch.stack(new_classes, 0)

    @staticmethod
    def _radius_components(n, pairs, connected_components, coo_matrix):
        """Connected-component labels (n,) for n nodes joined by the undirected edges in
        ``pairs`` ((M,2) local indices). Isolated nodes (no surviving edge) are their own
        component. Uses scipy's C ``connected_components`` -- no Python loop."""
        if pairs.shape[0] == 0:
            return np.arange(n)
        data = np.ones(pairs.shape[0], dtype=np.uint8)
        g = coo_matrix((data, (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(g, directed=False)
        return labels

    @staticmethod
    def _views_to_mv(v, m):
        """Encode single-view (origin view ``v`` (n,), mask ``m`` (n,)) per-point labels
        as the (n, S) per-view mask-id array consumed by ``_strict_refine``: column c holds
        the mask id of the c-th distinct labeled view, or -1. Each point is labeled in at
        most its origin view, so every row has <= 1 non-negative entry -- the single-view
        cannot-link (same view, different mask) is then exactly the generic per-view
        conflict ``_strict_refine`` already detects, so both split paths share one engine."""
        n = v.shape[0]
        valid = m >= 0
        if not valid.any():
            return np.full((n, 1), -1, dtype=np.int64)
        uniq_views = np.unique(v[valid])
        view_to_col = {int(vv): c for c, vv in enumerate(uniq_views)}
        mv = np.full((n, len(uniq_views)), -1, dtype=np.int64)
        vi = np.nonzero(valid)[0]
        cols = np.fromiter((view_to_col[int(vv)] for vv in v[vi]), dtype=np.int64,
                           count=vi.shape[0])
        mv[vi, cols] = m[vi]
        return mv

    @staticmethod
    def _strict_refine(base, pairs, mv):
        """Refine transitive connected-component labels ``base`` (n,) into a
        TRANSITIVELY-CLOSED cannot-link partition over per-node per-view mask ids ``mv``
        (n, S): no output component may hold two distinct valid mask ids in any single
        view. This closes the detour leak of the default path -- which deletes only the
        direct distinct-mask/same-view edge and then takes (transitive) connected
        components, so two such points re-merge whenever a path of legal edges (e.g. an
        unlabeled mask==-1 seam point) bridges them.

        Fast path / slow path: components already mask-consistent (every view carries <= 1
        valid mask across their members) are returned unchanged -- a dense single-object or
        all-unlabeled slab pays only the vectorized conflict scan, never the Python walk,
        preserving the sub-second behaviour ``_radius_components`` was written for. Only a
        component that actually leaked two masks is re-split by constrained region-growing:
        a node joins a growing sub-cluster iff it conflicts with NO member already in it
        (the per-cluster ``view -> mask_id`` map stays single-valued, so every member pair
        agrees in every shared view -> guarantee holds regardless of growth order). The
        adjacency walk is built ONLY over conflicted nodes. ``pairs`` are (M,2) local
        indices (already cannot-link-filtered upstream). Returns contiguous labels (n,)."""
        n = base.shape[0]
        S = mv.shape[1]
        # --- vectorized per-view conflict detection: which base components carry >=2
        #     distinct valid mask ids in some single view ---
        bad_comps = set()
        for s in range(S):
            col = mv[:, s]
            valid = col >= 0
            if not valid.any():
                continue
            comp_v = base[valid]
            mask_v = col[valid]
            order = np.lexsort((mask_v, comp_v))
            cs = comp_v[order]
            ms = mask_v[order]
            # within a run of equal comp id, a mask id change => that comp mixes >=2 masks
            change = np.zeros(cs.shape[0], dtype=bool)
            change[1:] = (cs[1:] == cs[:-1]) & (ms[1:] != ms[:-1])
            for c in np.unique(cs[change]):
                bad_comps.add(int(c))
        if not bad_comps:
            # every component is mask-consistent: keep the grouping, compactify labels
            _, inv = np.unique(base, return_inverse=True)
            return inv.astype(np.int64)

        conflicted = np.isin(base, list(bad_comps))
        out = np.full(n, -1, dtype=np.int64)
        keep = ~conflicted
        next_label = 0
        if keep.any():
            _, inv = np.unique(base[keep], return_inverse=True)
            out[keep] = inv
            next_label = int(inv.max()) + 1

        # --- constrained region-growing over conflicted nodes only ---
        conf_nodes = np.nonzero(conflicted)[0]
        adj = {int(i): [] for i in conf_nodes}
        if pairs.shape[0]:
            em = conflicted[pairs[:, 0]] & conflicted[pairs[:, 1]]
            for a, b in pairs[em]:
                a = int(a)
                b = int(b)
                adj[a].append(b)
                adj[b].append(a)
        seen = {}
        for seed in conf_nodes:
            seed = int(seed)
            if seed in seen:
                continue
            cluster_mask = {}  # view -> the single valid mask id this sub-cluster occupies
            for s in np.nonzero(mv[seed] >= 0)[0]:
                cluster_mask[int(s)] = int(mv[seed, s])
            seen[seed] = next_label
            stack = [seed]
            while stack:
                u = stack.pop()
                for w in adj[u]:
                    if w in seen:
                        continue
                    wv = mv[w]
                    ok = True
                    for s in np.nonzero(wv >= 0)[0]:
                        cur = cluster_mask.get(int(s))
                        if cur is not None and cur != int(wv[s]):
                            ok = False
                            break
                    if not ok:
                        continue
                    seen[w] = next_label
                    for s in np.nonzero(wv >= 0)[0]:
                        cluster_mask[int(s)] = int(wv[s])
                    stack.append(w)
            next_label += 1
        for node, lab in seen.items():
            out[node] = lab
        return out

    @torch.no_grad()
    def _split_proposals_by_mask_multiview(
        self, proposals_pred, instance_pred, coord, point_mask_views
    ):
        """Multi-view generalization of ``_split_proposals_by_mask``.

        ``point_mask_views`` is (N, S) int: column s holds the globally-unique 2D-mask
        id point i lands in when reprojected into view s (z-buffer-occluded), or -1 if
        not visible / not covered there (see tools/vggt_to_scene.py --mask-all-views).
        Mask ids are unique per (view, instance), so two points fall in DIFFERENT masks
        of the SAME view iff some column s holds two distinct non-negative values.

        Within each proposal we build the same radius graph on ``coord`` (physical radius
        ``mask_split_radius * voxel_size``), drop any adjacent pair (u, v) for which
        ``any(mv[u] >= 0 and mv[v] >= 0 and mv[u] != mv[v])`` holds across the S view
        columns (a cannot-link in ANY shared view), and take the connected components of
        what remains. Points that share no labeled view, or agree in every shared view,
        still group by adjacency, so an object spanning views is not shattered. Components
        >= ``cluster_min_points`` become sub-instances inheriting the parent's class; a
        proposal with no cut is returned unchanged. No-op if SciPy is unavailable.

        Like ``_split_proposals_by_mask``, component labelling uses scipy's C
        ``connected_components`` (no Python loop), so dense proposals stay fast.
        """
        try:
            from scipy.spatial import cKDTree
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
        except ImportError:
            return proposals_pred, instance_pred

        num_points = proposals_pred.shape[1]
        coord_np = coord.detach().cpu().numpy()
        mv_np = point_mask_views.detach().cpu().numpy()
        radius = float(self.mask_split_radius * self.voxel_size)

        new_masks, new_classes = [], []
        for p in range(proposals_pred.shape[0]):
            idx = torch.nonzero(proposals_pred[p], as_tuple=False).view(-1).cpu().numpy()
            n = idx.shape[0]
            if n == 0:
                continue
            pts = coord_np[idx]
            mv = mv_np[idx]  # (n, S)
            all_pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")  # (M,2)
            cannot_pairs = all_pairs[:0]
            pairs = all_pairs
            if all_pairs.shape[0]:
                va, vb = mv[all_pairs[:, 0]], mv[all_pairs[:, 1]]  # (M,S) each
                # cannot-link: distinct (valid) masks in some shared view -> drop edge
                cannot = ((va >= 0) & (vb >= 0) & (va != vb)).any(axis=1)
                cannot_pairs = all_pairs[cannot]
                pairs = all_pairs[~cannot]
            labels = self._radius_components(n, pairs, connected_components, coo_matrix)
            if self.mask_constraint_strict:
                # transitively-closed cannot-link: re-split any component that re-merged
                # two distinct masks (in some shared view) through a detour of legal edges
                labels = self._strict_refine(labels, pairs, mv)
            # record the seam: cannot-link endpoints that ended in DIFFERENT final
            # sub-instances (grid indices). No-op unless recording is enabled.
            if getattr(self, "record_split_cause", False) and cannot_pairs.shape[0]:
                ca, cb = cannot_pairs[:, 0], cannot_pairs[:, 1]
                cut = labels[ca] != labels[cb]
                if cut.any():
                    self._split_cause_idx.append(
                        idx[np.unique(np.concatenate([ca[cut], cb[cut]]))])
            for r in np.unique(labels):
                member = labels == r
                if int(member.sum()) < self.cluster_min_points:
                    continue
                full = torch.zeros(num_points, dtype=proposals_pred.dtype)
                full[torch.from_numpy(idx[member]).long()] = 1
                new_masks.append(full)
                new_classes.append(instance_pred[p])
        if len(new_masks) == 0:
            return (
                torch.zeros((0, num_points), dtype=proposals_pred.dtype),
                instance_pred[:0],
            )
        return torch.stack(new_masks, 0), torch.stack(new_classes, 0)

    @torch.no_grad()
    def _merge_proposals_by_mask_histogram(
        self, proposals_pred, instance_pred, coord, point_mask_views
    ):
        """Heal over-segmentation by merging adjacent same-class proposals whose cluster-level
        2D-mask histograms are highly similar -- the inverse of the cannot-link split.

        Each proposal's points are aggregated into a normalized histogram over the global
        2D-mask ids they fall in across ALL views (``point_mask_views`` (N, S); entry >= 0 is a
        globally-unique mask id, -1 = absent). A Region Adjacency Graph is then built: nodes are
        the proposals, and an edge joins two proposals that have points within
        ``rag_adjacency_radius * voxel_size`` of each other -- found with ONE global cKDTree over
        all member points and keeping the cross-owner pairs. An edge becomes a MERGE edge iff the
        two proposals share the predicted class AND pass the ``rag_sim_metric`` agreement test:

          * ``"intersection"`` (default) / ``"cosine"`` -- legacy GLOBAL histogram SIMILARITY of the
            normalized mask histograms; merge when ``sim >= rag_merge_thresh``.
          * ``"shared_view"`` -- restrict to the views the two proposals SHARE (``point_mask_views``
            column = view) and compute the ratio of co-visible point-pairs that land in DIFFERENT
            2D masks: ``conflict = 1 - (cnt_a . cnt_b) / (vc_a . vc_b)`` where ``cnt`` is the count
            histogram over global (view, mask) ids and ``vc[s]`` is the per-view visible count.
            Merge when ``conflict <= rag_merge_thresh``.
          * ``"shared_view_dominant"`` -- coarser per-view vote: ``conflict`` = fraction of shared
            views in which the two proposals' DOMINANT (modal) mask differs; merge when
            ``conflict <= rag_merge_thresh``.

        The shared-view metrics fix the legacy flaw whereby two genuine fragments visible in
        DIFFERENT view sets score as dissimilar (disjoint global histograms) even when they agree
        perfectly in their shared views. For them ``rag_merge_thresh`` is the MAX conflict ratio
        that still merges (lower = stricter), the OPPOSITE direction from the legacy similarity
        metrics; adjacent same-class pairs sharing NO view carry no mask evidence and are not merged.
        Connected components over the merge edges are unioned (proposal masks are disjoint, so a
        clamped sum is the union); the merged instance inherits the class of its largest fragment.
        Proposals in no merge edge pass through unchanged.

        Because the histogram aggregates hundreds of points, a lone noisy point that severed a
        real object is drowned out -> the two halves re-merge, while two genuinely-distinct
        touching objects (different dominant masks) keep dissimilar histograms and stay apart.
        No-op (returns inputs) if SciPy is unavailable or there are < 2 proposals.
        """
        try:
            from scipy.spatial import cKDTree
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
        except ImportError:
            return proposals_pred, instance_pred

        P = proposals_pred.shape[0]
        if P < 2:
            return proposals_pred, instance_pred

        coord_np = coord.detach().cpu().numpy()
        mv_np = point_mask_views.detach().cpu().numpy()  # (N, S)
        cls_np = instance_pred.detach().cpu().numpy()
        pred_np = proposals_pred.cpu().numpy()
        radius = float(self.rag_adjacency_radius * self.voxel_size)

        members = [np.nonzero(pred_np[p])[0] for p in range(P)]
        if sum(m.size for m in members) == 0:
            return proposals_pred, instance_pred

        conflict_metrics = ("shared_view", "shared_view_dominant")
        is_conflict = self.rag_sim_metric in conflict_metrics

        # --- per-cluster COUNT histogram over global 2D-mask ids (each id = a unique
        #     (view, object-mask)) and, for the legacy metrics, its normalization ---
        max_id = int(mv_np.max()) + 1 if mv_np.size and mv_np.max() >= 0 else 0
        cnts = np.zeros((P, max_id), dtype=np.float64)
        if max_id > 0:
            for p, idx in enumerate(members):
                if idx.size == 0:
                    continue
                vals = mv_np[idx]
                vals = vals[vals >= 0]
                if vals.size == 0:
                    continue
                cnts[p] = np.bincount(vals, minlength=max_id).astype(np.float64)
        # legacy intersection/cosine compare NORMALIZED histograms
        totals = cnts.sum(axis=1)
        hists = np.divide(
            cnts, totals[:, None], out=np.zeros_like(cnts), where=totals[:, None] > 0
        )

        # --- per-view structure for the shared-view conflict metrics ---
        # ``point_mask_views`` column s IS view s; entry >= 0 is that view's 2D-mask id.
        # vc[p, s] = # of proposal p's points visible in view s; dom[p, s] = p's dominant
        # (modal) mask id in view s (-1 where p is absent). Built only when needed.
        S = mv_np.shape[1] if mv_np.ndim == 2 else 0
        vc = np.zeros((P, S), dtype=np.float64)
        dom = None
        if is_conflict and S > 0:
            if self.rag_sim_metric == "shared_view_dominant":
                dom = np.full((P, S), -1, dtype=np.int64)
            for p, idx in enumerate(members):
                if idx.size == 0:
                    continue
                mvp = mv_np[idx]  # (n_p, S)
                vc[p] = (mvp >= 0).sum(axis=0)
                if dom is not None:
                    for s in range(S):
                        col = mvp[:, s]
                        col = col[col >= 0]
                        if col.size:
                            dom[p, s] = int(np.argmax(np.bincount(col)))

        # --- RAG adjacency: cross-cluster point pairs within radius (one global tree) ---
        owner = np.concatenate(
            [np.full(idx.size, p, dtype=np.int64) for p, idx in enumerate(members)]
        )
        allpts = np.concatenate([coord_np[idx] for idx in members], axis=0)
        adj_pairs = cKDTree(allpts).query_pairs(radius, output_type="ndarray")  # (M, 2)
        adj = set()
        if adj_pairs.shape[0]:
            oa, ob = owner[adj_pairs[:, 0]], owner[adj_pairs[:, 1]]
            cross = oa != ob
            for a, b in zip(oa[cross].tolist(), ob[cross].tolist()):
                adj.add((a, b) if a < b else (b, a))

        # --- keep merge edges: same predicted class + agreement test ---
        # Legacy metrics ("intersection"/"cosine"): merge when histogram SIMILARITY
        # >= rag_merge_thresh. Shared-view conflict metrics: merge when the ratio of
        # co-visible points landing in DIFFERENT masks of a shared view is <= thresh
        # (lower thresh = stricter = fewer merges; opposite direction from legacy).
        # Adjacent same-class pairs that share NO view carry no mask evidence -> skip.
        norms = np.linalg.norm(hists, axis=1) if max_id > 0 else np.zeros(P)
        ea, eb = [], []
        for a, b in adj:
            if cls_np[a] != cls_np[b]:
                continue
            if is_conflict:
                if S == 0:
                    continue
                shared = (vc[a] > 0) & (vc[b] > 0)
                if not shared.any():
                    continue  # no shared view -> no evidence to merge
                if self.rag_sim_metric == "shared_view":
                    den = float(vc[a] @ vc[b])  # co-visible cross-pairs over shared views
                    if den <= 0:
                        continue
                    num = float(cnts[a] @ cnts[b])  # same-(view,mask) cross-pairs
                    # num <= den when mask ids are unique per (view, object) (the data
                    # contract); clamp defends against malformed labels that break it.
                    conflict = min(max(1.0 - num / den, 0.0), 1.0)
                else:  # shared_view_dominant: per-view dominant-mask disagreement
                    ns = int(shared.sum())
                    conflict = float((dom[a][shared] != dom[b][shared]).sum()) / ns
                merge_edge = conflict <= self.rag_merge_thresh
            else:
                if self.rag_sim_metric == "cosine":
                    denom = float(norms[a] * norms[b])
                    sim = float(hists[a] @ hists[b] / denom) if denom > 0 else 0.0
                else:  # histogram intersection (default)
                    sim = float(np.minimum(hists[a], hists[b]).sum())
                merge_edge = sim >= self.rag_merge_thresh
            if merge_edge:
                ea.append(a)
                eb.append(b)

        if not ea:
            return proposals_pred, instance_pred

        g = coo_matrix((np.ones(len(ea), dtype=np.uint8), (ea, eb)), shape=(P, P))
        n_comp, comp = connected_components(g, directed=False)

        sizes = proposals_pred.sum(1)
        new_masks, new_classes = [], []
        for c in range(n_comp):
            grp = np.nonzero(comp == c)[0]
            if grp.size == 1:
                new_masks.append(proposals_pred[int(grp[0])])
                new_classes.append(instance_pred[int(grp[0])])
                continue
            grp_t = torch.from_numpy(grp).long()
            merged = proposals_pred[grp_t].sum(0).clamp(max=1).to(proposals_pred.dtype)
            new_masks.append(merged)
            # merged instance inherits the class of its largest fragment
            biggest = int(grp[int(torch.argmax(sizes[grp_t]))])
            new_classes.append(instance_pred[biggest])
        return torch.stack(new_masks, 0), torch.stack(new_classes, 0)
