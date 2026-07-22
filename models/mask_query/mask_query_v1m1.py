"""Query-based instance segmentation head (MQ-v1m1, "Mask-Query").

A Mask3D / SPFormer-style top-down alternative to PG-v1m2's bottom-up
offset+ball-query+BFS clustering. The shared LitePT backbone produces per-point
features; a set of learnable queries then attend (masked cross-attention) to a
coarse set of context tokens through a small transformer decoder, and each query
directly emits one instance (a class + a soft mask over points). Instances are
produced by the network, not by a geometric grouping heuristic -- there are no
`cluster_*`/`voxel_size` knobs and the head is scene-scale agnostic.

An auxiliary per-point semantic head (CE+Lovasz, identical to the -embed config) is
retained to stabilize early training. Output contract matches PG-v1m2 so the
existing InsSegEvaluator consumes it unchanged: at eval the model returns
pred_masks (M, N) / pred_scores (M,) / pred_classes (M,) on CPU; during training it
returns only scalar losses (InformationWriter calls .item() on every returned key).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter

from models.utils.structure import Point
from models.builder import MODELS, build_model
from models.losses import build_criteria

from .criterion import HungarianMatcher, SetCriterion


class MLP(nn.Module):
    """Simple multi-layer perceptron (used for the mask-embedding head)."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class FourierPos(nn.Module):
    """Fixed random-Fourier positional encoding of normalized 3D coordinates."""

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("proj", torch.randn(3, dim // 2) * scale)

    def forward(self, coord_norm):  # (Nc, 3) in ~[0, 1]
        x = 2 * math.pi * coord_norm @ self.proj  # (Nc, dim/2)
        return torch.cat([x.sin(), x.cos()], dim=-1)  # (Nc, dim)


class QueryDecoderLayer(nn.Module):
    """One pre-norm decoder layer: masked cross-attention (queries <- tokens),
    query self-attention, and an FFN."""

    def __init__(self, dim, nhead, ffn_dim, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.ReLU(inplace=True), nn.Linear(ffn_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, query, query_pos, tokens, token_pos, attn_mask):
        # query (1, K, d), tokens (1, Nc, d), attn_mask (K, Nc) bool (True = blocked)
        q = self.norm1(query)
        attended = self.cross_attn(
            query=q + query_pos,
            key=tokens + token_pos,
            value=tokens,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]
        query = query + attended
        q = self.norm2(query)
        qp = q + query_pos
        query = query + self.self_attn(qp, qp, q, need_weights=False)[0]
        q = self.norm3(query)
        query = query + self.ffn(q)
        return query


@MODELS.register_module("MQ-v1m1")
class MaskQuery(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_out_channels=64,
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        num_queries=100,
        dec_dim=256,
        dec_layers=6,
        dec_num_head=8,
        dec_ffn_dim=1024,
        context_grid_factor=5,
        mask_threshold=0.5,
        cost_class=2.0,
        cost_mask=5.0,
        cost_dice=5.0,
        loss_class_weight=2.0,
        loss_mask_weight=5.0,
        loss_dice_weight=5.0,
        loss_overlap_weight=0.0,
        use_boundary_weight=False,
        boundary_weight=5.0,
        boundary_radius=1,
        eos_coef=0.1,
        class_weight=None,
        criteria=None,
        freeze_backbone=False,
        unknown_bg_index=None,
        object_class_index=6,
        not_object_loss_weight=1.0,
        use_semantic_head=True,
    ):
        super().__init__()
        # Real-scene partial labels (tools/convert_ssl_scenes.py): points labeled
        # `unknown_bg_index` are known NOT to be `object_class_index` but their true
        # class is unknown. They get a -log(1 - p_object) penalty and are remapped to
        # `semantic_ignore_index` before the CE/Lovasz criteria. None disables the
        # path entirely (synthetic-only training is byte-identical to before).
        self.unknown_bg_index = unknown_bg_index
        self.object_class_index = object_class_index
        self.not_object_loss_weight = not_object_loss_weight
        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.segment_ignore_index = tuple(segment_ignore_index)
        self.instance_ignore_index = instance_ignore_index
        self.num_queries = num_queries
        self.dec_dim = dec_dim
        self.context_grid_factor = context_grid_factor
        self.mask_threshold = mask_threshold
        # anti-merge loss knobs (default off -> byte-identical to prior configs):
        #   loss_overlap_weight > 0  -> pairwise query repulsion (Method 1)
        #   use_boundary_weight      -> boundary-weighted mask BCE (Method 2),
        #     up-weighting tokens within `boundary_radius` coarse cells of a token
        #     owned by a DIFFERENT GT instance by `boundary_weight`.
        self.use_boundary_weight = use_boundary_weight
        self.boundary_weight = boundary_weight
        self.boundary_radius = boundary_radius

        self.backbone = build_model(backbone)
        # auxiliary per-point semantic head (retained supervision). use_semantic_head=
        # False drops the head entirely: no seg_head params, no CE/Lovasz seg_loss, and
        # no unknown_bg not-object partial loss (both read seg_logits).
        self.use_semantic_head = use_semantic_head
        if self.use_semantic_head:
            self.seg_head = nn.Linear(backbone_out_channels, semantic_num_classes)
            self.seg_criteria = build_criteria(criteria)

        # project backbone features into the decoder / mask-embedding space
        self.input_proj = nn.Sequential(
            nn.Linear(backbone_out_channels, dec_dim), nn.LayerNorm(dec_dim)
        )
        self.pos_enc = FourierPos(dec_dim)
        self.pos_proj = nn.Linear(dec_dim, dec_dim)

        self.query_feat = nn.Embedding(num_queries, dec_dim)
        self.query_pos = nn.Embedding(num_queries, dec_dim)

        self.layers = nn.ModuleList(
            QueryDecoderLayer(dec_dim, dec_num_head, dec_ffn_dim)
            for _ in range(dec_layers)
        )
        self.decoder_norm = nn.LayerNorm(dec_dim)
        self.class_head = nn.Linear(dec_dim, semantic_num_classes + 1)  # +1 no-object
        self.mask_head = MLP(dec_dim, dec_dim, dec_dim, num_layers=3)
        self.dec_num_head = dec_num_head

        matcher = HungarianMatcher(
            cost_class=cost_class, cost_mask=cost_mask, cost_dice=cost_dice
        )
        self.criterion = SetCriterion(
            num_classes=semantic_num_classes,
            matcher=matcher,
            loss_class_weight=loss_class_weight,
            loss_mask_weight=loss_mask_weight,
            loss_dice_weight=loss_dice_weight,
            loss_overlap_weight=loss_overlap_weight,
            eos_coef=eos_coef,
            class_weight=class_weight,
        )

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # ---- backbone feature extraction (copied verbatim from PG-v1m2) ----
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

    # ---- context tokenization ----
    def _tokenize(self, feat_s, grid_coord_s, coord_s):
        """Grid-pool per-scene point features to coarse context tokens.

        Returns tokens (Nc, d), token coords (Nc, 3), integer coarse-grid coords
        token_grid_coord (Nc, 3), and a point->token index map p2t (n,). Coarse cell =
        grid_coord // context_grid_factor (e.g. 5 * 2 mm = 1 cm).
        """
        coarse = torch.div(
            grid_coord_s, self.context_grid_factor, rounding_mode="floor"
        )
        token_grid_coord, p2t = torch.unique(coarse, dim=0, return_inverse=True)
        nc = int(p2t.max().item()) + 1
        tokens = torch_scatter.scatter_mean(feat_s, p2t, dim=0, dim_size=nc)
        token_coord = torch_scatter.scatter_mean(coord_s, p2t, dim=0, dim_size=nc)
        return tokens, token_coord, token_grid_coord, p2t

    @staticmethod
    def _norm_coord(coord):
        lo = coord.min(0)[0]
        span = coord.max(0)[0] - lo
        return (coord - lo) / (span + 1e-6)

    # ---- transformer decoder (per scene) ----
    def _predict(self, query, tokens):
        """query (1, K, d), tokens (1, Nc, d) -> class_logits (K, C+1),
        mask_logits (K, Nc), and the next-layer attn_mask (K, Nc) bool."""
        q = self.decoder_norm(query).squeeze(0)  # (K, d)
        class_logits = self.class_head(q)  # (K, C+1)
        mask_embed = self.mask_head(q)  # (K, d)
        mask_logits = mask_embed @ tokens.squeeze(0).transpose(0, 1)  # (K, Nc)
        # Mask3D masked attention: block tokens the query does NOT currently cover.
        attn_mask = (mask_logits.sigmoid() < 0.5).detach()
        # guard: a query that would mask out every token attends to all instead
        attn_mask[attn_mask.all(dim=-1)] = False
        return class_logits, mask_logits, attn_mask

    def _decode(self, tokens, token_pos):
        """Run all decoder layers. tokens (Nc, d), token_pos (Nc, d).
        Returns a list (len = dec_layers + 1) of (class_logits, mask_logits) with
        deep supervision, last entry = final prediction used for inference."""
        tokens_b = tokens.unsqueeze(0)  # (1, Nc, d)
        token_pos_b = token_pos.unsqueeze(0)
        query = self.query_feat.weight.unsqueeze(0)  # (1, K, d)
        query_pos = self.query_pos.weight.unsqueeze(0)

        preds = []
        cls, msk, attn_mask = self._predict(query, tokens_b)
        preds.append((cls, msk))
        for layer in self.layers:
            query = layer(query, query_pos, tokens_b, token_pos_b, attn_mask)
            cls, msk, attn_mask = self._predict(query, tokens_b)
            preds.append((cls, msk))
        return preds

    # ---- ground-truth token construction (per scene) ----
    def _build_scene_gt(self, instance_s, segment_s, p2t, nc, token_grid_coord=None):
        """Build token-level GT for one scene. Returns gt_labels (G,) long and
        gt_masks (G, Nc) float. G = # of thing instances (class not in
        segment_ignore_index). Each token is assigned to the instance (or background)
        that owns the majority of its points. Also returns token_ignore (Nc,) bool:
        tokens majority-owned by instance-ignored OBJECT points (real-scene seg_id=-2,
        object evidence the matcher left ungrouped) — these are excluded from the
        mask/dice loss so they never act as negative (background) mask evidence.

        Fourth return is token_weight (Nc,) float, the per-token mask-BCE multiplier
        (all ones unless boundary weighting is enabled; see _boundary_token_weight)."""
        device = instance_s.device
        valid = instance_s != self.instance_ignore_index
        uids = torch.unique(instance_s[valid]) if valid.any() else instance_s[:0]

        ignore_pts = (~valid) & (segment_s == self.object_class_index)
        if ignore_pts.any():
            ign_cnt = torch.bincount(p2t[ignore_pts], minlength=nc)
            tok_cnt = torch.bincount(p2t, minlength=nc)
            token_ignore = ign_cnt * 2 > tok_cnt
        else:
            token_ignore = torch.zeros(nc, dtype=torch.bool, device=device)

        keep_uids, keep_labels = [], []
        for uid in uids.tolist():
            m = instance_s == uid
            cls = int(torch.mode(segment_s[m])[0].item())
            if cls in self.segment_ignore_index:
                continue
            keep_uids.append(uid)
            keep_labels.append(cls)

        g = len(keep_uids)
        if g == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(0, nc, device=device),
                token_ignore,
                torch.ones(nc, device=device),
            )

        # compact per-point label: index into keep_uids, or g (=background)
        lbl = torch.full_like(instance_s, g)
        for gi, uid in enumerate(keep_uids):
            lbl[instance_s == uid] = gi
        flat = p2t * (g + 1) + lbl
        counts = torch.bincount(flat, minlength=nc * (g + 1)).view(nc, g + 1)
        token_assign = counts.argmax(dim=1)  # (Nc,) values in [0, g], g == background
        arange_g = torch.arange(g, device=device)
        gt_masks = (token_assign[None, :] == arange_g[:, None]).float()  # (G, Nc)
        gt_labels = torch.tensor(keep_labels, dtype=torch.long, device=device)
        token_weight = self._boundary_token_weight(token_assign, g, token_grid_coord, nc)
        return gt_labels, gt_masks, token_ignore, token_weight

    def _boundary_token_weight(self, token_assign, g, token_grid_coord, nc):
        """Per-token mask-BCE multiplier (Nc,). All ones unless boundary weighting is
        on, in which case tokens within `boundary_radius` coarse cells of a token owned
        by a DIFFERENT real GT instance (foreground<->foreground boundary) are set to
        `boundary_weight`. Computed once per scene on the integer coarse grid via a
        sorted-key neighbour lookup. token_assign (Nc,) in [0, g] with g == background;
        token_grid_coord (Nc, 3) integer coarse-cell coords."""
        device = token_assign.device
        weight = torch.ones(nc, device=device)
        if not self.use_boundary_weight or token_grid_coord is None or g < 2:
            return weight

        fg = token_assign < g  # foreground tokens (assigned to a real instance)
        coord = token_grid_coord.long()
        coord = coord - coord.min(dim=0)[0]  # shift to >= 0, order preserved
        span = coord.max(dim=0)[0] + 1  # per-axis extent
        sx = int(span[1].item()) * int(span[2].item())
        sy = int(span[2].item())
        # monotone bijective pack of in-range coords -> 1-D key
        key = coord[:, 0] * sx + coord[:, 1] * sy + coord[:, 2]  # (Nc,)
        order = torch.argsort(key)
        key_sorted = key[order]
        assign_sorted = token_assign[order]

        r = self.boundary_radius
        boundary = torch.zeros(nc, dtype=torch.bool, device=device)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx, ny, nz = coord[:, 0] + dx, coord[:, 1] + dy, coord[:, 2] + dz
                    in_range = (
                        (nx >= 0) & (nx < span[0])
                        & (ny >= 0) & (ny < span[1])
                        & (nz >= 0) & (nz < span[2])
                    )
                    nkey = nx * sx + ny * sy + nz
                    idx = torch.searchsorted(key_sorted, nkey).clamp(max=nc - 1)
                    hit = in_range & (key_sorted[idx] == nkey)  # neighbour exists
                    na = assign_sorted[idx]  # neighbour's assignment
                    boundary |= hit & fg & (na < g) & (na != token_assign)
        weight[boundary] = self.boundary_weight
        return weight

    # ---- inference (per scene) ----
    def _infer_scene(self, pred_last, p2t, start, end, n_total):
        cls, msk = pred_last  # (K, C+1), (K, Nc)
        prob = cls.softmax(-1)
        real = prob[:, : self.semantic_num_classes]  # drop no-object
        cls_score, labels = real.max(dim=-1)  # (K,)
        mask_prob = msk.sigmoid()  # (K, Nc)
        mask_bin = mask_prob > self.mask_threshold
        tok_cnt = mask_bin.sum(dim=-1)  # (K,)
        mask_score = (mask_prob * mask_bin.float()).sum(dim=-1) / tok_cnt.clamp(min=1)
        score = cls_score * mask_score

        scene_idx = torch.arange(start, end, device=cls.device)
        masks, scores, classes = [], [], []
        for k in range(cls.shape[0]):
            if int(tok_cnt[k]) < 1:
                continue
            if int(labels[k]) in self.segment_ignore_index:
                continue
            pt_fg = mask_bin[k][p2t]  # (n_scene,) bool
            if int(pt_fg.sum()) < 1:
                continue
            full = torch.zeros(n_total, dtype=torch.int)
            full[scene_idx[pt_fg].cpu()] = 1
            masks.append(full)
            scores.append(score[k].detach().cpu())
            classes.append(labels[k].detach().cpu())
        return masks, scores, classes

    def forward(self, data_dict, return_point=False):
        if return_point:
            return dict(point=self.backbone(data_dict))

        coord = data_dict["coord"]
        grid_coord = data_dict["grid_coord"]
        offset = data_dict["offset"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]

        feat = self._backbone_feat(data_dict)  # (N, backbone_out_channels)
        if self.use_semantic_head:
            seg_logits = self.seg_head(feat)  # (N, C)
            not_object_loss = seg_logits.sum() * 0.0
            if self.unknown_bg_index is not None:
                unknown_bg = segment == self.unknown_bg_index
                if unknown_bg.any():
                    # -log(1 - p_object) on known-not-object points, from log-softmax
                    # for stability; clamp keeps the loss finite when p_object
                    # saturates.
                    log_prob = F.log_softmax(seg_logits[unknown_bg].float(), dim=-1)
                    p_obj = (
                        log_prob[:, self.object_class_index].exp().clamp(max=1.0 - 1e-6)
                    )
                    not_object_loss = -torch.log1p(-p_obj).mean()
                    segment = segment.clone()
                    segment[unknown_bg] = self.semantic_ignore_index
            if (segment != self.semantic_ignore_index).any():
                seg_loss = self.seg_criteria(seg_logits, segment)
            else:
                # Real scenes with no detected objects have every point remapped to
                # ignore above; CE/Lovasz cannot handle an all-ignored target.
                seg_loss = seg_logits.sum() * 0.0
        else:
            # No semantic head: keep zero-valued loss keys so InformationWriter /
            # wandb logging and the total-loss sum are shape-identical. unknown_bg
            # points still need remapping so _build_scene_gt treats them as plain
            # background rather than a distinct class.
            seg_loss = feat.sum() * 0.0
            not_object_loss = feat.sum() * 0.0
            if self.unknown_bg_index is not None:
                unknown_bg = segment == self.unknown_bg_index
                if unknown_bg.any():
                    segment = segment.clone()
                    segment[unknown_bg] = self.semantic_ignore_index
        mask_feat = self.input_proj(feat)  # (N, d)

        n_total = feat.shape[0]
        total = dict(loss_ce=0.0, loss_mask=0.0, loss_dice=0.0, loss_overlap=0.0)
        n_scenes = 0
        all_masks, all_scores, all_classes = [], [], []

        for b in range(offset.shape[0]):
            start = 0 if b == 0 else int(offset[b - 1])
            end = int(offset[b])
            feat_s = mask_feat[start:end]
            coord_s = coord[start:end]
            grid_coord_s = grid_coord[start:end]
            instance_s = instance[start:end]
            segment_s = segment[start:end]

            tokens, token_coord, token_grid_coord, p2t = self._tokenize(
                feat_s, grid_coord_s, coord_s
            )
            token_pos = self.pos_proj(self.pos_enc(self._norm_coord(token_coord)))
            preds = self._decode(tokens, token_pos)

            gt_labels, gt_masks, token_ignore, token_weight = self._build_scene_gt(
                instance_s, segment_s, p2t, tokens.shape[0], token_grid_coord
            )
            tw = token_weight if self.use_boundary_weight else None
            for cls, msk in preds:
                d = self.criterion(cls, msk, gt_labels, gt_masks, token_ignore, tw)
                total["loss_ce"] = total["loss_ce"] + d["loss_ce"]
                total["loss_mask"] = total["loss_mask"] + d["loss_mask"]
                total["loss_dice"] = total["loss_dice"] + d["loss_dice"]
                total["loss_overlap"] = total["loss_overlap"] + d["loss_overlap"]
            n_scenes += 1

            if not self.training:
                m, s, c = self._infer_scene(preds[-1], p2t, start, end, n_total)
                all_masks += m
                all_scores += s
                all_classes += c

        denom = max(n_scenes, 1)
        loss_ce = total["loss_ce"] / denom
        loss_mask = total["loss_mask"] / denom
        loss_dice = total["loss_dice"] / denom
        loss_overlap = total["loss_overlap"] / denom
        loss = (
            seg_loss
            + self.not_object_loss_weight * not_object_loss
            + loss_ce
            + loss_mask
            + loss_dice
            + loss_overlap
        )

        return_dict = dict(
            loss=loss,
            loss_ce=loss_ce,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            loss_overlap=loss_overlap,
            seg_loss=seg_loss,
            loss_not_object=not_object_loss,
        )

        if not self.training:
            if len(all_masks) > 0:
                return_dict["pred_masks"] = torch.stack(all_masks, dim=0)
                return_dict["pred_scores"] = torch.stack(all_scores, dim=0)
                return_dict["pred_classes"] = torch.stack(all_classes, dim=0)
            else:
                return_dict["pred_masks"] = torch.zeros((0, n_total), dtype=torch.int)
                return_dict["pred_scores"] = torch.tensor([])
                return_dict["pred_classes"] = torch.tensor([])
        return return_dict
