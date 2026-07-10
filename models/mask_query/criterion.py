"""Set-prediction criterion for the query-based instance head (MQ-v1m1).

A DETR / Mask2Former-style bipartite matcher + set criterion adapted to 3D point
masks. Everything is per-scene (instance ids restart per scene) and operates at the
context-token resolution the decoder runs on: predictions are (K, C+1) class logits
and (K, Nc) mask logits; ground truth is (G,) instance classes and (G, Nc) binary
token masks. The matcher pairs each GT instance with one query (Hungarian, via
scipy.optimize.linear_sum_assignment); the loss is classification CE (unmatched
queries -> no-object) + mask sigmoid-BCE + mask Dice on the matched pairs.

The model is responsible for building the token-level GT and for running this per
decoder layer (deep supervision) and per scene, then summing/averaging the scalar
dicts returned here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_sigmoid_bce_cost(logits, targets):
    """Pairwise mean sigmoid-BCE cost. logits (K, Nc), targets (G, Nc) in {0,1}.
    Returns (K, G)."""
    nc = logits.shape[1]
    pos = F.binary_cross_entropy_with_logits(
        logits, torch.ones_like(logits), reduction="none"
    )  # (K, Nc)
    neg = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction="none"
    )  # (K, Nc)
    cost = pos @ targets.transpose(0, 1) + neg @ (1.0 - targets).transpose(0, 1)
    return cost / nc


def batch_dice_cost(logits, targets):
    """Pairwise Dice cost. logits (K, Nc), targets (G, Nc) in {0,1}. Returns (K, G)."""
    probs = logits.sigmoid()  # (K, Nc)
    numerator = 2.0 * (probs @ targets.transpose(0, 1))  # (K, G)
    denominator = probs.sum(-1)[:, None] + targets.sum(-1)[None, :]  # (K, G)
    return 1.0 - (numerator + 1.0) / (denominator + 1.0)


def dice_loss(logits, targets):
    """Mean Dice loss over matched pairs. logits (m, Nc), targets (m, Nc)."""
    probs = logits.sigmoid()
    numerator = 2.0 * (probs * targets).sum(-1)
    denominator = probs.sum(-1) + targets.sum(-1)
    loss = 1.0 - (numerator + 1.0) / (denominator + 1.0)
    return loss.mean()


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_mask=5.0, cost_dice=5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(self, pred_logits, pred_masks, gt_labels, gt_masks):
        """Match one scene / one layer.

        pred_logits (K, C+1), pred_masks (K, Nc) logits, gt_labels (G,) long,
        gt_masks (G, Nc) float in {0,1}. Returns (row, col) long tensors: row indexes
        queries, col indexes GT instances, both length min(K, G)."""
        from scipy.optimize import linear_sum_assignment

        g = gt_labels.shape[0]
        if g == 0:
            empty = pred_logits.new_zeros(0, dtype=torch.long)
            return empty, empty
        prob = pred_logits.softmax(-1)  # (K, C+1)
        cost_class = -prob[:, gt_labels]  # (K, G)
        cost_mask = batch_sigmoid_bce_cost(pred_masks, gt_masks)  # (K, G)
        cost_dice = batch_dice_cost(pred_masks, gt_masks)  # (K, G)
        cost = (
            self.cost_class * cost_class
            + self.cost_mask * cost_mask
            + self.cost_dice * cost_dice
        )
        cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=-1e6)
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        row = torch.as_tensor(row, dtype=torch.long, device=pred_logits.device)
        col = torch.as_tensor(col, dtype=torch.long, device=pred_logits.device)
        return row, col


class SetCriterion(nn.Module):
    """Per-scene, per-layer set-prediction loss. `num_classes` counts the real
    classes (thing + stuff); the no-object slot lives at index `num_classes`, so
    class logits must have `num_classes + 1` columns."""

    def __init__(
        self,
        num_classes,
        matcher,
        loss_class_weight=2.0,
        loss_mask_weight=5.0,
        loss_dice_weight=5.0,
        eos_coef=0.1,
        class_weight=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_class_weight = loss_class_weight
        self.loss_mask_weight = loss_mask_weight
        self.loss_dice_weight = loss_dice_weight
        # empty_weight: (C+1,) CE weights; real classes get `class_weight` (or 1),
        # the no-object slot gets `eos_coef`.
        empty_weight = torch.ones(num_classes + 1)
        if class_weight is not None:
            empty_weight[:num_classes] = torch.as_tensor(class_weight, dtype=torch.float)
        empty_weight[num_classes] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, pred_logits, pred_masks, gt_labels, gt_masks):
        """One scene, one decoder layer. Returns dict(loss_ce, loss_mask, loss_dice),
        all scalar tensors on the prediction device."""
        device = pred_logits.device
        k = pred_logits.shape[0]
        row, col = self.matcher(pred_logits, pred_masks, gt_labels, gt_masks)

        target_classes = torch.full(
            (k,), self.num_classes, dtype=torch.long, device=device
        )
        if row.numel() > 0:
            target_classes[row] = gt_labels[col]
        loss_ce = F.cross_entropy(
            pred_logits, target_classes, weight=self.empty_weight.to(device)
        )

        if row.numel() > 0:
            src = pred_masks[row]  # (m, Nc)
            tgt = gt_masks[col]  # (m, Nc)
            loss_mask = F.binary_cross_entropy_with_logits(src, tgt)
            loss_dice = dice_loss(src, tgt)
        else:
            loss_mask = pred_masks.sum() * 0.0
            loss_dice = pred_masks.sum() * 0.0

        return dict(
            loss_ce=self.loss_class_weight * loss_ce,
            loss_mask=self.loss_mask_weight * loss_mask,
            loss_dice=self.loss_dice_weight * loss_dice,
        )
