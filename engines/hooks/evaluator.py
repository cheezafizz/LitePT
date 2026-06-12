import os
import re
import gc
import numpy as np
import wandb
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import utils.comm as comm
from utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS

from datasets.transform import Compose
from metrics.semantic import ConfusionMatrix

from torch_scatter import scatter_mean
from torch.nn.functional import normalize
from tqdm import tqdm

@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=True):
        self.write_cls_iou = write_cls_iou

    def before_train(self):
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="Epoch")
        
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]

            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.trainer.cfg.enable_wandb:
                log_dict = {
                    "Epoch": current_epoch,
                    "val/loss": loss_avg,
                    "val/mIoU": m_iou,
                    "val/mAcc": m_acc,
                    "val/allAcc": all_acc,
                }
                if self.write_cls_iou:
                    for i in range(self.trainer.cfg.data.num_classes):
                        log_dict[f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU"] = iou_class[i]
                wandb.log(log_dict)
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )

@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 50
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]
        self.steps_per_epoch = len(self.trainer.train_loader)
        self.eval_step_interval = getattr(self.trainer.cfg, "eval_step_interval", None)
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("val/*", step_metric="global_step")

    def after_epoch(self):
        if self.trainer.cfg.evaluate and self.eval_step_interval is None:
            self.eval()
            self.trainer.model.train()

    def after_step(self):
        if not self.trainer.cfg.evaluate or self.eval_step_interval is None:
            return
        global_step = (
            self.trainer.epoch * self.steps_per_epoch
            + self.trainer.comm_info["iter"]
            + 1
        )
        if global_step % self.eval_step_interval != 0:
            return
        for key in list(self.trainer.storage.histories().keys()):
            if key.startswith("val_"):
                self.trainer.storage.history(key).reset()
        self.eval()
        self.trainer.model.train()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.isin(segment, self.segment_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1

        # Drop the full-resolution boolean masks before returning. They are only
        # needed for the per-scene intersection bookkeeping above; evaluate_matches
        # works purely off the scalar fields (intersection / vert_count /
        # void_intersection / confidence). Keeping them would accumulate one mask
        # per predicted instance (~num_points bytes each) across every scene in
        # `scenes`, which exhausts host RAM on large val sets (e.g. the 6350-scene
        # v1.1.1 split) and stalls evaluation via swapping. The masks live both in
        # pred_instances and in the shallow copies referenced from
        # gt_instances[...]["matched_pred"], so strip both to actually free them.
        for label_name in pred_instances:
            for pred_inst in pred_instances[label_name]:
                pred_inst.pop("mask", None)
        for label_name in gt_instances:
            for gt_inst in gt_instances[label_name]:
                for matched_pred in gt_inst["matched_pred"]:
                    matched_pred.pop("mask", None)
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def _save_viz(self, scene_idx, input_dict, output_dict, idx):
        from .insseg_viz import (
            voxel_downsample_indices,
            instance_palette,
            colorize_predicted_instances,
            colorize_gt_instances,
            save_pointcloud_glb,
        )

        coord = input_dict["origin_coord"].detach().cpu().numpy()
        # The val Collect concatenates color+normal into `feat` (cols 0-2 = color, 3-5 = normal)
        color_voxel = input_dict["feat"][:, :3].detach().cpu().numpy()
        idx_np = idx.cpu().numpy() if hasattr(idx, "cpu") else np.asarray(idx)
        color = np.clip(color_voxel[idx_np] * 255.0, 0, 255).astype(np.uint8)

        pred = output_dict["pred_masks"].detach().cpu().numpy().astype(bool)
        scores = output_dict["pred_scores"].detach().cpu().numpy()
        gt_inst = input_dict["origin_instance"].detach().cpu().numpy()

        # AABBs computed on the full origin coord (before downsampling) so boxes hug true extents.
        # Filter uses self.min_region_sizes — same threshold the evaluator applies for AP matching.
        P = pred.shape[0]
        palette_pred = instance_palette(P)
        pred_boxes = []
        for p_idx in range(P):
            m = pred[p_idx]
            if m.sum() < self.min_region_sizes:
                continue
            pts = coord[m]
            pred_boxes.append((pts.min(0), pts.max(0), palette_pred[p_idx]))

        uniq_gt = np.unique(gt_inst[gt_inst != self.instance_ignore_index])
        palette_gt = instance_palette(len(uniq_gt), seed=1)
        gt_boxes = []
        for k, inst in enumerate(uniq_gt):
            m = gt_inst == inst
            if m.sum() < self.min_region_sizes:
                continue
            pts = coord[m]
            gt_boxes.append((pts.min(0), pts.max(0), palette_gt[k]))

        kept = voxel_downsample_indices(coord, 0.005)
        coord_s = coord[kept]
        color_s = color[kept]
        pred_s = pred[:, kept] if pred.shape[0] > 0 else pred
        gt_inst_s = gt_inst[kept]

        pred_color = colorize_predicted_instances(coord_s.shape[0], pred_s, scores)
        gt_color = colorize_gt_instances(
            gt_inst_s, ignore_value=self.instance_ignore_index
        )

        raw_name = input_dict.get("name", None)
        if isinstance(raw_name, (list, tuple)) and raw_name:
            raw_name = raw_name[0]
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode()
        if isinstance(raw_name, str) and raw_name:
            scene_tag = os.path.splitext(os.path.basename(raw_name))[0]
            scene_tag = re.sub(r"[^A-Za-z0-9._-]", "_", scene_tag)
        else:
            scene_tag = "scene{}".format(scene_idx)

        project_name = os.path.basename(os.path.normpath(self.trainer.cfg.save_path))
        base = os.path.join("/home/fai/workspace/jhp/LitePT/eval", project_name)
        ep = self.trainer.epoch + 1
        out_dir = os.path.join(base, "epoch_{:04d}".format(ep))
        save_pointcloud_glb(
            os.path.join(out_dir, "{}_rgb.glb".format(scene_tag)),
            coord_s,
            color_s,
        )
        save_pointcloud_glb(
            os.path.join(out_dir, "{}_pred.glb".format(scene_tag)),
            coord_s,
            pred_color,
            boxes=pred_boxes,
        )
        save_pointcloud_glb(
            os.path.join(out_dir, "{}_gt.glb".format(scene_tag)),
            coord_s,
            gt_color,
            boxes=gt_boxes,
        )

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]
            loss_keys = [
                k for k in ("loss", "seg_loss", "bias_l1_loss", "bias_cosine_loss")
                if k in output_dict
            ]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            if comm.get_rank() == 0 and i < 5 and "origin_coord" in input_dict:
                try:
                    self._save_viz(i, input_dict, output_dict, idx)
                except Exception as exc:
                    self.trainer.logger.warning(
                        "viz save failed at scene {idx}: {err}".format(idx=i, err=exc)
                    )

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            for key in loss_keys:
                self.trainer.storage.put_scalar(f"val_{key}", output_dict[key].item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        loss_component_avg = {
            key: self.trainer.storage.history(f"val_{key}").avg
            for key in ("seg_loss", "bias_l1_loss", "bias_cosine_loss")
            if f"val_{key}" in self.trainer.storage.histories()
        }
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            for key, value in loss_component_avg.items():
                self.trainer.writer.add_scalar(f"val/{key}", value, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
            for label_name in self.valid_class_names:
                cls = ap_scores["classes"][label_name]
                self.trainer.writer.add_scalar(
                    f"val/AP_{label_name}", cls["ap"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"val/AP50_{label_name}", cls["ap50%"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    f"val/AP25_{label_name}", cls["ap25%"], current_epoch
                )
            if self.trainer.cfg.enable_wandb:
                global_step = (
                    self.trainer.epoch * self.steps_per_epoch
                    + self.trainer.comm_info.get("iter", -1)
                    + 1
                )
                log_dict = {
                    "global_step": global_step,
                    "Epoch": current_epoch,
                    "val/loss": loss_avg,
                    "val/mAP": all_ap,
                    "val/AP50": all_ap_50,
                    "val/AP25": all_ap_25,
                }
                for key, value in loss_component_avg.items():
                    log_dict[f"val/{key}"] = value
                for label_name in self.valid_class_names:
                    cls = ap_scores["classes"][label_name]
                    log_dict[f"val/AP_{label_name}"] = cls["ap"]
                    log_dict[f"val/AP50_{label_name}"] = cls["ap50%"]
                    log_dict[f"val/AP25_{label_name}"] = cls["ap25%"]
                wandb.log(log_dict)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver
        # Release the per-eval buffers promptly (one bookkeeping dict per val scene)
        # so they don't linger until the next eval and inflate the memory peak.
        del scenes, scenes_sync, ap_scores
        gc.collect()
        torch.cuda.empty_cache()
