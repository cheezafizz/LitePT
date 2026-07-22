import sys
import glob
import os
import shutil
import subprocess
import time
import gc
import csv
import wandb
import torch
import torch.utils.data
from collections import OrderedDict
from packaging import version
from functools import partial
from tqdm import tqdm

if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from utils.timer import Timer
from utils.comm import is_main_process, synchronize
from utils.cache import shared_dict
from utils.scheduler import CosineScheduler
import utils.comm as comm

from .default import HookBase
from .builder import HOOKS


AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)

def get_testers():
    # lazy import to prevent circular import
    from engines.test import TESTERS
    return TESTERS
@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.curr_iter = 0
        self.model_output_keys = []

    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)
        # Per-step wandb logging cadence. 1 = log every step (legacy). On long runs
        # (~847k steps) logging every step floods the wandb client; throttle via config.
        self.wandb_log_interval = getattr(self.trainer.cfg, "wandb_log_interval", 1)
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("params/*", step_metric="global_step")
            wandb.define_metric("train_batch/*", step_metric="global_step")
            wandb.define_metric("train/*", step_metric="epoch")

    def before_step(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info


    def compute_total_grad_norm(self, parameters, norm_type=2):
        parameters = [p for p in parameters if p.grad is not None]
        norm_list = [p.grad.detach().norm(norm_type) for p in parameters]
        total_norm = torch.norm(torch.stack(norm_list), norm_type)
        return total_norm

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            self.model_output_keys = model_output_dict.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )
            if (
                self.trainer.cfg.enable_wandb
                and self.curr_iter % self.wandb_log_interval == 0
            ):
                # compute_total_grad_norm reduces over all params; only pay it on
                # steps we actually log to wandb.
                log_dict = {
                    "global_step": self.curr_iter,
                    "params/lr": lr,
                    "params/norm": self.compute_total_grad_norm(self.trainer.model.parameters(), norm_type=2.0),
                }
                for key in self.model_output_keys:
                    log_dict[f"train_batch/{key}"] = self.trainer.storage.history(key).val
                wandb.log(log_dict, step=self.curr_iter)

    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )

            if self.trainer.cfg.enable_wandb:
                log_dict = {"epoch": self.trainer.epoch + 1}
                for key in self.model_output_keys:
                    log_dict[f"train/{key}"] = self.trainer.storage.history(key).avg
                wandb.log(log_dict)


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last

    def after_epoch(self):
        self._save()
        if is_main_process():
            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )

    def after_step(self):
        # With step-interval evaluation a mid-epoch death would otherwise lose up
        # to a whole epoch of work, and best-tracking would only consider the last
        # eval of each epoch. Save (and update best) right after each step eval —
        # InsSegEvaluator runs earlier in the hook list, so comm_info already
        # holds this eval's metric.
        eval_step_interval = getattr(self.trainer.cfg, "eval_step_interval", None)
        if not self.trainer.cfg.evaluate or eval_step_interval is None:
            return
        global_step = (
            self.trainer.epoch * len(self.trainer.train_loader)
            + self.trainer.comm_info["iter"]
            + 1
        )
        if global_step % eval_step_interval != 0:
            return
        self._save()

    def _save(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info.get("current_metric_value")
                current_metric_name = self.trainer.comm_info.get("current_metric_name")
                if current_metric_value is None:
                    # Step-interval evaluation hasn't produced a metric yet; skip best-tracking.
                    self.trainer.logger.info(
                        "No validation metric available this epoch; skipping best-model update."
                    )
                else:
                    if current_metric_value > self.trainer.best_metric_value:
                        self.trainer.best_metric_value = current_metric_value
                        is_best = True
                        self.trainer.logger.info(
                            "Best validation {} updated to: {:.4f}".format(
                                current_metric_name, current_metric_value
                            )
                        )
                    self.trainer.logger.info(
                        "Currently Best {}: {:.4f}".format(
                            current_metric_name, self.trainer.best_metric_value
                        )
                    )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            self.trainer.logger.info(f"Unexpected keys: {load_state_info[1]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")



@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        TESTERS = get_testers()
        tester = TESTERS.build(
            dict(type=cfg.test.type, cfg=cfg, model=self.trainer.model)
        )
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            self.trainer.logger.info(f"Loading weight at: {best_path}")
            checkpoint = torch.load(best_path, weights_only=False)
            state_dict = checkpoint["state_dict"]
            load_state_info = tester.model.load_state_dict(state_dict, strict=True)
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            self.trainer.logger.info(f"Unexpected keys: {load_state_info[1]}")
        tester.test()


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()



@HOOKS.register_module()
class RuntimeProfiler_training(HookBase):
    def __init__(
        self,
        interrupt=True,
        warm_up=5,
    ):
        self.interrupt = interrupt
        self.warm_up = warm_up

    def nvidia_smi_mem(self, device_id=0):
        """Return memory.used (MB) for a specific GPU."""
        cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {device_id}"
        out = subprocess.check_output(cmd.split()).decode().strip()
        return int(out)

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")

        total_fwd_time = 0.0
        num_batches = 0

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # baseline driver memory
        base_mem = self.nvidia_smi_mem(torch.cuda.current_device())

        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler()

        # optional warmup
        print("\n Warming up...\n")
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= self.warm_up:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with auto_cast(
                enabled=True, dtype=AMP_DTYPE['float16']
            ):  
                output_dict = self.trainer.model(input_dict)
                loss = output_dict['loss']

            self.trainer.optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(self.trainer.optimizer)
            scaler.update()
            
        print("\n Measuring inference latency across dataset...\n")
        
        for i, input_dict in enumerate(tqdm(self.trainer.train_loader)):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
        
            torch.cuda.synchronize()
            fwd_start = time.perf_counter()

            with auto_cast(
                enabled=True, dtype=AMP_DTYPE['float16']
            ):  
                output_dict = self.trainer.model(input_dict)
                loss = output_dict['loss']

            self.trainer.optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(self.trainer.optimizer)
            scaler.update()
            
            torch.cuda.synchronize()
            fwd_end = time.perf_counter()
            total_fwd_time += (fwd_end - fwd_start)

            num_batches += 1


        torch.cuda.synchronize()
        alloc_MB = torch.cuda.max_memory_allocated() / 1024**3
        resv_MB = torch.cuda.max_memory_reserved() / 1024**3
        smi_MB = self.nvidia_smi_mem(torch.cuda.current_device())
        delta_MB = smi_MB - base_mem

        # ---- Final report ----
        print("\n===== Runtime Profiling Results =====")
        print(f"Average Inference time: {total_fwd_time/num_batches*1000:.3f} ms")
        print(f"PyTorch max_memory_allocated : {alloc_MB:.2f} GB")
        # max_memory_reserved is used for benchmarking memory in the paper
        print(f"PyTorch max_memory_reserved  : {resv_MB:.2f} GB")
        print(f"nvidia-smi memory.used       : {smi_MB/1024:.3f} GB")
        print(f"Δ from baseline              : {delta_MB/1024:.3f} GB")
        print("=====================================\n")
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfiler_inference(HookBase):
    """Per-scene inference profiler.

    Runs over the val loader (batch size 1 -> one scene per iteration) and reports,
    for each requested precision, the per-scene distribution of forward+clustering
    latency and (optionally) the full end-to-end breakdown (preprocess / H2D /
    forward / back-projection), plus peak GPU VRAM and peak host RAM (RSS).
    Runs in ``before_train`` and exits when ``interrupt=True`` so no training starts.
    """

    def __init__(
        self,
        interrupt=True,
        warm_up=5,
        precisions=("fp32",),
        measure_end_to_end=False,
        csv_path=None,
    ):
        self.interrupt = interrupt
        self.warm_up = warm_up
        if isinstance(precisions, str):
            precisions = (precisions,)
        self.precisions = tuple(str(p).lower() for p in precisions)
        self.measure_end_to_end = measure_end_to_end
        self.csv_path = csv_path

    def nvidia_smi_mem(self, device_id=0):
        """Return memory.used (MB) for a specific GPU."""
        cmd = f"nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i {device_id}"
        out = subprocess.check_output(cmd.split()).decode().strip()
        return int(out)

    def _autocast_ctx(self, precision):
        """Autocast context for the model forward under ``precision``."""
        use_amp = precision != "fp32"
        dtype = AMP_DTYPE["bfloat16"] if precision == "bf16" else AMP_DTYPE["float16"]
        if version.parse(torch.__version__) >= version.parse("2.4"):
            return torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=dtype)
        if use_amp:
            return torch.cuda.amp.autocast(dtype=dtype)
        import contextlib

        return contextlib.nullcontext()

    @staticmethod
    def _to_cuda(input_dict):
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

    @staticmethod
    def _stats(values):
        if not values:
            return dict(mean=0.0, median=0.0, p95=0.0, max=0.0, min=0.0)
        s = sorted(values)
        n = len(s)

        def pct(p):
            if n == 1:
                return s[0]
            k = (n - 1) * p
            f = int(k)
            c = min(f + 1, n - 1)
            return s[f] + (s[c] - s[f]) * (k - f)

        return dict(mean=sum(s) / n, median=pct(0.5), p95=pct(0.95), max=s[-1], min=s[0])

    def before_train(self):
        self.trainer.logger.info("Profiling inference runtime ...")
        loader = self.trainer.val_loader
        if loader is None:
            self.trainer.logger.info(
                "No val_loader available (evaluate=False?); skipping inference profiling."
            )
            if self.interrupt:
                sys.exit(0)
            return

        try:
            import psutil

            proc = psutil.Process()
        except Exception as exc:  # pragma: no cover - psutil expected to be present
            self.trainer.logger.warning(f"psutil unavailable; host RAM not measured: {exc}")
            proc = None

        self.trainer.model.eval()
        device = torch.cuda.current_device()

        summaries = []
        all_rows = []
        failures = []
        for precision in self.precisions:
            try:
                result = self._profile_precision(precision, loader, proc, device)
                summaries.append(result["summary"])
                all_rows.extend(result["rows"])
            except Exception as exc:
                import traceback as _tb

                msg = f"{type(exc).__name__}: {exc}"
                self.trainer.logger.warning(
                    f"Precision '{precision}' profiling failed and was skipped: {msg}"
                )
                print(f"\n[precision = {precision}] FAILED: {msg}\n{_tb.format_exc()}")
                failures.append((precision, msg))

        self._report(summaries, failures)
        if self.csv_path is not None:
            self._write_csv(all_rows)

        if self.interrupt:
            sys.exit(0)

    def _profile_precision(self, precision, loader, proc, device):
        print(f"\n[Profiling precision = {precision}]")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        base_smi = self.nvidia_smi_mem(device)
        base_rss = proc.memory_info().rss if proc is not None else 0
        peak_rss = base_rss
        # Per-scene reset_peak_memory_stats() (below) clobbers the global peak, so
        # track the run-level high-water marks ourselves to keep the run-level report.
        run_peak_alloc_mb = 0.0
        run_peak_reserved_mb = 0.0

        pointops = None
        if self.measure_end_to_end:
            try:
                import pointops as _pointops

                pointops = _pointops
            except Exception as exc:
                self.trainer.logger.warning(
                    f"pointops unavailable; back-projection time will be 0: {exc}"
                )

        # ---- warmup ----
        print(f"  warming up ({self.warm_up}) ...")
        with torch.no_grad():
            for i, input_dict in enumerate(loader):
                if i >= self.warm_up:
                    break
                self._to_cuda(input_dict)
                with self._autocast_ctx(precision):
                    self.trainer.model(input_dict)
        torch.cuda.synchronize()

        # ---- measure ----
        print("  measuring ...")
        rows = []
        with torch.no_grad():
            it = iter(loader)
            idx = 0
            pbar = tqdm(total=len(loader))
            while True:
                # ---- per-scene memory reset: capture this scene's peak in isolation ----
                torch.cuda.reset_peak_memory_stats()
                rss_start = proc.memory_info().rss if proc is not None else 0

                # ---- load: CPU preprocessing (GridSample 2mm) + collate ----
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    input_dict = next(it)
                except StopIteration:
                    break
                load_ms = (time.perf_counter() - t0) * 1000.0

                if "offset" in input_dict:
                    num_voxels = int(input_dict["offset"][-1])
                else:
                    num_voxels = int(input_dict["coord"].shape[0])
                if "origin_coord" in input_dict:
                    num_origin = int(input_dict["origin_coord"].shape[0])
                else:
                    num_origin = num_voxels
                name = input_dict.get("name", None)
                if isinstance(name, (list, tuple)):
                    name = name[0] if len(name) else f"scene{idx}"
                if not isinstance(name, str):
                    name = f"scene{idx}"

                # ---- H2D transfer ----
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                self._to_cuda(input_dict)
                torch.cuda.synchronize()
                h2d_ms = (time.perf_counter() - t0) * 1000.0

                # ---- forward + clustering (GPU backbone + ballquery + CPU bfs) ----
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with self._autocast_ctx(precision):
                    output_dict = self.trainer.model(input_dict)
                torch.cuda.synchronize()
                fwd_ms = (time.perf_counter() - t0) * 1000.0

                # ---- back-projection to full-resolution points ----
                backproj_ms = 0.0
                if (
                    self.measure_end_to_end
                    and pointops is not None
                    and "origin_coord" in input_dict
                    and "pred_masks" in output_dict
                ):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    bp_idx, _ = pointops.knn_query(
                        1,
                        input_dict["coord"].float(),
                        input_dict["offset"].int(),
                        input_dict["origin_coord"].float(),
                        input_dict["origin_offset"].int(),
                    )
                    bp_idx = bp_idx.cpu().flatten().long()
                    _ = output_dict["pred_masks"][:, bp_idx]
                    torch.cuda.synchronize()
                    backproj_ms = (time.perf_counter() - t0) * 1000.0

                total_ms = load_ms + h2d_ms + fwd_ms + backproj_ms

                # ---- per-scene peak memory (forward + clustering + back-projection) ----
                # max_memory_allocated is the high-water since the reset at loop top, so
                # it captures this scene's true activation peak. reserved is the caching
                # allocator's pool (monotonic across the run; per-scene value is mostly the
                # running high-water, kept for completeness).
                gpu_alloc_mb = torch.cuda.max_memory_allocated() / 1024**2
                gpu_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
                run_peak_alloc_mb = max(run_peak_alloc_mb, gpu_alloc_mb)
                run_peak_reserved_mb = max(run_peak_reserved_mb, gpu_reserved_mb)

                if proc is not None:
                    rss_now = proc.memory_info().rss
                    peak_rss = max(peak_rss, rss_now)
                    rss_mb = rss_now / 1024**2
                    rss_delta_mb = (rss_now - rss_start) / 1024**2
                else:
                    rss_mb = 0.0
                    rss_delta_mb = 0.0

                rows.append(
                    dict(
                        precision=precision,
                        name=name,
                        num_voxels=num_voxels,
                        num_origin_points=num_origin,
                        load_ms=load_ms,
                        h2d_ms=h2d_ms,
                        fwd_ms=fwd_ms,
                        backproj_ms=backproj_ms,
                        total_ms=total_ms,
                        gpu_alloc_mb=gpu_alloc_mb,
                        gpu_reserved_mb=gpu_reserved_mb,
                        rss_mb=rss_mb,
                        rss_delta_mb=rss_delta_mb,
                    )
                )
                idx += 1
                pbar.update(1)
            pbar.close()

        torch.cuda.synchronize()
        smi_now = self.nvidia_smi_mem(device)
        n = max(len(rows), 1)
        total_origin = sum(r["num_origin_points"] for r in rows)
        total_fwd_s = sum(r["fwd_ms"] for r in rows) / 1000.0
        summary = dict(
            precision=precision,
            n=len(rows),
            gpu_alloc_gb=run_peak_alloc_mb / 1024,
            gpu_reserved_gb=run_peak_reserved_mb / 1024,
            gpu_smi_gb=smi_now / 1024,
            gpu_smi_delta_gb=(smi_now - base_smi) / 1024,
            rss_peak_gb=peak_rss / 1024**3,
            rss_delta_gb=(peak_rss - base_rss) / 1024**3,
            mean_voxels=sum(r["num_voxels"] for r in rows) / n,
            mean_origin=total_origin / n,
            throughput_pts_s=(total_origin / total_fwd_s) if total_fwd_s > 0 else 0.0,
            fwd=self._stats([r["fwd_ms"] for r in rows]),
            total=self._stats([r["total_ms"] for r in rows]),
            load=self._stats([r["load_ms"] for r in rows]),
            h2d=self._stats([r["h2d_ms"] for r in rows]),
            backproj=self._stats([r["backproj_ms"] for r in rows]),
            gpu_alloc=self._stats([r["gpu_alloc_mb"] for r in rows]),
            gpu_reserved=self._stats([r["gpu_reserved_mb"] for r in rows]),
            rss=self._stats([r["rss_mb"] for r in rows]),
            rss_delta=self._stats([r["rss_delta_mb"] for r in rows]),
        )
        return dict(summary=summary, rows=rows)

    def _report(self, summaries, failures=None):
        lines = ["\n========== Inference Benchmark =========="]
        cfg = self.trainer.cfg
        lines.append(f"config save_path : {getattr(cfg, 'save_path', 'n/a')}")
        lines.append(f"weight           : {getattr(cfg, 'weight', None)}")
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            gpu_name = "n/a"
        lines.append(f"GPU              : {gpu_name}")
        for precision, msg in failures or []:
            lines.append(f"--- precision = {precision} | SKIPPED (unsupported): {msg} ---")
        for s in summaries:
            f = s["fwd"]
            lines.append("")
            lines.append(f"--- precision = {s['precision']} | scenes = {s['n']} ---")
            lines.append(
                f"  scene size       : {s['mean_voxels']:.0f} voxels/scene (input), "
                f"{s['mean_origin']:.0f} origin pts/scene"
            )
            lines.append("  per-scene latency (ms)     mean / median /    p95 /    max")
            lines.append(
                f"    forward+cluster: {f['mean']:8.2f} / {f['median']:8.2f} / "
                f"{f['p95']:8.2f} / {f['max']:8.2f}"
            )
            if self.measure_end_to_end:
                t, l, h, b = s["total"], s["load"], s["h2d"], s["backproj"]
                lines.append(
                    f"    end-to-end     : {t['mean']:8.2f} / {t['median']:8.2f} / "
                    f"{t['p95']:8.2f} / {t['max']:8.2f}"
                )
                lines.append(
                    f"    breakdown(mean): preprocess {l['mean']:.2f} + H2D {h['mean']:.2f} "
                    f"+ forward {f['mean']:.2f} + backproj {b['mean']:.2f} ms"
                )
            lines.append(
                f"  throughput       : {s['throughput_pts_s'] / 1e6:.2f} M origin-pts/s (by forward time)"
            )
            ma, rs = s["gpu_alloc"], s["rss"]
            lines.append("  per-scene GPU alloc (MB)   mean / median /    p95 /    max")
            lines.append(
                f"    activation peak: {ma['mean']:8.1f} / {ma['median']:8.1f} / "
                f"{ma['p95']:8.1f} / {ma['max']:8.1f}"
            )
            lines.append("  per-scene host RSS  (MB)   mean / median /    p95 /    max")
            lines.append(
                f"    process peak   : {rs['mean']:8.1f} / {rs['median']:8.1f} / "
                f"{rs['p95']:8.1f} / {rs['max']:8.1f}"
            )
            lines.append("  (reserved / nvidia-smi / RSS below are run-level high-water marks)")
            lines.append(f"  GPU max_allocated: {s['gpu_alloc_gb']:.2f} GB")
            lines.append(f"  GPU max_reserved : {s['gpu_reserved_gb']:.2f} GB")
            lines.append(
                f"  nvidia-smi used  : {s['gpu_smi_gb']:.2f} GB (Δ {s['gpu_smi_delta_gb']:+.2f} GB)"
            )
            lines.append(
                f"  host RAM pk RSS  : {s['rss_peak_gb']:.2f} GB (Δ {s['rss_delta_gb']:+.2f} GB)"
            )
        lines.append("=========================================\n")
        report = "\n".join(lines)
        print(report)
        try:
            self.trainer.logger.info(report)
        except Exception:
            pass

    def _write_csv(self, rows):
        if not rows:
            return
        path = self.csv_path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        fields = [
            "precision",
            "name",
            "num_voxels",
            "num_origin_points",
            "load_ms",
            "h2d_ms",
            "fwd_ms",
            "backproj_ms",
            "total_ms",
            "gpu_alloc_mb",
            "gpu_reserved_mb",
            "rss_mb",
            "rss_delta_mb",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in fields})
        self.trainer.logger.info(f"Per-scene benchmark CSV written to {path}")
        print(f"Per-scene benchmark CSV: {path}")



@HOOKS.register_module()
class WeightDecaySchedular(HookBase):
    def __init__(
        self,
        base_value=0.04,
        final_value=0.2,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.scheduler = None

    def before_train(self):
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.scheduler = CosineScheduler(
            base_value=self.base_value,
            final_value=self.final_value,
            total_iters=self.trainer.cfg.scheduler.total_steps,
        )
        self.scheduler.iter = curr_step

    def before_step(self):
        wd = self.scheduler.step()
        for param_group in self.trainer.optimizer.param_groups:
            param_group["weight_decay"] = wd
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/wd", wd, self.scheduler.iter)


@HOOKS.register_module()
class GarbageHandler(HookBase):
    def __init__(self, interval=150, disable_auto=True, empty_cache=False):
        self.interval = interval
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self.iter = 1

    def before_train(self):
        if self.disable_auto:
            gc.disable()
            self.trainer.logger.info("Disable automatic garbage collection")

    def before_epoch(self):
        self.iter = 1

    def after_step(self):
        if self.iter % self.interval == 0:
            gc.collect()
            if self.empty_cache:
                torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        gc.collect()
        torch.cuda.empty_cache()
