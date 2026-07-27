import os
import sys
import weakref
import wandb
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial
from pathlib import Path

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import utils.comm as comm
from datasets import build_dataset, point_collate_fn, collate_fn
from models import build_model
from utils.logger import get_root_logger
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.events import EventStorage, ExceptionWriter
from utils.registry import Registry

TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        # The val loader streams full-resolution scenes (no SphereCrop) worker->main
        # through the multiprocessing queue. PyTorch's default "file_descriptor"
        # sharing strategy fails partway through eval in _share_fd_cpu_ (surfacing as
        # a mangled `AttributeError: 'super' object has no attribute 'super'`). Use
        # "file_system" sharing -- backed by /tmp files instead of shm fds -- matching
        # TesterBase. Set in the main process so spawned DataLoader workers inherit it.
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if self.writer is not None and comm.is_main_process():
            self.writer.close()
        # Cleanly close the wandb run so its in-process buffers/state are flushed
        # and released at end of training (no finish() was called previously).
        cfg = getattr(self, "cfg", None)
        if cfg is not None and getattr(cfg, "enable_wandb", False) and comm.is_main_process():
            wandb.finish()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    @staticmethod
    def _split_input_dict(input_dict, accum):
        """Split a collated batch into `accum` scene-contiguous micro-batches for
        gradient accumulation. Returns a list of (sub_input_dict, weight) where the
        weights sum to 1 (weight = #scenes in the micro-batch / total #scenes), so
        summing the per-micro-batch (scene-mean) losses reproduces the full-batch
        scene-mean. Per-point tensors (first dim == total points) are sliced to the
        micro-batch's point range, `offset` is rebased, everything else is passed
        through. With accum <= 1 (the default) this returns the batch unchanged with
        weight 1.0, so the single-step path is bit-identical to before."""
        offset = input_dict.get("offset", None)
        if offset is None or accum is None or accum <= 1:
            return [(input_dict, 1.0)]
        n_scenes = int(offset.shape[0])
        n_groups = min(int(accum), n_scenes)
        if n_groups <= 1:
            return [(input_dict, 1.0)]
        total_pts = int(offset[-1])
        base, rem = divmod(n_scenes, n_groups)
        micro_batches, start_scene = [], 0
        for i in range(n_groups):
            size = base + (1 if i < rem else 0)
            a, b = start_scene, start_scene + size - 1  # inclusive scene range
            start_scene += size
            p0 = 0 if a == 0 else int(offset[a - 1])
            p1 = int(offset[b])
            sub = {}
            for key, val in input_dict.items():
                if key == "offset":
                    sub[key] = offset[a : b + 1] - p0
                elif torch.is_tensor(val) and val.dim() >= 1 and val.shape[0] == total_pts:
                    sub[key] = val[p0:p1]
                else:
                    sub[key] = val
            micro_batches.append((sub, size / n_scenes))
        return micro_batches

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        # Gradient accumulation: split the batch into scene-contiguous micro-batches,
        # backward each (loss scaled by its scene fraction), and take ONE optimizer +
        # scheduler step per loader iteration. This keeps peak memory to a single
        # micro-batch while preserving the effective batch size and the scheduler's
        # step count (total_steps = iters/epoch * epochs is unchanged). accum=1 (the
        # default for every existing config) reduces to the original single-forward path.
        accum = getattr(self.cfg, "gradient_accumulation_steps", 1)
        micro_batches = self._split_input_dict(input_dict, accum)

        self.optimizer.zero_grad()
        log_dict = {}
        for sub_dict, weight in micro_batches:
            with auto_cast(
                enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
            ):
                output_dict = self.model(sub_dict)
                loss = output_dict["loss"] * weight
            if self.cfg.enable_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            for key, val in output_dict.items():
                if torch.is_tensor(val) and val.dim() == 0:
                    log_dict[key] = log_dict.get(key, 0.0) + val.detach() * weight

        if self.cfg.enable_amp:
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = log_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        if self.cfg.enable_wandb and comm.is_main_process():
            tag, name = Path(self.cfg.save_path).parts[-2:]
            # Reattach to the original wandb run on resume instead of minting a new
            # run per restart (crash/resume otherwise splits one training across
            # several dashboard runs, and few-point stub runs render as bar charts).
            # The run id is persisted next to the checkpoints on first init.
            run_id_file = Path(self.cfg.save_path) / "wandb_run_id.txt"
            run_id = None
            if self.cfg.resume:
                if run_id_file.is_file():
                    run_id = run_id_file.read_text().strip() or None
                else:
                    # Older runs predate the id file; fall back to the id encoded in
                    # the latest local run dir (wandb/run-<timestamp>-<id>).
                    latest = Path(self.cfg.save_path) / "wandb" / "latest-run"
                    if latest.exists():
                        run_id = Path(os.path.realpath(latest)).name.split("-")[-1]
            run = wandb.init(
                project=self.cfg.wandb_project,
                name=f"{tag}/{name}",
                tags=[tag],
                id=run_id,
                # "allow" resumes the run if it exists but still starts cleanly if
                # the server has no such run (e.g. the original never synced).
                resume="allow" if run_id else None,
                dir=self.cfg.save_path,
                settings=wandb.Settings(api_key=self.cfg.wandb_key),
                config=self.cfg,
            )
            run_id_file.write_text(run.id)
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            subset_size = getattr(self.cfg, "val_subset_size", None)
            if subset_size is not None:
                n = min(subset_size, len(val_data))
                # Evenly-spaced (not first-n) so the subset spans the whole sorted
                # val split — first-n silently dropped every scene of later name
                # prefixes (e.g. all machine-004 real-ssl scenes).
                indices = torch.linspace(0, len(val_data) - 1, n).long().tolist()
                val_data = torch.utils.data.Subset(val_data, indices)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            # Validation is bs=1 over full-resolution scenes (no SphereCrop). With the full
            # train worker count the val loader pins num_workers * prefetch_factor large
            # scenes in host RAM during every eval -- a ~25GB spike that OOM-kills mid-eval.
            # Cap workers/prefetch and drop pin_memory (eval is not H2D-bound).
            num_worker_val = getattr(self.cfg, "num_worker_val", None)
            if num_worker_val is None:
                num_worker_val = min(self.cfg.num_worker_per_gpu, 4)
            loader_kwargs = dict(
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=num_worker_val,
                pin_memory=False,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
            if num_worker_val > 0:
                loader_kwargs["prefetch_factor"] = 2
                # Keep the (capped, pin_memory=False) val workers alive across evals.
                # With eval_step_interval the loader is re-iterated ~hundreds of times;
                # persistent workers remove that many spawn/teardown cycles (IPC churn /
                # fragility) at the cost of a small fixed resident baseline. The eval-time
                # peak is unchanged (still num_workers * prefetch_factor), so this does not
                # regress the original val-loader OOM driver.
                loader_kwargs["persistent_workers"] = True
            val_loader = torch.utils.data.DataLoader(val_data, **loader_kwargs)
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
