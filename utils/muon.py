"""Hybrid Muon + AdamW optimizer.

Muon (MomentUm Orthogonalized by Newton-Schulz, Jordan et al. 2024) applies an
orthogonalized momentum update to 2D weight matrices; everything that is not a
plain matrix (biases, norms, embeddings, spconv kernels, scalars) keeps standard
AdamW. This file vendors the single-device update: DDP all-reduces gradients
before step(), so per-rank steps are identical and no distributed variant is
needed; gradient accumulation also needs nothing special (it acts before step()).

Used by the `-query-muon` config as an optimizer race against the AdamW `-query`
run -- see configs/scannet-v1.1.1-2of3/insseg-litept-small-v1m2-2of3-query-muon.py.
"""

import torch


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Approximate UV^T of the SVD of G via 5 Newton-Schulz iterations.

    Quintic iteration with coefficients tuned to maximize slope at zero
    (Keller Jordan's constants). Runs in fp32 regardless of input dtype --
    the training loop produces bf16 grads and NS is numerically touchy.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class MuonAdamW(torch.optim.Optimizer):
    """Two fixed param groups: group 0 = Muon (2D matrices), group 1 = AdamW (rest).

    `params` must be an iterable of (name, parameter) tuples -- the shape-based
    split needs names to exclude nn.Embedding weights (2D but must NOT be
    orthogonalized). Group order is stable (Muon first) so a scheduler max_lr
    list maps 1:1: max_lr=[muon_lr, lr].

    `lr` is the AdamW-group LR (keeps the build_optimizer(cfg.lr) contract);
    `muon_lr` is the Muon-group LR, internally rescaled per-matrix by
    sqrt(max(1, rows/cols)) as in the reference implementation.
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        muon_lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        weight_decay=0.0,
        embedding_keywords=("query_feat", "query_pos"),
    ):
        named = list(params)
        assert named and isinstance(named[0], (tuple, list)) and len(named[0]) == 2, (
            "MuonAdamW requires (name, param) tuples; "
            "build_optimizer passes model.named_parameters() for type='MuonAdamW'"
        )
        muon_params, adamw_params = [], []
        for name, p in named:
            if p.ndim == 2 and not any(k in name for k in embedding_keywords):
                muon_params.append(p)
            else:
                adamw_params.append(p)
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=weight_decay,
            use_muon=False,
        )
        groups = [
            dict(params=muon_params, lr=muon_lr, use_muon=True),
            dict(params=adamw_params, lr=lr, use_muon=False),
        ]
        super().__init__(groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(g)
                    update = (
                        g.add(buf, alpha=group["momentum"])
                        if group["nesterov"]
                        else buf
                    )
                    update = zeropower_via_newtonschulz5(
                        update, steps=group["ns_steps"]
                    )
                    # decoupled weight decay, matching the AdamW group's semantics
                    if group["weight_decay"] != 0.0:
                        p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    p.add_(update.to(p.dtype), alpha=-group["lr"] * scale)
            else:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if "exp_avg" not in state:
                        state["step_count"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step_count"] += 1
                    t = state["step_count"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    if group["weight_decay"] != 0.0:
                        p.mul_(1.0 - group["lr"] * group["weight_decay"])
                    bias_c1 = 1.0 - beta1**t
                    bias_c2 = 1.0 - beta2**t
                    denom = (exp_avg_sq / bias_c2).sqrt_().add_(group["eps"])
                    p.addcdiv_(exp_avg / bias_c1, denom, value=-group["lr"])
        return loss
