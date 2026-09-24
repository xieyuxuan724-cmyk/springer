# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Small extensions to mjlab's native distributed PPO runner."""

import torch
from rsl_rl.modules.normalization import EmpiricalNormalization
from mjlab.utils.gpu import select_gpus


def resolve_gpu_ids(requested):
    """Use at most eight visible devices, retaining mjlab's visible-index semantics."""
    if requested is None:
        return None
    available = min(select_gpus("all")[1], torch.cuda.device_count())
    if requested in ("auto", "all"):
        count = min(8, available)
        return list(range(count)) if count else None
    if requested is not None:
        if not requested or len(requested) > 8 or len(set(requested)) != len(requested):
            raise ValueError("Select between one and eight distinct GPUs, or None for CPU")
        if any(index < 0 or index >= available for index in requested):
            raise ValueError("GPU indices must refer to visible devices")
    return requested


class DistributedEmpiricalNormalization(EmpiricalNormalization):
    """Pool observations across ranks; preserve the native checkpoint/export format."""

    @torch.jit.unused
    @torch.no_grad()
    def update(self, x):
        if not self.training or (self.until is not None and self.count >= self.until):
            return
        if not torch.distributed.is_initialized():
            return super().update(x)
        # Center before forming second moments; float64 avoids cancellation for offsets.
        delta = x.double() - self._mean.double()
        width = self._mean.numel()
        moments = torch.cat((delta.sum(0).flatten(), delta.square().sum(0).flatten(),
                             x.new_tensor([len(x)], dtype=torch.float64)))
        torch.distributed.all_reduce(moments)
        n = moments[-1]
        if n == 0:
            return
        shift = (moments[:width] / n).reshape_as(self._mean)
        variance = (moments[width:2*width] / n).reshape_as(self._var) - shift.square()
        self.count += n.long()
        rate = n / self.count
        self._mean.add_((rate * shift).to(self._mean.dtype))
        self._var.add_((rate * (variance.clamp_min(0) - self._var.double()
                               + (1-rate) * shift.square())).to(self._var.dtype))
        self._var.clamp_min_(0)
        self._std.copy_(self._var.sqrt())


def synchronize_normalizers(*models):
    for model in models:
        old = model.obs_normalizer
        if not isinstance(old, EmpiricalNormalization):
            raise TypeError("Distributed BENI training requires empirical observation normalization")
        norm = DistributedEmpiricalNormalization(old._mean.shape[1:], old.eps, old.until).to(old._mean.device)
        norm.load_state_dict(old.state_dict())
        norm.train(old.training)
        model.obs_normalizer = norm
