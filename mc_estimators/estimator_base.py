from abc import ABC, abstractmethod
from typing import Optional

import torch


class MCEstimator(ABC, torch.nn.Module):
    def __init__(self, distribution, sample_size, *args, **kwargs):
        super().__init__()
        self.baseline = kwargs["with_baseline"] if "with_baseline" in kwargs else True
        self.distribution = distribution
        self.param_dims = distribution.param_dims
        self.sample_size = sample_size

    def forward(self, raw_params):
        with torch.no_grad():
            return self.distribution.sample(raw_params)[0]

    def get_std(self, raw_params, zero_grad_fn, loss_fn, n_estimates=500):
        old_sample_size = self.sample_size
        self.sample_size = 1
        grads = []
        zero_grad_fn()
        for i in range(n_estimates):
            grad = self.backward(raw_params, loss_fn, retain_graph=(i + 1) < n_estimates, return_grad=True)
            grads.append(grad)
            zero_grad_fn()
        self.sample_size = old_sample_size
        return torch.stack(grads).std(dim=0).mean()

    @abstractmethod
    def backward(self, raw_params, loss_fn, retain_graph=False, return_grad=False) -> Optional[torch.Tensor]:
        pass

    def __str__(self):
        return f'{type(self).__name__} {type(self.distribution).__name__} {self.sample_size} sample(s)'
