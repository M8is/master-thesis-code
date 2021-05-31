from abc import ABC, abstractmethod
import torch


class MCEstimator(ABC, torch.nn.Module):
    def __init__(self, distribution, sample_size, *args, **kwargs):
        super().__init__()
        self.baseline = kwargs["with_baseline"] if "with_baseline" in kwargs else True
        self.distribution = distribution
        self.param_dims = distribution.param_dims
        self.sample_size = sample_size

    def forward(self, raw_params):
        return self._sample(raw_params) if self.training else self.distribution.sample(raw_params, with_grad=False)

    @abstractmethod
    def _sample(self, raw_params):
        pass

    @abstractmethod
    def backward(self, raw_params, loss_fn, retain_graph=False):
        pass

    def __str__(self):
        return f'{type(self).__name__} {type(self.distribution).__name__} {self.sample_size} sample(s)'
