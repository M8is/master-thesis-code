from abc import ABC, abstractmethod
import torch


class MCEstimator(ABC, torch.nn.Module):
    def __init__(self, distribution, sample_size, *args, **kwargs):
        super().__init__()
        self.with_baseline = False
        self.distribution = distribution
        self.param_dims = distribution.param_dims
        self.sample_size = sample_size

    def forward(self, params):
        return self._sample(params) if self.training else self.distribution.sample(params)

    def backward(self, params, losses, retain_graph=False):
        self._backward(params, losses, retain_graph=retain_graph)

    def __str__(self):
        return f'{type(self).__name__} {type(self.distribution).__name__} {self.sample_size} sample(s)'

    @abstractmethod
    def _sample(self, params):
        pass

    @abstractmethod
    def _backward(self, params, losses, retain_graph):
        pass
