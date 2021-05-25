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

    def backward(self, raw_params, loss_fn, retain_graph=False):
        self._backward(raw_params, loss_fn, retain_graph=retain_graph)

    def __str__(self):
        return f'{type(self).__name__} {type(self.distribution).__name__} {self.sample_size} sample(s)'

    @abstractmethod
    def _sample(self, raw_params):
        pass

    @abstractmethod
    def _backward(self, raw_params, loss_fn, retain_graph):
        pass
