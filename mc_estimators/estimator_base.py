from abc import ABC, abstractmethod
import torch


class MCEstimator(ABC, torch.nn.Module):
    def __init__(self, distribution, sample_size, *args, **kwargs):
        super().__init__()
        self.with_kl = kwargs["with_kl"] if "with_kl" in kwargs else True
        self.baseline = kwargs["with_baseline"] if "with_baseline" in kwargs else True
        self.distribution = distribution
        self.param_dims = distribution.param_dims
        self.sample_size = sample_size

    def forward(self, raw_params):
        return self._sample(raw_params) if self.training else self.distribution.sample(raw_params, with_grad=False)

    def backward(self, raw_params, losses, retain_graph=False):
        if self.with_kl:
            self.distribution.kl(raw_params).mean().backward(retain_graph=True)
        self._backward(raw_params, losses, retain_graph=retain_graph)

    def __str__(self):
        return f'{type(self).__name__} {type(self.distribution).__name__} {self.sample_size} sample(s)'

    @abstractmethod
    def _sample(self, params):
        pass

    @abstractmethod
    def _backward(self, params, losses, retain_graph):
        pass
