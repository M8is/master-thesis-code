from abc import ABC, abstractmethod
import torch


class MCEstimator(ABC, torch.nn.Module):
    def __init__(self, distribution, sample_size, *args, **kwargs):
        super().__init__()
        self.with_baseline = False
        self.distribution = distribution
        self.sample_size = sample_size

    def forward(self, params):
        return self._sample(params) if self.training else self.distribution.sample(params)

    def backward(self, params, losses):
        self.distribution.kl(params).mean().backward(retain_graph=True)
        self._backward(params, losses)

    def __str__(self):
        return f'{type(self).__name__} {type(self.distribution).__name__} {self.sample_size} sample(s)'

    @abstractmethod
    def _sample(self, params):
        pass

    @abstractmethod
    def _backward(self, params, losses):
        pass
