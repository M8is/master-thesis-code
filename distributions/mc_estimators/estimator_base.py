from abc import ABC, abstractmethod
from typing import Callable

import torch

from distributions.distribution_base import Distribution


class MCEstimator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def backward(self, distribution: Distribution, loss_fn: Callable[[torch.Tensor], torch.Tensor], sample_size: int,
                 retain_graph: bool) -> None:
        pass

    def __str__(self):
        return self.name + ' gradient estimator'
