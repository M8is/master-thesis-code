from abc import ABC, abstractmethod
from typing import Callable

import torch

from distributions.distribution_base import Distribution


class MCEstimator(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._frozen = False

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    def freeze(self) -> None:
        """ Freeze the internal state of the estimator, s.t. gradient estimates will not update internal values.

        :return: Nothing.
        """
        self._frozen = True

    def unfreeze(self) -> None:
        """ Unfreeze the internal state of the estimator to resume training.

        :return: Nothing.
        """
        self._frozen = False

    @abstractmethod
    def backward(self, distribution: Distribution, loss_fn: Callable[[torch.Tensor], torch.Tensor], sample_size: int,
                 retain_graph: bool) -> None:
        pass

    def __str__(self):
        return self.name() + ' gradient estimator'
