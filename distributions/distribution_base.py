from abc import ABC, abstractmethod
from typing import Tuple, Callable, Union

import torch


class Distribution(ABC):
    def __init__(self, raw_params):
        self.params = self._as_params(raw_params)

    def backward(self, gradient_estimator, loss_fn: Callable[[torch.Tensor], torch.Tensor],
                 sample_size: int, retain_graph: bool = False):
        gradient_estimator.backward(self, loss_fn, sample_size, retain_graph)

    @staticmethod
    def param_dims(latent_dim: int) -> Tuple[int, ...]:
        return latent_dim,

    @abstractmethod
    def _as_params(self, raw_params):
        pass

    @property
    @abstractmethod
    def kl(self):
        pass

    @property
    def pdf(self):
        raise NotImplemented(f"PDF is not yet implemented for {type(self).__name__}")

    @abstractmethod
    def mvd_backward(self, losses, retain_graph) -> None:
        pass

    @abstractmethod
    def sample(self, sample_shape: Union[torch.Size, Tuple[int, ...]] = torch.Size([])) -> torch.Tensor:
        pass

    @abstractmethod
    def rsample(self, sample_shape: Union[torch.Size, Tuple[int, ...]] = torch.Size([])) -> torch.Tensor:
        pass

    @abstractmethod
    def mvsample(self, size: int) -> torch.Tensor:
        # TODO: refactor parameter to sample_shape, to match sample and rsample signatures?
        pass

    @abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        pass
