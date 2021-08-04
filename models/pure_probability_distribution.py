from typing import Type

import torch

from distributions.categorical import Categorical
from distributions.distribution_base import Distribution
from models.stochastic_model import StochasticModel


class PureProbDistModel(StochasticModel):
    def __init__(self, distribution_type: Type[Distribution], init_params: torch.FloatTensor,
                 normalize_samples: bool = False, **kwargs):
        super().__init__()
        self.distribution_type = distribution_type
        self.raw_params = torch.nn.Parameter(init_params)
        self.kwargs = kwargs

        if normalize_samples and distribution_type != Categorical:
            raise ValueError("Normalized samples only work with categorical distribution.")
        self.normalize_samples = normalize_samples

    def encode(self, data: torch.Tensor) -> Distribution:
        return self.distribution_type(self.raw_params, **self.kwargs)

    def interpret(self, sample: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        if self.normalize_samples:
            sample = sample / (self.raw_params.shape[-1] - 1)
        return sample
