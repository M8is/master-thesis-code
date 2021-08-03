from typing import Type

import torch

from distributions.distribution_base import Distribution
from models.stochastic_model import StochasticModel


class LinearRegressor(StochasticModel):
    def __init__(self, latent_dim: int, distribution_type: Type[Distribution]):
        super().__init__()
        param_dims = distribution_type.param_dims(latent_dim)
        self.raw_params = torch.nn.Parameter(torch.randn((1, sum(param_dims))))
        self.distribution_type = distribution_type

    def encode(self, data: torch.Tensor) -> Distribution:
        raw_params = self.raw_params.repeat_interleave(data.size()[0], dim=0)
        return self.distribution_type(raw_params)

    def interpret(self, sample: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return (data * sample).sum(dim=-1)
