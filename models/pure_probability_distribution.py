from typing import Type, List

import torch

from distributions.distribution_base import Distribution
from models.stochastic_model import StochasticModel


class PureProbDistModel(StochasticModel):
    def __init__(self, distribution_type: Type[Distribution], init_params: List[float]):
        super().__init__()
        self.distribution_type = distribution_type
        self.raw_params = torch.nn.Parameter(torch.FloatTensor([init_params]))

    def encode(self, data: torch.Tensor) -> Distribution:
        return self.distribution_type(self.raw_params)

    def interpret(self, sample: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return sample
