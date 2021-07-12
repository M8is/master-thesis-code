from typing import List

import torch
import torch.utils.data

from models.pure_probability_distribution import PureProbDistModel
from models.stochastic_model import StochasticModel
from tasks.trainer import StochasticTrainer
from utils.distribution_factory import get_distribution_type


class TrainPolynomial(StochasticTrainer):
    def __init__(self, learning_rate: float, distribution: str, init_params: List[float], repeat_params: bool = 0,
                 *args, **kwargs):
        super().__init__(*args, dataset='empty', batch_size=kwargs.get('batch_size', 0), optimize_kld=False,
                         print_interval=kwargs.get('print_interval', float('inf')), **kwargs)

        self.__model = PureProbDistModel(get_distribution_type(distribution), repeat_params * init_params)
        self.__optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    @property
    def model(self) -> StochasticModel:
        return self.__model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return self.polynomial(outputs)

    @staticmethod
    def polynomial(x: torch.Tensor) -> torch.Tensor:
        y = 0.25 * (x - 1.) ** 2
        return y.float().squeeze(-1)
