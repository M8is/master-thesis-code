import math
from typing import List

import torch
import torch.utils.data

from models.pure_probability_distribution import PureProbDistModel
from models.stochastic_model import StochasticModel
from tasks.trainer import StochasticTrainer
from utils.distribution_factory import get_distribution_type
from utils.eval_util import eval_mode


class TrainPolynomial(StochasticTrainer):
    def __init__(self, learning_rate: float, distribution: str, polynomial: str, init_params: List[float] = None,
                 repeat_params: bool = 1, random_params: int = 0, *args, **kwargs):
        super().__init__(*args, dataset='empty', batch_size=kwargs.get('batch_size', 0), **kwargs)
        self.polynomial = getattr(self, polynomial)
        if random_params > 0:
            init_params = torch.randn((random_params,))
        else:
            init_params = torch.FloatTensor([repeat_params * init_params])
        distribution_type = get_distribution_type(distribution)
        self.__model = PureProbDistModel(distribution_type, init_params, **kwargs)
        self.__optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    @property
    def model(self) -> StochasticModel:
        return self.__model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return self.polynomial(outputs).float().squeeze(-1)

    def _test_epoch(self) -> None:
        with eval_mode(self.model):
            empty = torch.empty(0)
            distribution = self.model.encode(empty)
            x_recons = self.model.interpret(distribution.sample((10000,)), empty)
            losses = self.loss(empty, empty, x_recons).mean()
            self.test_losses.add(losses)

    def post_epoch(self, epoch: int) -> None:
        # Override removes info logging
        pass

    def post_iteration(self, batch_id: int, loss: torch.Tensor, kld: torch.Tensor) -> None:
        # Override removes info logging
        pass

    @staticmethod
    def quadratic(x: torch.Tensor) -> torch.Tensor:
        return (x - .5) ** 2

    @staticmethod
    def quadratic_shifted(x: torch.Tensor) -> torch.Tensor:
        return (x - 3) ** 2

    @staticmethod
    def sinusoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(4 * math.pi * x)
