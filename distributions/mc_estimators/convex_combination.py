from contextlib import contextmanager
from typing import Callable

import torch

from distributions.distribution_base import Distribution
from distributions.mc_estimators.estimator_base import MCEstimator
from distributions.mc_estimators.gradient_filter import grad_filter
from distributions.mc_estimators.measure_valued import MVEstimator
from distributions.mc_estimators.score_function import SFEstimator


class MVSFEstimator(MCEstimator):
    def __init__(self, mv_dims: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mv_dims = mv_dims
        self.sfe = SFEstimator()
        self.mve = MVEstimator()

    def name(self) -> str:
        return "MVSF"

    def freeze(self) -> None:
        super().freeze()
        self.sfe.freeze()
        self.mve.freeze()

    def unfreeze(self) -> None:
        super().unfreeze()
        self.sfe.unfreeze()
        self.mve.unfreeze()

    def backward(self, distribution: Distribution, loss_fn: Callable[[torch.Tensor], torch.Tensor], sample_size: int,
                 retain_graph: bool) -> None:
        filter_ = self.__get_random_filter(distribution)
        with self.__filtered_grad(distribution, filter_):
            self.sfe.backward(distribution, loss_fn, sample_size, retain_graph=True)

        filter_ = torch.ones_like(distribution.params) - filter_
        with self.__filtered_grad(distribution, filter_):
            self.mve.backward(distribution, loss_fn, 1, retain_graph)

    def __get_random_filter(self, distribution):
        filter_ = torch.ones_like(distribution.params)
        rand_dims = torch.randint(filter_.size()[-1] - 1, (self.mv_dims,)).to(filter_.device)
        filter_.index_fill_(-1, rand_dims, 0)
        return filter_

    @contextmanager
    def __filtered_grad(self, distribution, filter_: torch.Tensor) -> None:
        orig_params = distribution.params
        distribution.params = grad_filter(distribution.params, filter_)
        yield
        distribution.params = orig_params
