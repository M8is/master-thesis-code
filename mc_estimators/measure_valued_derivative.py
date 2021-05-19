import torch

from .estimator_base import MCEstimator


class MVD(MCEstimator):
    def _sample(self, raw_params):
        with torch.no_grad():
            return self.distribution.mvd_sample(raw_params, self.sample_size)

    def _backward(self, raw_params, losses, retain_graph):
        self.distribution.mvd_backward(raw_params, losses, retain_graph)
