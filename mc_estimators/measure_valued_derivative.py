import torch

from .estimator_base import MCEstimator


class MVD(MCEstimator):
    def _sample(self, params):
        with torch.no_grad():
            return self.distribution.mvd_sample(params, self.sample_size)

    def _backward(self, params, losses, retain_graph):
        with torch.no_grad():
            grad = self.distribution.mvd_grad(params, losses)
        params.backward(gradient=grad, retain_graph=retain_graph)
