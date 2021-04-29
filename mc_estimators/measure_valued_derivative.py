import torch

from .estimator_base import MCEstimator


class MVD(MCEstimator):
    def _sample(self, raw_params):
        params, samples = self.distribution.mvd_sample(raw_params, self.sample_size)
        return params, samples.detach()

    def _backward(self, params, losses, retain_graph):
        with torch.no_grad():
            grad = self.distribution.mvd_grad(params, losses)
        params.squeeze_()
        grad.squeeze_()
        assert grad.shape == params.shape, f"Grad shape {grad.shape} != params shape {params.shape}"
        params.backward(gradient=grad, retain_graph=retain_graph)
