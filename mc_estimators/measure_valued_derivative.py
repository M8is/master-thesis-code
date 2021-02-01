import torch

from mc_estimators.base import MCEstimator


class MVD(MCEstimator):
    def sample(self, params):
        return self.distribution.mvd_sample(params, self.sample_size, with_grad=True)

    def backward(self, params, losses):
        torch.stack(params).backward(self.distribution.mvd_grad(params, losses))
