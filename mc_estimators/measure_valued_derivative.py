import torch

from .estimator_base import MCEstimator


class MVD(MCEstimator):
    def sample(self, params):
        with torch.no_grad():
            return self.distribution.mvd_sample(params, self.sample_size)

    def backward(self, params, losses):
        with torch.no_grad():
            pos_losses, neg_losses = losses
            delta = (pos_losses - neg_losses).mean(dim=1).transpose(-2, -1)
            c = self.distribution.mvd_constant(params)
            grad = c * delta
            if len(params) <= 1:
                grad.unsqueeze_(0)
        torch.stack(params).backward(gradient=grad)
