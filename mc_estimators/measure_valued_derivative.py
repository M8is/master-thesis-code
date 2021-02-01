import torch

from .estimator_base import MCEstimator


class MVD(MCEstimator):
    def sample(self, params):
        return self.distribution.mvd_sample(params, self.sample_size, with_grad=True)

    def backward(self, params, losses):
        with torch.no_grad():
            split_dim = 1 if len(params) > 1 else 0
            pos_losses, neg_losses = torch.split(losses, losses.size(split_dim) // 2, dim=split_dim)
            delta = (pos_losses - neg_losses).mean(dim=split_dim)
            grad = self.distribution.mvd_constant(params) * delta
            torch.stack(params).backward(gradient=grad)
