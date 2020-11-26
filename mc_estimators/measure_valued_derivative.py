import torch

from numpy.random import weibull
from math import pi, sqrt
from mc_estimators.probabilistic_objective_gradient import Probabilistic


class MVD(Probabilistic):
    def __init__(self, episode_size, distribution, coupled=False):
        super().__init__(episode_size, distribution)
        self.coupled = coupled
    
    def grad_samples(self, params):
        self._to_backward(params)

        # TODO: Only supporting MultivariateNormal right now
        mean, cov = params

        # TODO: Sample for each dimension in mean
        pos_samples = torch.Tensor(weibull(sqrt(2.), (self.episode_size, len(mean))))
        if self.coupled:
            neg_samples = pos_samples
        else:
            neg_samples = torch.Tensor(weibull(sqrt(2.), (self.episode_size, len(mean))))
        samples = torch.cat((pos_samples, -neg_samples))

        if len(mean) > 1:
            l = torch.cholesky(cov)
            return samples @ l + mean
        else:
            std = sqrt(cov)
            return samples * std + mean

    def backward(self, losses):
        # TODO: Only supporting MultivariateNormal right now
        # TODO: Sample for each dimension in mean
        mean, cov = self._from_forward()
        pos_losses, neg_losses = torch.split(losses, len(losses) // 2)
        if len(mean) > 1:
            c = torch.inverse(sqrt(2 * pi) * cov)
            grad_estimate = ((pos_losses - neg_losses) @ c).mean(dim=0)
        else:
            c = 1 / (sqrt(2 * pi) * cov)
            grad_estimate = ((pos_losses - neg_losses) * c).mean(dim=0)
        return mean.backward(grad_estimate)
