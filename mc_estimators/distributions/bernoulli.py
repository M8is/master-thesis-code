import torch
from .distribution_base import Distribution


class Bernoulli(Distribution):
    def sample(self, raw_params, size=1, with_grad=False):
        params = self.__as_prob(raw_params)
        return torch.distributions.Bernoulli(params).sample((size,))

    def mvd_sample(self, raw_params, size):
        return torch.tensor([1, 0]).reshape(2, 1, 1, 1, 1).to(self.device)

    def mvd_backward(self, raw_params, losses, retain_graph):
        prob = self.__as_prob(raw_params)
        with torch.no_grad():
            pos_losses, neg_losses = losses.mean(dim=0)
            grad = pos_losses - neg_losses
        assert grad.shape == prob.shape, f"Grad shape {grad.shape} != params shape {prob.shape}"
        prob.backward(grad, retain_graph=retain_graph)

    def kl(self, raw_params):
        p = self.__as_prob(raw_params)
        return torch.log(torch.tensor(.5)) - .5 * (torch.log(p) + torch.log(1 - p))

    def log_prob(self, raw_params, samples):
        params = self.__as_prob(raw_params)
        return torch.distributions.Bernoulli(params).log_prob(samples)

    @staticmethod
    def __as_prob(raw_params):
        return torch.sigmoid(raw_params)
