import torch

from .distribution_base import Distribution


class Categorical(Distribution):
    def sample(self, params, size=1, with_grad=False):
        return torch.distributions.Categorical(self.__as_probs(params)).sample((size,))

    def mvd_grad(self, params, losses):
        n_classes = params[0].size(-1)
        neg_grad = (1 / (n_classes - 1)) * (losses.T - (losses * torch.eye(losses.size(0), device=self.device))).sum(
            dim=-1)
        return (losses.T - neg_grad).unsqueeze(0)

    def mvd_sample(self, params, size):
        return torch.tensor(range(params.size(-1))).to(self.device)

    def _mvd_constant(self, params):
        return 1

    def kl(self, probs):
        probs = self.__as_probs(probs)
        p = torch.distributions.Categorical(probs)
        q = torch.distributions.Categorical(torch.ones_like(probs).mean(-1, keepdim=True))
        return torch.distributions.kl_divergence(p, q)

    def log_prob(self, params, samples):
        return torch.distributions.Categorical(self.__as_probs(params)).log_prob(samples)

    @staticmethod
    def __as_probs(params):
        return torch.softmax(params[0], -1)
