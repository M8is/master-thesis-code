import torch
import torch.nn.functional as F

from .distribution_base import Distribution


class Categorical(Distribution):
    def sample(self, raw_params, size=1, with_grad=False):
        probs = self.__as_probs(raw_params.squeeze(1))
        if with_grad:
            log_probs = torch.log(probs)
            # TODO: convert one-hot back to range
            return probs, F.gumbel_softmax(log_probs)
        else:
            return probs, torch.distributions.Categorical(probs).sample((size,))

    def mvd_sample(self, raw_params, size):
        if size > 1:
            print("Warning: Categorical ignores sample sizes greater than one.")
        return self.__as_probs(raw_params.squeeze(1)), torch.tensor(range(raw_params.size(-1)),
                                                                    device=self.device).reshape(1, 1, -1)

    def mvd_grad(self, params, losses):
        losses.mean(dim=0)

        last_only = False
        if last_only:
            neg_grad = losses[:,-1]
            return losses - neg_grad
        else:
            n_classes = params.size(-1)
            neg_grad = (1 / (n_classes - 1)) * (losses.diag() - losses).T.sum(dim=-1)
            return losses - neg_grad

    def pdf(self, params):
        x = torch.tensor(range(params.size(-1)))
        return x, self.__as_probs(params).squeeze(0)

    def _mvd_constant(self, params):
        return 1.

    def kl(self, params):
        p = torch.distributions.Categorical(params)
        q = torch.distributions.Categorical(torch.ones_like(params).mean(-1, keepdim=True))
        return torch.distributions.kl_divergence(p, q)

    def log_prob(self, params, samples):
        return torch.distributions.Categorical(params).log_prob(samples)

    @staticmethod
    def __as_probs(params):
        return torch.softmax(params, -1).unsqueeze(0)
