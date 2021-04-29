import torch
import torch.nn.functional as F

from .distribution_base import Distribution


class Categorical(Distribution):
    def sample(self, raw_params, size=1, with_grad=False):
        probs = self.__as_probs(raw_params.squeeze())
        if with_grad:
            log_probs = torch.log(probs)
            # TODO: test this
            return probs, torch.argmax(F.gumbel_softmax(log_probs))
        else:
            return probs, torch.distributions.Categorical(probs).sample((size,))

    def mvd_sample(self, raw_params, size):
        if size > 1:
            print("Warning: Categorical MVD ignores sample sizes greater than one.")
        params = self.__as_probs(raw_params)
        return params, torch.arange(params.size(-1), device=self.device)

    def mvd_grad(self, params, losses):
        while losses.shape != params.shape and len(losses.shape) > len(params.shape):
            losses = losses.mean(dim=-1)
        assert losses.shape == params.shape

        last_only = True
        if last_only:
            return losses - losses[:, -1]
        else:
            n_classes = params.size(-1)
            neg_loss_sum_of_other_classes = (losses.diag_embed() - losses.unsqueeze(-1)).transpose(-2, -1).sum(dim=-1)
            return losses - (1 / (n_classes - 1)) * neg_loss_sum_of_other_classes

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
