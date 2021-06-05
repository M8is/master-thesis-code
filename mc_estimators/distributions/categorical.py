import torch
import torch.nn.functional as F

from .distribution_base import Distribution


class Categorical(Distribution):
    def _get_param_dims(self, output_dim):
        raise NotImplemented

    def sample(self, raw_params, size=1, with_grad=False):
        probs = self.__as_probs(raw_params.squeeze())
        if with_grad:
            log_probs = torch.log(probs)
            # TODO: test this
            return torch.argmax(F.gumbel_softmax(log_probs)), probs
        else:
            return torch.distributions.Categorical(probs).sample((size,)), probs

    def mvd_sample(self, raw_params, size):
        if size > 1:
            print("Warning: Categorical MVD ignores sample sizes greater than one.")
        params = self.__as_probs(raw_params)
        return torch.arange(params.size(-1), device=self.device)

    def mvd_backward(self, raw_params, losses, retain_graph):
        params = self.__as_probs(raw_params)
        with torch.no_grad():
            while losses.shape != params.shape and len(losses.shape) > len(params.shape):
                losses = losses.mean(dim=-1)
            grad = losses - losses[:, -1]
        assert grad.shape == params.shape, f"Grad shape {grad.shape} != params shape {params.shape}"
        params.backward(grad, retain_graph=retain_graph)
        return grad

    def pdf(self, raw_params):
        params = self.__as_probs(raw_params)
        x = torch.tensor(range(params.size(-1)))
        return x, self.__as_probs(params).squeeze(0)

    def kl(self, raw_params):
        params = self.__as_probs(raw_params)
        p = torch.distributions.Categorical(params)
        q = torch.distributions.Categorical(torch.ones_like(params).mean(-1, keepdim=True))
        return torch.distributions.kl_divergence(p, q)

    def log_prob(self, raw_params, samples):
        params = self.__as_probs(raw_params)
        return torch.distributions.Categorical(params).log_prob(samples)

    @staticmethod
    def __as_probs(raw_params):
        return torch.softmax(raw_params, -1).unsqueeze(0)
