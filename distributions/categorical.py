from enum import Enum
from typing import Any, Callable

import torch
import torch.nn.functional as F

from .distribution_base import Distribution


class CategoricalDomain(Enum):
    INTEGER = 1
    ZERO_ONE = 2


class Categorical(Distribution):
    def __init__(self, *args, domain: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.warned = False

        if CategoricalDomain(domain) == CategoricalDomain.ZERO_ONE:
            self.sample = self.with_normalized_outputs(self.sample)
            self.rsample = self.with_normalized_outputs(self.rsample)
            self.mvsample = self.with_normalized_outputs(self.mvsample)

    def _as_params(self, raw_params):
        return torch.softmax(raw_params.squeeze(), -1).unsqueeze(0)

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        return torch.distributions.Categorical(self.params).sample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size([])):
        return self.__as_index(F.gumbel_softmax(torch.log(self.params).expand((*sample_shape, *self.params.shape)),
                                                hard=True))

    def mvsample(self, size):
        if size > 1 and not self.warned:
            print("[WARNING] Categorical MVD ignores sample sizes greater than one.")
            self.warned = True
        return torch.arange(self.params.size()[-1], device=self.params.device).unsqueeze(0)

    def mvd_backward(self, losses, retain_graph):
        with torch.no_grad():
            repeated = losses.repeat_interleave(losses.size()[-1], dim=-2)
            diag = losses.diag_embed()
            grad = (2 * diag - repeated).sum(dim=-1)
        params = self.params.mean(dim=0, keepdim=True)
        assert grad.shape == params.shape, f"Grad shape {grad.shape} != params shape {params.shape}"
        params.backward(grad, retain_graph=retain_graph)

    def pdf(self):
        x = torch.arange(self.params.size(-1))
        return x, self.params.detach().squeeze(0)

    def kl(self):
        p = torch.distributions.Categorical(self.params)
        q = torch.distributions.Categorical(torch.ones_like(self.params).mean(-1, keepdim=True))
        return torch.distributions.kl_divergence(p, q)

    def log_prob(self, value):
        return torch.distributions.Categorical(self.params).log_prob(value)

    def with_normalized_outputs(self, f: Callable[[Any], torch.Tensor]) -> Callable[[Any], torch.Tensor]:
        return lambda *args, **kwargs: f(*args, **kwargs) / self.params.size()[-1]

    @staticmethod
    def __as_index(one_hot_encoding: torch.Tensor) -> torch.Tensor:
        arange = torch.arange(one_hot_encoding.size()[-1], device=one_hot_encoding.device)
        return (one_hot_encoding * arange).sum(dim=-1)
