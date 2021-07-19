from enum import Enum
from typing import Any, Callable

import torch
from torch.nn.functional import gumbel_softmax

from .distribution_base import Distribution


class CategoricalDomain(Enum):
    INTEGER = 1
    ZERO_ONE = 2


class Categorical(Distribution):
    def __init__(self, *args, domain: int = 1, gumbel_temperature: float = 1., **kwargs):
        super().__init__(*args, **kwargs)
        self.warned = False
        self.gumbel_temperature = gumbel_temperature

        if CategoricalDomain(domain) == CategoricalDomain.ZERO_ONE:
            self.sample = self.with_normalized_outputs(self.sample)
            self.rsample = self.with_normalized_outputs(self.rsample)
            self.domain = torch.linspace(0, 1, self.num_categories)
        elif CategoricalDomain(domain) == CategoricalDomain.INTEGER:
            self.domain = torch.arange(self.num_categories)

    @property
    def num_categories(self):
        return self.params.shape[-1]

    def _as_params(self, raw_params):
        return torch.softmax(raw_params.squeeze(), -1).unsqueeze(0)

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        return torch.distributions.Categorical(self.params).sample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size([])):
        log_params = torch.log(self.params).expand((*sample_shape, *self.params.shape))
        return self.__as_index(gumbel_softmax(log_params, hard=True, tau=self.gumbel_temperature))

    def mvsample(self, size):
        if size > 1 and not self.warned:
            print("[WARNING] Categorical mvsample ignores sample sizes greater than one.")
            self.warned = True
        return self.domain.clone().to(self.params.device).unsqueeze(0)

    def mvd_backward(self, losses, retain_graph):
        with torch.no_grad():
            diag = losses.diag_embed()
            repeated = losses.repeat_interleave(self.num_categories, dim=-2)
            repeated.diagonal().zero_()
            repeated /= self.num_categories - 1
            grad = (diag - repeated).sum(dim=-1)
        params = self.params.mean(dim=0, keepdim=True)
        assert grad.shape == params.shape, f"Grad shape {grad.shape} != params shape {params.shape}"
        params.backward(grad, retain_graph=retain_graph)

    def pdf(self):
        return self.domain.clone(), self.params.detach().squeeze(0)

    def kl(self):
        p = torch.distributions.Categorical(self.params)
        q = torch.distributions.Categorical(torch.ones_like(self.params).mean(-1, keepdim=True))
        return torch.distributions.kl_divergence(p, q)

    def log_prob(self, value):
        return torch.distributions.Categorical(self.params).log_prob(value)

    def with_normalized_outputs(self, f: Callable[[Any], torch.Tensor]) -> Callable[[Any], torch.Tensor]:
        return lambda *args, **kwargs: f(*args, **kwargs) / self.num_categories

    @staticmethod
    def __as_index(one_hot_encoding: torch.Tensor) -> torch.Tensor:
        arange = torch.arange(one_hot_encoding.size()[-1], device=one_hot_encoding.device)
        return (one_hot_encoding * arange).sum(dim=-1)
