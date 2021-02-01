import torch
import torch.distributions

from . import base


class MultivariateNormalReinforce(base.MultivariateNormalProbabilistic):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        self.samples = None

    def grad_samples(self, params):
        self.samples = self.sample(params, self.sample_size, with_grad=False)
        return self.samples

    def backward(self, params, losses):
        if self.samples is None:
            raise ValueError("No forward call or multiple backward calls.")
        log_probs = self.log_prob(params, self.samples)
        assert losses.shape == log_probs.shape
        (losses * log_probs).mean().backward()
        self.samples = None

    @staticmethod
    def log_prob(params, samples):
        mean, log_std = params
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(torch.exp(2 * log_std)))
        return dist.log_prob(samples)


class ExponentialReinforce(base.ExponentialProbabilistic):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        self.samples = None

    def grad_samples(self, params):
        self.samples = self.sample(params, self.sample_size, with_grad=False)
        return self.samples

    def backward(self, params, losses):
        if self.samples is None:
            raise ValueError("No forward call or multiple backward calls.")
        log_probs = self.log_prob(params, self.samples)
        assert losses.shape == log_probs.shape
        (losses * log_probs).mean().backward()
        self.samples = None

    @staticmethod
    def log_prob(params, samples):
        log_rate, = params
        return torch.distributions.Exponential(torch.exp(log_rate)).log_prob(samples)


class PoissonReinforce(base.PoissonProbabilistic):
    def __init__(self, sample_size):
        super().__init__(sample_size)
        self.samples = None

    def grad_samples(self, params):
        self.samples = self.sample(params, self.sample_size, with_grad=False)
        return self.samples

    def backward(self, params, losses):
        if self.samples is None:
            raise ValueError("No forward call or multiple backward calls.")
        log_probs = self.log_prob(params, self.samples)
        assert losses.shape == log_probs.shape
        (losses * log_probs).mean().backward()
        self.samples = None

    @staticmethod
    def log_prob(params, samples):
        log_rate, = params
        return torch.distributions.Poisson(torch.exp(log_rate)).log_prob(samples)
