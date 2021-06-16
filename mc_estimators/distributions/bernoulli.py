import torch
from .distribution_base import Distribution


class Bernoulli(Distribution):
    def _as_params(self, raw_params):
        return torch.sigmoid(raw_params)

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        return torch.distributions.Bernoulli(self.params).sample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size([])):
        return torch.distributions.Bernoulli(self.params).rsample(sample_shape)

    def mvd_sample(self, size):
        return torch.tensor([1, 0]).reshape(2, 1, 1, 1, 1).to(self.device)

    def mvd_backward(self, losses, retain_graph):
        with torch.no_grad():
            pos_losses, neg_losses = losses.mean(dim=0)
            grad = pos_losses - neg_losses
        assert grad.shape == self.params.shape, f"Grad shape {grad.shape} != params shape {self.params.shape}"
        self.params.backward(grad, retain_graph=retain_graph)

    def kl(self):
        return torch.log(torch.tensor(.5)) - .5 * (torch.log(self.params) + torch.log(1 - self.params))

    def log_prob(self, value):
        return torch.distributions.Bernoulli(self.params).log_prob(value)
