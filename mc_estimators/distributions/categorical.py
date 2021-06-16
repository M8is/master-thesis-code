import torch
import torch.nn.functional as F

from .distribution_base import Distribution


class Categorical(Distribution):
    def _as_params(self, raw_params):
        return torch.softmax(raw_params.squeeze(), -1).unsqueeze(0)

    def sample(self, sample_shape: torch.Size = torch.Size([])):
        return torch.distributions.Categorical(self.params).sample(sample_shape)

    def rsample(self, sample_shape: torch.Size = torch.Size([])):
        # TODO: test this
        return torch.argmax(F.gumbel_softmax(torch.log(self.params)))

    def mvd_sample(self, size):
        if size > 1:
            print("Warning: Categorical MVD ignores sample sizes greater than one.")
        return torch.arange(self.params.size(-1), device=self.device)

    def mvd_backward(self, losses, retain_graph):
        with torch.no_grad():
            while losses.shape != self.params.shape and len(losses.shape) > len(self.params.shape):
                losses = losses.mean(dim=-1)
            grad = losses - losses[:, -1]
        assert grad.shape == self.params.shape, f"Grad shape {grad.shape} != params shape {self.params.shape}"
        self.params.backward(grad, retain_graph=retain_graph)

    def pdf(self):
        x = torch.tensor(range(self.params.size(-1)))
        return x, self.params.squeeze(0)

    def kl(self):
        p = torch.distributions.Categorical(self.params)
        q = torch.distributions.Categorical(torch.ones_like(self.params).mean(-1, keepdim=True))
        return torch.distributions.kl_divergence(p, q)

    def log_prob(self, value):
        return torch.distributions.Categorical(self.params).log_prob(value)
