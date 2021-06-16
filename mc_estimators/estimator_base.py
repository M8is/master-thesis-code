from abc import ABC, abstractmethod

import torch


class MCEstimator(ABC, torch.nn.Module):
    def __init__(self, distribution_type, sample_size, latent_dim, device, *args, **kwargs):
        super().__init__()
        self.distribution_type = distribution_type
        self.sample_size = sample_size
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, raw_params):
        distribution = self.distribution_type(raw_params, self.device)
        return distribution, distribution.sample((1,))

    def get_std(self, distribution, zero_grad_fn, loss_fn, n_estimates=500):
        old_sample_size = self.sample_size
        self.sample_size = 1
        grads = []
        zero_grad_fn()
        for i in range(n_estimates):
            distribution.params.retain_grad()
            self.backward(distribution, loss_fn, retain_graph=(i + 1) < n_estimates)
            grads.append(distribution.params.grad)
            zero_grad_fn()
        self.sample_size = old_sample_size
        return torch.stack(grads).std(dim=0).mean()

    @abstractmethod
    def backward(self, distribution, loss_fn, retain_graph=False):
        pass
