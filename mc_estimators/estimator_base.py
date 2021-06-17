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

    @abstractmethod
    def backward(self, distribution, loss_fn, retain_graph=False):
        pass
