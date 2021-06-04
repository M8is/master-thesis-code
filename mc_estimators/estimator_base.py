from abc import ABC, abstractmethod
import torch


class MCEstimator(ABC, torch.nn.Module):
    def __init__(self, distribution, sample_size, *args, **kwargs):
        super().__init__()
        self.baseline = kwargs["with_baseline"] if "with_baseline" in kwargs else True
        self.distribution = distribution
        self.param_dims = distribution.param_dims
        self.sample_size = sample_size

    def forward(self, raw_params):
        with torch.no_grad():
            return self.distribution.sample(raw_params)

    def generate_stds(self, raw_params, loss_fn, zero_grad_fn, n_estimates=100):
        grads = []
        zero_grad_fn()
        for i in range(n_estimates - 1):
            raw_params.retain_grad()
            self.backward(raw_params, loss_fn, retain_graph=True)
            grads.append(raw_params.grad)
            zero_grad_fn()
        raw_params.retain_grad()
        self.backward(raw_params, loss_fn, retain_graph=False)
        grads.append(raw_params.grad)
        zero_grad_fn()
        return torch.stack(grads).flatten(start_dim=1).mean(dim=1).std(dim=0)

    @abstractmethod
    def backward(self, raw_params, loss_fn, retain_graph=False):
        pass

    def __str__(self):
        return f'{type(self).__name__} {type(self.distribution).__name__} {self.sample_size} sample(s)'
