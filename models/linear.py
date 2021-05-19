import torch


class LinearProbabilistic(torch.nn.Module):
    def __init__(self, input_dim, probabilistic, with_kl=False):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, sum(probabilistic.param_dims))
        self.probabilistic = probabilistic
        self.with_kl = with_kl

    def forward(self, x):
        raw_params = self.linear(x)
        if self.training:
            samples = self.probabilistic(raw_params)
        else:
            samples = self.probabilistic.distribution.sample(raw_params, with_grad=False)
        return raw_params, samples

    def backward(self, raw_params, losses):
        self.probabilistic.backward(raw_params, losses.detach(), retain_graph=True)
        if losses.requires_grad:
            losses.mean().backward()


class PureProbabilistic(torch.nn.Module):
    def __init__(self, probabilistic, initial_params, with_kl=False):
        super().__init__()
        self.raw_params = initial_params
        self.probabilistic = probabilistic
        self.with_kl = with_kl

    def forward(self):
        if self.training:
            samples = self.probabilistic(self.raw_params)
        else:
            samples = self.probabilistic.distribution.sample(self.raw_params, with_grad=False)
        return self.raw_params, samples

    def backward(self, raw_params, losses):
        self.probabilistic.backward(raw_params, losses.detach(), retain_graph=True)
        if losses.requires_grad:
            losses.mean().backward()
