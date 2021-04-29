import torch


class LinearProbabilistic(torch.nn.Module):
    def __init__(self, input_dim, probabilistic, with_kl=False):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, sum(probabilistic.param_dims))
        self.probabilistic = probabilistic
        self.with_kl = with_kl

    def forward(self, x):
        return self.probabilistic(self.linear(x))

    def backward(self, params, losses):
        self.probabilistic.backward(params, losses.detach(), retain_graph=True)
        if losses.requires_grad:
            losses.mean().backward()


class PureProbabilistic(torch.nn.Module):
    def __init__(self, probabilistic, initial_params, with_kl=False):
        super().__init__()
        self.raw_params = initial_params
        self.probabilistic = probabilistic
        self.with_kl = with_kl

    def forward(self):
        return self.probabilistic(self.raw_params)

    def backward(self, params, losses):
        self.probabilistic.backward(params, losses.detach(), retain_graph=True)
        if losses.requires_grad:
            losses.mean().backward()
