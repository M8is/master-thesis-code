import torch


class LinearProbabilistic(torch.nn.Module):
    def __init__(self, input_dim, probabilistic):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, sum(probabilistic.param_dims))
        self.probabilistic = probabilistic

    def forward(self, x):
        params = self.linear(x)
        return params, self.probabilistic(params)

    def backward(self, params, losses):
        self.probabilistic.backward(params, losses.detach())
