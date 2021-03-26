import torch


class LinearProbabilistic(torch.nn.Module):
    def __init__(self, input_dim, param_dims, probabilistic):
        super().__init__()
        self.linear = torch.nn.ModuleList([torch.nn.Linear(input_dim, param_dim) for param_dim in param_dims])
        self.probabilistic = probabilistic

    def forward(self, x):
        params = [fc(x) for fc in self.linear]
        return params, self.probabilistic(params)

    def backward(self, params, losses):
        self.probabilistic.backward(params, losses.detach())
