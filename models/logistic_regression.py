import torch


class LinearLogisticRegression(torch.nn.Module):
    def __init__(self, latent_dims, probabilistic):
        super().__init__()
        self.params = torch.nn.Parameter(torch.randn((sum(latent_dims),)))
        self.probabilistic = probabilistic

    def forward(self, x):
        params = self.params.unsqueeze(0).repeat_interleave(x.size(0), dim=0)
        samples = self.probabilistic(params) if self.training else self.probabilistic.distribution.sample(params)
        return params, self.__logistic((x * samples).sum(dim=-1))

    def backward(self, params, losses):
        self.probabilistic.backward(params, losses.detach())

    @staticmethod
    def __logistic(z):
        return 1 / (1 + torch.exp(-z))
