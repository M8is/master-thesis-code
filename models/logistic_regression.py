import torch


class LinearLogisticRegression(torch.nn.Module):
    def __init__(self, params_size, probabilistic):
        super().__init__()
        self.raw_params = torch.nn.Parameter(torch.randn((1, params_size)))
        self.probabilistic = probabilistic

    def forward(self, x):
        raw_params = self.raw_params.repeat_interleave(x.size(0), dim=0)
        samples = self.probabilistic(raw_params)
        return raw_params, self.predict(samples, x)

    @staticmethod
    def predict(samples, x):
        return torch.sigmoid((x * samples).sum(dim=-1))
