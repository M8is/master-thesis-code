import torch


class LinearLogisticRegression(torch.nn.Module):
    def __init__(self, latent_dim, probabilistic):
        super().__init__()
        param_dims = probabilistic.distribution_type.param_dims(latent_dim)
        self.raw_params = torch.nn.Parameter(torch.randn((1, sum(param_dims))))
        self.probabilistic = probabilistic

    def forward(self, x):
        raw_params = self.raw_params.repeat_interleave(x.size(0), dim=0)
        distribution, samples = self.probabilistic(raw_params)
        return distribution, self.predict(samples, x)

    @staticmethod
    def predict(samples, x):
        return torch.sigmoid((x * samples).sum(dim=-1))
