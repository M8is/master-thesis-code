import torch


class Probabilistic(torch.nn.Module):
    def forward(self, x, objective, params):
        if self.training:
            return self.forward_mc(x, objective, params)
        return self.distribution(*params).sample()

    def forward_mc(self, x, objective, params):
        raise NotImplementedError
