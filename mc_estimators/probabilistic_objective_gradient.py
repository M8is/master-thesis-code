import torch


class Probabilistic(torch.nn.Module):
    def forward(self, objective, params):
        if self.training:
            return self.forward_mc(objective, params)
        return self.distribution(*params).sample()

    def forward_mc(self, objective, params):
        raise NotImplementedError
