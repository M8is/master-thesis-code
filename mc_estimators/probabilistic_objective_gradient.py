from typing import Any

import torch


class Probabilistic(torch.nn.Module):
    def forward(self, x):
        if self.train:
            return self.forward_mc(x)
        return self.distribution(*x).sample()

    def forward_mc(self, params):
        raise NotImplementedError
