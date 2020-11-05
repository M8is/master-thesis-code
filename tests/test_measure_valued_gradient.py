import unittest

import torch
from torch import nn

from mc_estimators import measure_valued_gradient


class TestMVD(unittest.TestCase):
    def test_squared_to_zero_1d(self):
        def f(z): return z**2

        mean = nn.Linear(1, 1)
        normal = measure_valued_gradient.MVD(f, 100, 1)

        optimizer = torch.optim.SGD(mean.parameters(), 1e-2)

        x = torch.ones(1)
        for episode in range(2000):
            optimizer.zero_grad()
            normal((mean(x), 1)).backward()
            optimizer.step()

        actual_mean = mean(torch.ones(1))
        torch.allclose(actual_mean, torch.tensor([0.]), atol=1e-2)

    def test_squared_to_zero_2d(self):
        def f(z): return z**2

        mean = nn.Linear(2, 2)
        normal = measure_valued_gradient.MVD(f, 100, 2)

        optimizer = torch.optim.SGD(mean.parameters(), 1e-2)

        x = torch.ones(2)
        cov = torch.eye(2)
        for episode in range(2000):
            optimizer.zero_grad()
            normal((mean(x), cov)).backward(torch.ones(2))
            optimizer.step()

        actual_mean = mean(torch.ones(2))
        torch.allclose(actual_mean, torch.tensor([0., 0.]), atol=1e-2)
