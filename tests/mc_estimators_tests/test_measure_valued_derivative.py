import unittest

import torch
from torch import nn

from mc_estimators import measure_valued_derivative


class TestMultivariateNormalMVD(unittest.TestCase):
    def test_squared_to_zero_2d(self):
        def f(z): return z**2

        mean = nn.Linear(2, 2)
        cov = nn.Linear(2, 2)
        normal = measure_valued_derivative.MultivariateNormalMVD(100, 2)

        optimizer = torch.optim.SGD(mean.parameters(), 1e-2)

        x = torch.ones(2).unsqueeze(0)
        for episode in range(2000):
            optimizer.zero_grad()
            samples = normal.grad_samples((mean(x), cov(x)))
            normal.backward(f(samples))
            optimizer.step()

        actual_mean = mean(torch.ones(2))
        torch.allclose(actual_mean, torch.tensor([0., 0.]), atol=1e-2)


if __name__ == '__main__':
    unittest.main()
