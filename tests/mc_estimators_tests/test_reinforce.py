import unittest

import torch
from torch import nn

from mc_estimators import reinforce


class TestReinforce(unittest.TestCase):
    def test_squared_to_zero_2d(self):
        def f(z): return (z - 5) ** 2

        mean = nn.Linear(2, 2)
        cov = nn.Linear(2, 2)
        normal = reinforce.MultivariateNormalReinforce(100)

        optimizer = torch.optim.SGD(mean.parameters(), 1e-2)

        x = torch.ones(2)
        for episode in range(2000):
            optimizer.zero_grad()
            samples = normal.grad_samples((mean(x), cov(x)))
            normal.backward(f(samples))
            optimizer.step()

        actual_mean = mean(x)
        expected_mean = 5 * torch.ones(2)
        assert torch.allclose(actual_mean, expected_mean, atol=5e-2), f"Diff is {actual_mean - expected_mean}"


if __name__ == '__main__':
    unittest.main()
