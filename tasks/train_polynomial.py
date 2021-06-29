from os import path
from typing import List

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt

from distributions.distribution_base import Distribution
from utils.distribution_factory import get_distribution_type
from utils.estimator_factory import get_estimator
from utils.tensor_holders import TensorHolder


def train_polynomial(results_dir: str, iterations: int, sample_size: int, learning_rate: float, distribution: str,
                     mc_estimator: str, init_params: List[int], device: str = 'cpu', print_interval: int = 100,
                     **_) -> List[str]:
    train_losses = TensorHolder(results_dir, 'train_loss')
    test_losses = TensorHolder(results_dir, 'test_loss')
    raw_params = torch.nn.Parameter(torch.FloatTensor(init_params)).to(device)
    distribution_type = get_distribution_type(distribution)

    # Create model
    estimator = get_estimator(mc_estimator)
    optimizer = torch.optim.SGD(raw_params, lr=learning_rate)

    print(f'Training with {estimator}.')
    for iteration in range(0, iterations):
        distribution = distribution_type(raw_params)
        loss = polynomial(distribution.sample()).mean()
        kld = distribution.kl(raw_params)
        optimizer.zero_grad()
        kld.mean().backward()
        distribution.backward(estimator, polynomial, sample_size)
        optimizer.step()
        train_losses.add(loss + kld)
        test_losses.add(__test_loss(distribution))
        if iteration % print_interval == 0:
            torch.save(raw_params, path.join(results_dir, f'params_{iteration}.pt'))
            print(f"Iteration: {iteration}/{iterations}, Train loss: {train_losses.numpy()[-1]:.2f}, "
                  f"Test loss: {test_losses.numpy()[-1]:.2f}")
    __try_plot_pdf(distribution_type(raw_params), iterations, results_dir)
    torch.save(raw_params, path.join(results_dir, f'params_{iterations}.pt'))

    return ['train_loss', 'test_loss']


def __test_loss(distribution: Distribution, n_samples: int = 10) -> torch.Tensor:
    with torch.no_grad():
        kld = distribution.kl().mean()
        loss = polynomial(distribution.sample((n_samples,))).mean()
        return kld + loss


def __try_plot_pdf(distribution: Distribution, iterations: int, results_dir: str) -> None:
    # Plot PDF
    x, pdf = distribution.pdf()
    plt.plot(x, pdf, label="p(x)")

    # Plot loss function
    x = torch.tensor(np.linspace(-3, 3, 200))
    plt.plot(x, polynomial(x), linestyle='dashed', label="f(x)")

    plt.title(f"At {iterations} iterations.")
    plt.legend()
    plt.xlabel("x")
    plt.ylim(0, 1)

    plt.savefig(path.join(results_dir, f"distribution_{iterations}.svg"))
    plt.show()


def polynomial(x: torch.Tensor) -> torch.Tensor:
    # return (x - .25) ** 2
    return (x - torch.sign(x)) ** 2
