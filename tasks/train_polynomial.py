from os import path
from typing import List

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt

from utils.estimator_factory import get_estimator
from utils.tensor_holders import TensorHolder


def train_polynomial(results_dir, iterations, sample_size, learning_rate, mc_estimator, param_dims, distribution,
                     init_params, device='cpu', print_interval=100, **kwargs) -> List[str]:
    train_losses = TensorHolder(results_dir, 'train_loss')
    test_losses = TensorHolder(results_dir, 'test_loss')

    device = torch.device(device)

    raw_params = torch.nn.Parameter(torch.FloatTensor(init_params))

    # Create model
    estimator = get_estimator(mc_estimator, distribution, sample_size, device, param_dims, **kwargs)
    estimator.train()
    optimizer = torch.optim.SGD(raw_params, lr=learning_rate)

    print(f'Training with {estimator}.')
    for iteration in range(0, iterations):
        distribution, x = estimator(raw_params)
        loss = polynomial(x).mean()
        kld = distribution.kl(raw_params)
        optimizer.zero_grad()
        estimator.backward(raw_params, lambda samples: polynomial(samples))
        if loss.requires_grad:
            loss.backward()
        optimizer.step()
        train_losses.add(loss + kld)
        test_losses.add(__test_loss(distribution))
        if iteration % print_interval == 0:
            __try_plot_pdf(distribution, iteration, results_dir)
            print(f"Iteration: {iteration}/{iterations}, Train loss: {train_losses.numpy()[-1]:.2f}, "
                  f"Test loss: {test_losses.numpy()[-1]:.2f}")
    distribution, _ = estimator(raw_params)
    __try_plot_pdf(distribution, iterations, results_dir)
    torch.save(raw_params, path.join(results_dir, f'{mc_estimator}_{iterations}.pt'))

    return ['train_loss', 'test_loss']


def __test_loss(distribution, n_samples=10):
    with torch.no_grad():
        losses = []
        kld = distribution.kl()
        for _ in range(n_samples):
            loss = polynomial(distribution.sample(1, False))
            losses.append(loss + kld)
        return torch.tensor(losses).mean()


def __try_plot_pdf(distribution, iterations, results_dir):
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


def polynomial(x):
    # return (x - .25) ** 2
    return (x - torch.sign(x)) ** 2
