from os import path, makedirs

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.utils.data

from models.linear import LinearProbabilistic
from utils.estimator_factory import get_estimator
from utils.loss_holder import LossHolder
from utils.seeds import fix_random_seed


def train_parabola(seed, results_dir, device, epochs, sample_size, learning_rate, mc_estimator, param_dims,
                   distribution, **_):
    if not path.exists(results_dir):
        makedirs(results_dir)
    else:
        print(f"Skipping: '{results_dir}' already exists.")
        return

    fix_random_seed(seed)

    # Useful for checking dims during debugging
    batch_size = 1

    # Create model
    estimator = get_estimator(mc_estimator, distribution, sample_size, device, param_dims)
    model = LinearProbabilistic(1, estimator).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f'Training with {estimator}.')

    __try_plot_gaussian_mixture_pdf(model, device, 0)

    # Train
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    for epoch in range(1, epochs + 1):

        print(f"Epoch: {epoch}/{epochs}", flush=True)
        ones = torch.ones([batch_size, 1, 1]).to(device)
        params, x = model(ones)
        losses = polynomial(x)
        optimizer.zero_grad()
        model.backward(params, losses)
        optimizer.step()
        train_losses.add(losses.detach().mean())
        test_losses.add(__test_loss(model, device).mean())

        if epoch % 50 == 0 or epoch == epochs:
            __try_plot_gaussian_mixture_pdf(model, device, epoch)

    train_losses.plot(logscale=False)
    test_losses.plot(logscale=False)
    torch.save(model, path.join(results_dir, f'{mc_estimator}_{epochs}.pt'))


def __test_loss(model, device):
    with torch.no_grad():
        prev_mode = model.train if model.training else model.eval
        model.eval()
        ones = torch.ones([1, 1, 1]).to(device)
        params, x = model(ones)
        losses = polynomial(x)
        prev_mode()
        return losses


def __try_plot_gaussian_mixture_pdf(model, device, iterations):
    try:
        with torch.no_grad():
            params, _ = model(torch.ones([1, 1, 1]).to(device))
            params = params.cpu().squeeze().split(model.probabilistic.param_dims)
            categorical_probs = torch.softmax(params[0], dim=0)
            gaussian_components = params[1:]

        x = np.linspace(-2, 2, 200)
        for weight, gaussian in zip(categorical_probs, gaussian_components):
            mean, std = gaussian
            plt.plot(x, weight * scipy.stats.norm.pdf(x, mean, std))

        plt.plot(x, polynomial(x), linestyle='dashdot')

        plt.title(f"At {iterations} iterations.")
        plt.ylabel("Gaussian Mixture p(x)")
        plt.xlabel("x")

        plt.savefig("distribution.svg")
        plt.show()
    except Exception:
        # Doesn't work for most configurations, just ignore it then.
        pass


def polynomial(x):
    return (x - 0.1)**2
