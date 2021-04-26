from os import path, makedirs

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt

from models.discrete_mixture import DiscreteMixture
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
    batch_size = 7

    # Create model
    estimator = get_estimator(mc_estimator, distribution, sample_size, device, param_dims)
    model = LinearProbabilistic(1, estimator, with_kl=True).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f'Training with {estimator}.')

    __try_plot_pdf(model, device, 0)

    # Train
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}", flush=True)
        ones = torch.ones([batch_size, 1, 1]).to(device)
        params, x = model(ones)
        losses = polynomial(x).mean(dim=1)  # mean over batch
        optimizer.zero_grad()
        model.backward(params, losses)
        optimizer.step()
        train_losses.add(losses.detach().mean())
        test_losses.add(__test_loss(model, device).mean())

        if epoch % 100 == 0 or epoch == epochs:
            __try_plot_pdf(model, device, epoch)

    train_losses.plot(logscale=False)
    test_losses.plot(logscale=False)
    torch.save(model, path.join(results_dir, f'{mc_estimator}_{epochs}.pt'))


def __test_loss(model, device):
    with torch.no_grad():
        prev_mode = model.train if model.training else model.eval
        model.eval()
        ones = torch.ones([100, 1, 1]).to(device)
        params, x = model(ones)
        losses = polynomial(x)
        prev_mode()
        return losses


def __try_plot_pdf(model, device, iterations):
    try:
        with torch.no_grad():
            params, _ = model(torch.ones([1, 1, 1]).to(device))
            params = params.cpu().squeeze().split(model.probabilistic.param_dims)

            if isinstance(model.probabilistic, DiscreteMixture):
                categorical_probs = model.probabilistic.selector.distribution.parameterize(params[0])
                for weight, params in zip(categorical_probs, params[1:]):
                    x, pdf = model.probabilistic.component.distribution.pdf(params)
                    plt.plot(x, weight * pdf)
            else:
                x, pdf = model.probabilistic.distribution.pdf(torch.cat(params))
                plt.scatter(x, pdf)

            x = np.linspace(-1.5, 2.5, 200)
            plt.plot(x, polynomial(x), linestyle='dotted')

            plt.title(f"At {iterations} iterations.")
            plt.ylabel("Gaussian Mixture p(x)")
            plt.xlabel("x")

            plt.ylim(0, 1)
            plt.show()
    except Exception as e:
        # Doesn't work for most configurations, just ignore it then.
        print(f"Exception plotting PDF: {e}")


def polynomial(x):
    x = x - .1  # Shift to avoid minima around zero (makes some issues more obvious).
    return x ** 2
