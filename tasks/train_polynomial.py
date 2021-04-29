from os import path, makedirs

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt

from models.discrete_mixture import DiscreteMixture
from models.linear import PureProbabilistic
from utils.estimator_factory import get_estimator
from utils.loss_holder import LossHolder
from utils.seeds import fix_random_seed


def train_polynomial(seed, results_dir, device, epochs, sample_size, learning_rate, mc_estimator, param_dims,
                     distribution, init_params, **kwargs):
    if not path.exists(results_dir):
        makedirs(results_dir)
    else:
        print(f"Skipping: '{results_dir}' already exists.")
        return

    fix_random_seed(seed)

    # Initial params
    initial_params = torch.nn.Parameter(torch.FloatTensor(init_params))

    # Create model
    estimator = get_estimator(mc_estimator, distribution, sample_size, device, param_dims, **kwargs)
    model = PureProbabilistic(estimator, initial_params).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f'Training with {estimator}.')

    __try_plot_pdf(model, 0, results_dir)

    # Train
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}", flush=True)
        params, x = model()
        losses = polynomial(x)
        optimizer.zero_grad()
        model.backward(params, losses)
        optimizer.step()
        train_losses.add(losses.detach().mean())
        test_losses.add(__test_loss(model).mean())
        if epoch % 100 == 0 or epoch == epochs:
            __try_plot_pdf(model, epoch, results_dir)
    train_losses.plot(logscale=False)
    test_losses.plot(logscale=False)
    torch.save(model, path.join(results_dir, f'{mc_estimator}_{epochs}.pt'))


def __test_loss(model):
    with torch.no_grad():
        prev_mode = model.train if model.training else model.eval
        model.eval()
        # TODO: sample more to get a mean test loss? One sample is very noisy...
        params, x = model()
        losses = polynomial(x)
        prev_mode()
        return losses


def __try_plot_pdf(model, iterations, results_dir):
    try:
        with torch.no_grad():
            params, _ = model()

            if isinstance(model.probabilistic, DiscreteMixture):
                x, pdf = None, 0
                for i, (weight, c_params) in enumerate(zip(params[0].cpu().T, params[1].cpu().transpose(0, 1))):
                    x, c_pdf = model.probabilistic.component.distribution.pdf(c_params)
                    c_pdf = weight * c_pdf
                    pdf = c_pdf + pdf
                    if c_params.size(-1) == 1:
                        plt.plot(x, c_pdf, linestyle='dotted', linewidth=.5, label=f"C{i}; C.prob. {float(weight):.3f}")
                if params[1].size(-1) == 1:
                    plt.plot(x, pdf, label="p(x)")
                    plt.ylim(0, 1)
                    plt.xlim(-3, 3)
                    x = torch.tensor(np.linspace(-3, 3, 200))
                    plt.plot(x, 0.3*polynomial(x), linestyle='dotted', label="f(x)")
                else:
                    x, y = x
                    fig = plt.figure().gca(projection='3d')
                    fig.plot_surface(x, y, pdf.numpy(), cmap='coolwarm', label="p(x)")
                    f = polynomial(torch.stack((x, y), dim=-1)).numpy()
                    fig.contourf(x, y, f, label="f(x)")
            else:
                x, pdf = model.probabilistic.distribution.pdf(params)
                plt.plot(x, pdf, label="p(x)")
                x = torch.tensor(np.linspace(-3, 3, 200))
                plt.plot(x, polynomial(x), linestyle='dashed', label="f(x)")

            plt.title(f"At {iterations} iterations.")
            plt.legend()
            plt.xlabel("x")
            plt.ylim(0, 1)

            plt.savefig(path.join(results_dir, f"distribution_{iterations}.svg"))
            plt.show()
    except Exception as e:
        raise
        # Doesn't work for most configurations, just ignore it then.
        print(f"Exception plotting PDF: {e}")
        import traceback
        traceback.print_exc()


def polynomial(x):
    #return (x - .25) ** 2
    return (x - torch.sign(x)) ** 2

