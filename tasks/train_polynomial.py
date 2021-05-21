from os import path, makedirs

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt

from mc_estimators.measure_valued_derivative import MVD
from models.discrete_mixture import DiscreteMixture
from models.linear import PureProbabilistic
from utils.estimator_factory import get_estimator
from utils.eval_util import eval_mode
from utils.loss_holder import LossHolder
from utils.seeds import fix_random_seed


def train_polynomial(seed, results_dir, device, epochs, sample_size, learning_rate, mc_estimator, param_dims,
                     distribution, init_params, **kwargs):
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)

    if path.exists(results_dir):
        print(f"Skipping training: '{results_dir}' already exists.")
    else:
        makedirs(results_dir)
        fix_random_seed(seed)

        model_params = torch.nn.Parameter(torch.FloatTensor(init_params))
        grad_mask = torch.FloatTensor(kwargs['grad_mask']).to(device) if 'grad_mask' in kwargs else None

        # Create model
        estimator = get_estimator(mc_estimator, distribution, sample_size, device, param_dims, **kwargs)
        model = PureProbabilistic(estimator, model_params).to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        print(f'Training with {estimator}.')

        __try_plot_pdf(model, 0, results_dir)
        for epoch in range(1, epochs + 1):
            print(f"Epoch: {epoch}/{epochs}", flush=True)
            raw_params, x = model()

            # TODO: kind of hacky workaround for now
            if isinstance(model.probabilistic, MVD):
                x.squeeze_(dim=2).squeeze_(dim=-1)
            losses = polynomial(x)
            optimizer.zero_grad()
            model.backward(raw_params, losses)
            if grad_mask is not None:
                model_params.grad *= grad_mask
            optimizer.step()
            train_losses.add(losses.detach().mean())
            test_losses.add(__test_loss(model).mean())
            if epoch % 100 == 0 or epoch == epochs:
                __try_plot_pdf(model, epoch, results_dir)
        torch.save(model, path.join(results_dir, f'{mc_estimator}_{epochs}.pt'))

    return train_losses, test_losses


def __test_loss(model):
    with eval_mode(model):
        # TODO: sample more to get a mean test loss? One sample is very noisy...
        raw_params, x = model()
        losses = polynomial(x)
        return losses


def __try_plot_pdf(model, iterations, results_dir):
    try:
        with eval_mode(model):
            raw_params, _ = model()

            if isinstance(model.probabilistic, DiscreteMixture):
                x, pdf = None, 0
                for i, (weight, c_raw_params) in enumerate(zip(raw_params[0].cpu().T, raw_params[1].cpu().transpose(0, 1))):
                    x, c_pdf = model.probabilistic.component.distribution.pdf(c_raw_params)
                    c_pdf = weight * c_pdf
                    pdf = c_pdf + pdf
                    if c_raw_params.size(-1) == 1:
                        plt.plot(x, c_pdf, linestyle='dotted', linewidth=.5, label=f"C{i}; C.prob. {float(weight):.3f}")
                if raw_params[1].size(-1) == 1:
                    plt.plot(x, pdf, label="p(x)")
                    plt.ylim(0, 1)
                    plt.xlim(-3, 3)
                    x = torch.tensor(np.linspace(-3, 3, 200))
                    plt.plot(x, polynomial(x), linestyle='dotted', label="f(x)")
                else:
                    x, y = x
                    fig = plt.figure().gca(projection='3d')
                    fig.plot_surface(x, y, pdf.numpy(), cmap='coolwarm', label="p(x)")
                    f = polynomial(torch.stack((x, y), dim=-1)).numpy()
                    fig.contourf(x, y, f, label="f(x)")
            else:
                x, pdf = model.probabilistic.distribution.pdf(raw_params)
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
        # Doesn't work for most configurations, just ignore it then.
        print(f"Exception plotting PDF: {e}")
        import traceback
        traceback.print_exc()


def polynomial(x):
    #return (x - .25) ** 2
    return (x - torch.sign(x)) ** 2

