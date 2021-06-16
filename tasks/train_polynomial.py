from os import path

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt

from models.discrete_mixture import DiscreteMixture
from utils.estimator_factory import get_estimator
from utils.eval_util import eval_mode
from utils.tensor_holders import LossHolder


# TODO: migrate to Trainer class (requires 'fake' data_holder)
def train_polynomial(results_dir, device, epochs, sample_size, learning_rate, mc_estimator, param_dims, distribution,
                     init_params, **kwargs):
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)

    raw_params = torch.nn.Parameter(torch.FloatTensor(init_params))
    grad_mask = torch.FloatTensor(kwargs['grad_mask']).to(device) if 'grad_mask' in kwargs else None

    # Create model
    estimator = get_estimator(mc_estimator, distribution, sample_size, device, param_dims, **kwargs)
    estimator.train()
    optimizer = torch.optim.SGD(raw_params, lr=learning_rate)

    print(f'Training with {estimator}.')

    __try_plot_pdf(raw_params, estimator, 0, results_dir)
    for epoch in range(1, epochs + 1):
        x = estimator(raw_params)
        loss = polynomial(x).mean()
        optimizer.zero_grad()
        kld = estimator.distribution.kl(raw_params)
        estimator.backward(raw_params, lambda samples: polynomial(samples))
        if loss.requires_grad:
            loss.backward()
        if grad_mask is not None:
            raw_params.grad *= grad_mask
        optimizer.step()
        train_losses.add(loss + kld)
        test_losses.add(__test_loss(raw_params, estimator))
        if epoch % 100 == 0 or epoch == epochs:
            __try_plot_pdf(raw_params, estimator, epoch, results_dir)
        print(f"Epoch: {epoch}/{epochs}, Train loss: {train_losses.numpy()[-1]:.2f}, "
              f"Test loss: {test_losses.numpy()[-1]:.2f}",
              flush=True)
    torch.save(raw_params, path.join(results_dir, f'{mc_estimator}_{epochs}.pt'))
    return train_losses, test_losses


def __test_loss(raw_params, estimator, n_samples=10):
    with eval_mode(estimator):
        losses = []
        kld = estimator.distribution.kl(raw_params)
        for _ in range(n_samples):
            loss = polynomial(estimator(raw_params))
            losses.append(loss + kld)
        return torch.tensor(losses).mean()


def __try_plot_pdf(raw_params, estimator, iterations, results_dir):
    try:
        with eval_mode(estimator):
            if isinstance(estimator, DiscreteMixture):
                x, pdf = None, 0
                for i, (weight, c_raw_params) in enumerate(zip(raw_params[0].cpu().T, raw_params[1].cpu().transpose(0, 1))):
                    x, c_pdf = estimator.component.distribution.pdf(c_raw_params)
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
                x, pdf = estimator.distribution.pdf(raw_params)
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

