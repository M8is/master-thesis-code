from os import path, makedirs

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

    # Create model
    estimator = get_estimator(mc_estimator, distribution, sample_size, device)
    model = LinearProbabilistic(1, param_dims, estimator).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print(f'Training with {estimator}.')

    # Train
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}", flush=True)
        ones = torch.ones([1, 1, 1]).to(device)
        params, x = model(ones)
        losses = (x - .1) ** 2
        optimizer.zero_grad()
        model.backward(params, losses)
        optimizer.step()
        train_losses.add(losses.mean())
        test_losses.add(__test_loss(model, device).mean())
    train_losses.plot(logscale=False)
    test_losses.plot(logscale=False)
    torch.save(model, path.join(results_dir, f'{mc_estimator}_{epochs}.pt'))


def __test_loss(model, device):
    prev_mode = model.train if model.training else model.eval
    model.eval()
    with torch.no_grad():
        ones = torch.ones([10000, 1, 1]).to(device)
        params, x = model(ones)
        losses = (x - .1) ** 2
    prev_mode()
    return losses
