from os import path, makedirs

import torch
import torch.utils.data

import models.logistic_regression
from utils.data_holder import DataHolder
from utils.estimator_factory import get_estimator
from utils.eval_util import eval_mode
from utils.tensor_holders import LossHolder


def train_log_reg(results_dir, dataset, device, param_dims, epochs, sample_size, learning_rate, mc_estimator,
                  distribution, batch_size, **kwargs):
    data_holder = DataHolder.get(dataset, batch_size)

    # Create model
    estimator = get_estimator(mc_estimator, distribution, sample_size, device, param_dims, **kwargs)
    model = models.logistic_regression.LinearLogisticRegression(param_dims, estimator).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f'Training with {estimator}.')
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    for epoch in range(1, epochs + 1):
        train_loss, test_loss = __train_epoch(model, data_holder, device, optimizer)
        train_losses.add(train_loss.mean())
        test_losses.add(test_loss.mean())
        train_losses.save()
        test_losses.save()
        print(f"Epoch: {epoch}/{epochs}, Train loss: {train_losses.numpy()[-1]:.2f}, "
              f"Test loss: {test_losses.numpy()[-1]:.2f}",
              flush=True)
    torch.save(model, path.join(results_dir, f'{mc_estimator}_{epochs}.pt'))

    return train_losses, test_losses


def __train_epoch(model, data_holder, device, optimizer):
    train_losses = []
    test_losses = []
    model.train()
    for x_batch, y_batch in data_holder.train:
        raw_params, y_preds = model(x_batch.to(device))
        losses = __bce_loss(y_batch.to(device), y_preds)
        # TODO: add kl backward
        optimizer.zero_grad()
        model.backward(raw_params, losses)
        optimizer.step()
        train_losses.append(__mean_train_loss(model, data_holder, device))
        test_losses.append(__mean_test_loss(model, data_holder, device))
    return torch.stack(train_losses), torch.stack(test_losses)


def __mean_train_loss(model, data_holder, device):
    with eval_mode(model):
        test_losses = []
        for x_batch, y_batch in data_holder.train:
            raw_params, y_preds = model(x_batch.to(device))
            kl = model.probabilistic.distribution.kl(raw_params)
            losses = __bce_loss(y_batch.to(device), y_preds) + kl
            test_losses.append(losses.detach().mean())
        return torch.stack(test_losses).mean()


def __mean_test_loss(model, data_holder, device):
    with eval_mode(model):
        test_losses = []
        for x_batch, y_batch in data_holder.test:
            raw_params, y_preds = model(x_batch.to(device))
            kl = model.probabilistic.distribution.kl(raw_params)
            losses = __bce_loss(y_batch.to(device), y_preds) + kl
            test_losses.append(losses.detach().mean())
        return torch.stack(test_losses).mean()


def __bce_loss(y, y_pred):
    binary_cross_entropy = torch.nn.BCELoss(reduction='none')
    return binary_cross_entropy(y_pred, y.expand_as(y_pred).double())
