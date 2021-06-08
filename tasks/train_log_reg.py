from math import prod
from os import path

import torch
import torch.utils.data

import models.logistic_regression
from utils.data_holder import DataHolder
from utils.estimator_factory import get_estimator
from utils.eval_util import eval_mode
from utils.tensor_holders import LossHolder


def train_log_reg(results_dir, dataset, device, epochs, sample_size, learning_rate, mc_estimator, distribution,
                  batch_size, **kwargs):
    data_holder = DataHolder.get(dataset, batch_size)
    n_features = prod(data_holder.dims)

    # Create model
    estimator = get_estimator(mc_estimator, distribution, sample_size, device, n_features, **kwargs)
    model = models.logistic_regression.LinearLogisticRegression(sum(estimator.param_dims), estimator).to(device)
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
        if epoch % 5 == 0:
            print(f"Epoch: {epoch}/{epochs}, Train loss: {train_losses.numpy()[-1]:.2f}, "
                  f"Test loss: {test_losses.numpy()[-1]:.2f}")
    torch.save(model, path.join(results_dir, f'{mc_estimator}_{epochs}.pt'))

    return train_losses, test_losses


def __train_epoch(model, data_holder, device, optimizer):
    train_losses = []
    test_losses = []
    model.train()
    for batch_id, (x_batch, y_batch) in enumerate(data_holder.train):
        x_batch = x_batch.flatten(start_dim=1).to(device)
        y_batch = y_batch.to(device).double()
        raw_params, y_preds = model(x_batch)
        loss = __bce_loss(y_batch, y_preds).mean()
        kld = model.probabilistic.distribution.kl(raw_params).mean()
        optimizer.zero_grad()
        kld.backward(retain_graph=True)
        model.probabilistic.backward(raw_params, lambda samples: __bce_loss(y_batch, model.predict(samples, x_batch)))
        optimizer.step()
        train_losses.append(loss + kld)
    test_losses.append(__test_epoch(model, data_holder, device))
    return torch.stack(train_losses), torch.stack(test_losses)


def __test_epoch(model, data_holder, device):
    with eval_mode(model):
        test_losses = []
        for x_batch, y_batch in data_holder.test:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).double()
            raw_params, y_preds = model(x_batch)
            loss = __bce_loss(y_batch, y_preds).mean()
            kld = model.probabilistic.distribution.kl(raw_params).mean()
            test_losses.append(loss + kld)
        return torch.tensor(test_losses).mean()


def __bce_loss(y, y_pred):
    # Use no reduction to get separate losses for each image
    binary_cross_entropy = torch.nn.BCELoss(reduction='none')
    return binary_cross_entropy(y_pred, y.expand_as(y_pred))
