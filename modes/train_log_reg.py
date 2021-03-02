from os import path, makedirs

import torch
import torch.utils.data

import mc_estimators
import models.logistic_regression
import train.logistic_regression
from utils.data_holder import DataHolder
from utils.loss_holder import LossHolder
from utils.seeds import fix_random_seed


def train_log_reg(seed, results_dir, dataset, device, param_dims, epochs, sample_size,
                  learning_rate, mc_estimator, distribution, batch_size):
    if not path.exists(results_dir):
        makedirs(results_dir)
    else:
        print(f"Skipping: '{results_dir}' already exists.")
        return

    fix_random_seed(seed)
    data_holder = DataHolder.get(dataset, batch_size)

    # Create model
    estimator = mc_estimators.get_estimator(mc_estimator, distribution, sample_size, device)
    linear = models.logistic_regression.LinearProbabilistic(param_dims, estimator)

    print(f'Training with {estimator}.')

    # Train
    log_reg = train.logistic_regression.LogisticRegression(linear, data_holder, device, torch.optim.Adam, learning_rate)
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    for epoch in range(1, epochs + 1):
        train_loss, test_loss = log_reg.train_epoch()
        print(f"Epoch: {epoch}/{epochs}", flush=True)
        train_losses.add(train_loss)
        test_losses.add(test_loss)
        train_losses.save()
        test_losses.save()
    file_name = path.join(results_dir, f'{mc_estimator}_{epochs}.pt')
    torch.save(linear, file_name)
