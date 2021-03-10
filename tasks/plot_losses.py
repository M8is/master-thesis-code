import os
import traceback

import matplotlib.pyplot as plt
import numpy as np

from utils.loss_holder import LossHolder


def plot_losses(configs):
    losses_per_estimator = load_losses_per_results_dir(configs)

    for output_dir, loss_holders in losses_per_estimator.items():
        train_losses = np.stack(
            [lh[0].numpy() for lh in loss_holders if len(lh[0].numpy()) == len(loss_holders[0][0].numpy())])
        train_mean = train_losses.mean(axis=0).flatten()
        train_std = train_losses.std(axis=0).flatten()
        plot(train_mean, train_std, os.path.join(output_dir, 'train.svg'))

        test_losses = np.stack(
            [lh[1].numpy() for lh in loss_holders if len(lh[1].numpy()) == len(loss_holders[0][1].numpy())])
        test_mean = test_losses.mean(axis=0).flatten()
        test_std = test_losses.std(axis=0).flatten()
        plot(test_mean, test_std, os.path.join(output_dir, 'test.svg'))


def load_losses_per_results_dir(configs):
    losses_per_results_dir = dict()
    for config in configs:
        results_dir = config['results_dir']
        try:
            train_losses = LossHolder(results_dir, train=True)
            test_losses = LossHolder(results_dir, train=False)
            if results_dir not in losses_per_results_dir:
                losses_per_results_dir[results_dir] = [(train_losses, test_losses)]
            else:
                losses_per_results_dir[results_dir].append((train_losses, test_losses))
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
    return losses_per_results_dir


def plot(mean, std, file_path):
    print(f"Plotting '{file_path}'.")
    plt.yscale('log')
    plt.ylim(0, 250)
    plt.plot(mean, '-', linewidth=1., alpha=.8)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.3)
    plt.savefig(file_path)
    plt.clf()
