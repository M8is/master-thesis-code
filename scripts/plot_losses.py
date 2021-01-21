import argparse
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import yaml

from train.utils import LossHolder


def plot_losses(configs):
    losses_per_estimator = load_losses_per_estimator(configs)

    for estimator, loss_holders in losses_per_estimator.items():
        train_losses = np.stack(
            [lh[0].numpy() for lh in loss_holders if len(lh[0].numpy()) == len(loss_holders[0][0].numpy())])
        plot(estimator, 'Train Loss', train_losses.mean(axis=0).flatten(), train_losses.std(axis=0).flatten())
    plt.savefig(os.path.join('plots', 'train.svg'))
    plt.clf()

    for estimator, loss_holders in losses_per_estimator.items():
        test_losses = np.stack(
            [lh[1].numpy() for lh in loss_holders if len(lh[1].numpy()) == len(loss_holders[0][1].numpy())])
        plot(estimator, 'Test Loss', test_losses.mean(axis=0).flatten(), test_losses.std(axis=0).flatten())
    plt.savefig(os.path.join('plots', 'test.svg'))
    plt.clf()


def load_losses_per_estimator(configs):
    losses_per_estimator = dict()

    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    for config in configs:
        try:
            estimator, train_losses, test_losses = load_losses(**config)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

        if estimator not in losses_per_estimator:
            losses_per_estimator[estimator] = [(train_losses, test_losses)]
        else:
            losses_per_estimator[estimator].append((train_losses, test_losses))

    return losses_per_estimator


def load_losses(results_dir, mc_estimator, sample_size, **_):
    estimator = f'{mc_estimator} {sample_size} sample(s)'
    return estimator, LossHolder(results_dir, train=True), LossHolder(results_dir, train=False)


def plot(estimator, x_label, mean, std):
    print(f"Plotting '{estimator}'.")
    x = range(len(mean))
    plt.yscale('log')
    # plt.xlim(0, 500)
    plt.xlabel(x_label)
    plt.plot(mean, '-', linewidth=.25, alpha=.8, label=estimator)
    plt.fill_between(x, mean - std, mean + std, alpha=0.3)
    plt.legend()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loss Plotting Util')
    parser.add_argument('CONFIGS', help='use a set of config files from config/configs.yaml')
    args = parser.parse_args()

    with open('config/configs.yaml', 'r') as f:
        index = yaml.safe_load(f)
    config_file_paths = index[args.CONFIGS]

    loaded_configs = []
    for config_file_path in config_file_paths:
        try:
            print(f"Reading '{config_file_path}'.")
            with open(config_file_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            loaded_configs.append(loaded_config)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

    plot_losses(loaded_configs)
