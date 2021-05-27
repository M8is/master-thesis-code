import numpy as np
from matplotlib import pyplot as plt
from os import path, makedirs


def plot_losses(results_dir, losses_per_task):
    for task, configs_and_losses in losses_per_task.items():
        plot_dir = path.join(results_dir, 'plots', task)
        if not path.exists(plot_dir):
            makedirs(plot_dir)

        print(f"Plotting losses for task '{task}' in {results_dir} ...")
        for config, losses in configs_and_losses.values():
            train_losses, _ = zip(*losses)
            __plot(np.stack(train_losses), **config)
        plt.legend()
        plt.savefig(path.join(plot_dir, 'train.png'))
        plt.clf()

        for config, losses in configs_and_losses.values():
            _, test_losses = zip(*losses)
            __plot(np.stack(test_losses), **config)
        plt.legend()
        plt.savefig(path.join(plot_dir, 'train.png'))
        plt.clf()


def __plot(losses, logscale=False, **kwargs):
    if logscale:
        plt.yscale('log')
    mean = losses.mean(axis=0)
    std = losses.std(axis=0)
    plt.plot(mean, label=kwargs.get('plot_label', kwargs['mc_estimator']))
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=.25)
