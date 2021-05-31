import numpy as np
from matplotlib import pyplot as plt
from os import path, makedirs


def plot_losses(plot_dir, losses_per_task):
    for task, configs_and_losses in losses_per_task.items():
        if not path.exists(plot_dir):
            makedirs(plot_dir)

        print(f"Plotting losses for task '{task}' in '{plot_dir}' ...")
        dataset = '<unnamed dataset>'
        for config, losses in configs_and_losses.values():
            train_losses, _ = zip(*losses)
            __plot(np.stack(train_losses), **config)
            dataset = config['dataset']
        plt.legend()
        plt.title(f"{dataset} train loss")
        plt.savefig(path.join(plot_dir, 'train.png'))
        plt.clf()

        for config, losses in configs_and_losses.values():
            _, test_losses = zip(*losses)
            __plot(np.stack(test_losses), **config)
        plt.legend()
        plt.title(f"{dataset} test loss")
        plt.savefig(path.join(plot_dir, 'test.png'))
        plt.clf()


def __plot(losses, **kwargs):
    plt.ylim(100, 220)
    mean = losses.mean(axis=0)
    std = losses.std(axis=0)
    plt.plot(mean, label=kwargs.get('plot_label', kwargs['mc_estimator']), linewidth=.5)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=.3)
