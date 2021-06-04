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
            train_losses = np.stack(train_losses)
            __plot(train_losses.mean(axis=0), train_losses.std(axis=0), **config)
            dataset = config['dataset']
        plt.legend()
        plt.title(f"{dataset} train loss")
        plt.savefig(path.join(plot_dir, 'train.png'))
        plt.clf()

        for config, losses in configs_and_losses.values():
            _, test_losses = zip(*losses)
            test_losses = np.stack(test_losses)
            __plot(test_losses.mean(axis=0), test_losses.std(axis=0), **config)
        plt.legend()
        plt.title(f"{dataset} test loss")
        plt.savefig(path.join(plot_dir, 'test.png'))
        plt.clf()


def plot_stds(plot_dir, stds_per_task):
    for task, configs_and_losses in stds_per_task.items():
        if not path.exists(plot_dir):
            makedirs(plot_dir)

        print(f"Plotting losses for task '{task}' in '{plot_dir}' ...")
        dataset = '<unnamed dataset>'
        for config, stds in configs_and_losses.values():
            __plot(np.stack(stds), None, **config)
            dataset = config['dataset']
        plt.legend()
        plt.title(f"{dataset} train loss")
        plt.savefig(path.join(plot_dir, 'stds.png'))
        plt.clf()


def __plot(means, stds, **kwargs):
    plt.plot(means, label=kwargs.get('plot_label', kwargs['mc_estimator']), linewidth=.5)
    if stds is not None:
        plt.fill_between(range(len(means)), means - stds, means + stds, alpha=.3)
