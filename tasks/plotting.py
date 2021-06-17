import numpy as np
from matplotlib import pyplot as plt
from os import path, makedirs


def plot_losses(plot_dir, losses_per_task):
    makedirs(plot_dir, exist_ok=True)

    for task, configs_and_losses in losses_per_task.items():
        print(f"Plotting losses for task '{task}' in '{plot_dir}' ...")
        dataset = '<unnamed dataset>'
        for config, losses in configs_and_losses.values():
            train_losses, _ = zip(*losses)
            train_losses = np.stack(train_losses)
            __plot(train_losses.mean(axis=0), train_losses.std(axis=0), **config)
            dataset = config['dataset']
        __legend()
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"{dataset} train loss")
        plt.savefig(path.join(plot_dir, 'train.png'))
        plt.clf()

        for config, losses in configs_and_losses.values():
            _, test_losses = zip(*losses)
            test_losses = np.stack(test_losses)
            __plot(test_losses.mean(axis=0), test_losses.std(axis=0), **config)
        __legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{dataset} test loss")
        plt.savefig(path.join(plot_dir, 'test.png'))
        plt.clf()


def plot_estimator_variances(plot_dir, stds_per_task):
    makedirs(plot_dir, exist_ok=True)

    for task, configs_and_losses in stds_per_task.items():
        print(f"Plotting variances for task '{task}' in '{plot_dir}' ...")
        dataset = '<unnamed dataset>'
        plt.yscale('log')
        for config, stds in configs_and_losses.values():
            variances = np.array(stds) ** 2
            __plot(variances.mean(axis=0), variances.std(axis=0), plot_label=config['mc_estimator'])
            dataset = config['dataset']
        __legend()
        plt.xlabel("Iterations [x10]")
        plt.ylabel("Variance")
        plt.title(f"{dataset} estimator variances")
        plt.savefig(path.join(plot_dir, 'variances.png'))
        plt.clf()


def plot_estimator_performances(plot_dir, times_per_task):
    for task, configs_and_losses in times_per_task.items():
        print(f"Writing out performances for task '{task}' in '{plot_dir}' ...")
        makedirs(plot_dir, exist_ok=True)
        with open(path.join(plot_dir, 'times_ns.yaml'), 'w+') as f:
            for config, performance in configs_and_losses.values():
                performance = np.array(performance).mean(axis=0)
                mean_perf, std_perf = performance
                print(f'{config["mc_estimator"]}:', file=f)
                print(f'  mean: {mean_perf}', file=f)
                print(f'  std: {std_perf}', file=f)


def __plot(means, stds, plot_label=None, **_):
    plt.plot(means, label=plot_label, linewidth=.5)
    if stds is not None:
        plt.fill_between(range(len(means)), means - stds, means + stds, alpha=.3)


def __legend():
    legend = plt.legend()
    plt.setp(legend.get_lines(), linewidth=2)
