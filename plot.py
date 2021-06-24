import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml
from matplotlib import pyplot as plt

from meta_vars import META_FILE_NAME
from os import path, makedirs

from tasks.train_vae import TrainVAE
from utils.tensor_holders import TensorHolder


def main(results_base_dir: str):
    losses_per_task = {}
    stds_per_task = {}
    times_per_task = {}

    configs = []
    for config_file_path in Path(results_base_dir).rglob(META_FILE_NAME):
        with open(config_file_path) as f:
            config = yaml.safe_load(f)
            config['config_dir'] = path.dirname(config_file_path)
            configs.append(config)

    for config in configs:
        for seed in config['seeds']:
            config_path = config['config_dir']
            results_dir = path.join(config_path, str(seed))
            config['results_dir'] = results_dir
            task = config.get('task')

            train_loss = TensorHolder(results_dir, 'train_loss')
            test_loss = TensorHolder(results_dir, 'test_loss')
            if not train_loss.is_empty() and not test_loss.is_empty():
                if task not in losses_per_task:
                    losses_per_task[task] = dict()

                if config_path not in losses_per_task[task]:
                    losses_per_task[task][config_path] = config, [(train_loss.numpy(), test_loss.numpy())]
                else:
                    losses_per_task[task][config_path][1].append((train_loss.numpy(), test_loss.numpy()))

            estimator_stds = TensorHolder(results_dir, 'estimator_stds')
            if not estimator_stds.is_empty():
                if task not in stds_per_task:
                    stds_per_task[task] = dict()

                if config_path not in stds_per_task[task]:
                    stds_per_task[task][config_path] = config, [estimator_stds.numpy()]
                else:
                    stds_per_task[task][config_path][1].append(estimator_stds.numpy())

                if task == 'vae':
                    TrainVAE(**config).generate_images()

            estimator_times = TensorHolder(results_dir, 'estimator_times')
            if not estimator_times.is_empty():
                if task not in times_per_task:
                    times_per_task[task] = dict()

                if config_path not in times_per_task[task]:
                    times_per_task[task][config_path] = config, [estimator_times.numpy()]
                else:
                    times_per_task[task][config_path][1].append(estimator_times.numpy())

    plot_losses(results_base_dir, losses_per_task)
    plot_estimator_variances(results_base_dir, stds_per_task)
    plot_estimator_performances(results_base_dir, times_per_task)


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


def __parse_label(config: Dict[str, Any], label: str):
    for k, v in config:
        label = label.replace(f'${k}$', str(v))
    return label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plotting utility')
    parser.add_argument('CONFIG', help='Path to config file')
    main(parser.parse_args().CONFIG)
