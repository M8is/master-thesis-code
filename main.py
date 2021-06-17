import argparse
from os import path, makedirs
from shutil import copyfile

import torch
import yaml

from tasks.plotting import plot_losses, plot_estimator_variances, plot_estimator_performances
from tasks.train_log_reg import TrainLogReg
from tasks.train_polynomial import train_polynomial
from tasks.train_vae import TrainVAE
from utils.clean import clean
from utils.tensor_holders import LossHolder, TensorHolder
from utils.seeds import fix_random_seed


def main(args):
    config_path = args.CONFIG
    with open(config_path, 'r') as f:
        meta_config = yaml.safe_load(f)

    results_base_dir = path.splitext(config_path)[0].replace('config', 'results', 1)

    losses_per_task = {}
    stds_per_task = {}
    times_per_task = {}
    if 'runs' in meta_config:
        configs = [{**meta_config, **run_config} for run_config in meta_config['runs']]
    else:
        configs = [meta_config]

    for config in configs:
        config_subpath = path.join(config['mc_estimator'], str(config['sample_size']))
        for seed in config['seeds']:
            results_dir = path.join(results_base_dir, config_subpath, str(seed))
            config['results_dir'] = results_dir

            dev = 'cpu'
            if 'device' in config and torch.cuda.is_available():
                dev = config['device']
            config['device'] = torch.device(dev)

            if args.clean:
                clean(**config)

            task = config.get('task')
            train_loss = LossHolder(results_dir, train=True)
            if not train_loss.is_empty():
                print(f"Skipping training; Loading existing results from '{results_dir}'...")
                test_loss = LossHolder(results_dir, train=False)
                estimator_stds = TensorHolder(results_dir, 'estimator_stds')
                estimator_times = TensorHolder(results_dir, 'estimator_times')
            else:
                makedirs(results_dir, exist_ok=True)

                fix_random_seed(seed)
                if task == 'vae':
                    train_loss, test_loss, estimator_stds, estimator_times = TrainVAE(**config).train()
                elif task == 'logreg':
                    train_loss, test_loss, estimator_stds, estimator_times = TrainLogReg(**config).train()
                elif task == 'polynomial':
                    train_loss, test_loss = train_polynomial(**config)
                    # TODO: Remove once train_polynomial is based on Trainer class.
                    estimator_stds = TensorHolder(results_dir, 'estimator_stds')
                    estimator_times = TensorHolder(results_dir, 'estimator_times')
                else:
                    raise ValueError(f"Unknown task '{task}'.")

            if task not in losses_per_task:
                losses_per_task[task] = dict()

            if config_subpath not in losses_per_task[task]:
                losses_per_task[task][config_subpath] = config, [(train_loss.numpy(), test_loss.numpy())]
            else:
                losses_per_task[task][config_subpath][1].append((train_loss.numpy(), test_loss.numpy()))

            if not estimator_stds.is_empty():
                if task not in stds_per_task:
                    stds_per_task[task] = dict()

                if config_subpath not in stds_per_task[task]:
                    stds_per_task[task][config_subpath] = config, [estimator_stds.numpy()]
                else:
                    stds_per_task[task][config_subpath][1].append(estimator_stds.numpy())

            if not estimator_times.is_empty():
                if task not in times_per_task:
                    times_per_task[task] = dict()

                if config_subpath not in times_per_task[task]:
                    times_per_task[task][config_subpath] = config, [estimator_times.numpy()]
                else:
                    times_per_task[task][config_subpath][1].append(estimator_times.numpy())

        # Copy config file to results dir
        config_results_path = path.join(results_base_dir, path.basename(config_path))
        makedirs(results_base_dir, exist_ok=True)
        copyfile(config_path, config_results_path)

    plot_losses(results_base_dir, losses_per_task)
    plot_estimator_variances(results_base_dir, stds_per_task)
    plot_estimator_performances(results_base_dir, times_per_task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('CONFIG', help='path to config file')
    parser.add_argument('--clean', action='store_true',
                        help='WARNING: deletes all result directories and starts a clean run')
    main(parser.parse_args())
