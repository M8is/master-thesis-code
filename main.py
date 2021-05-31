import argparse
from os import path, makedirs
from shutil import copyfile

import torch
import yaml

from tasks.plotting import plot_losses
from tasks.train_log_reg import train_log_reg
from tasks.train_polynomial import train_polynomial
from tasks.train_vae import train_vae
from utils.clean import clean
from utils.loss_holder import LossHolder
from utils.seeds import fix_random_seed


def main(args):
    if args.plot and args.clean:
        raise ValueError("Clean and plot is not allowed to avoid accidentally deleting results.")

    config_path = args.CONFIG
    with open(config_path, 'r') as f:
        meta_config = yaml.safe_load(f)

    results_base_dir = path.splitext(config_path)[0].replace('config', 'results', 1)

    # Copy config file to results dir
    config_results_path = path.join(results_base_dir, path.basename(config_path))
    if not path.exists(results_base_dir):
        makedirs(results_base_dir)
    copyfile(config_path, config_results_path)

    losses_per_task = {}
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

            task = config.get('task', None)
            if path.exists(results_dir):
                print(f"Skipping training; Loading existing results from '{results_dir}'...")
                train_loss = LossHolder(results_dir, train=True)
                test_loss = LossHolder(results_dir, train=False)
            else:
                makedirs(results_dir)

                fix_random_seed(seed)
                if task == 'vae':
                    train_loss, test_loss = train_vae(**config)
                elif task == 'logreg':
                    train_loss, test_loss = train_log_reg(**config)
                elif task == 'polynomial':
                    train_loss, test_loss = train_polynomial(**config)
                else:
                    raise ValueError(f"Unknown task '{task}'.")

            if task not in losses_per_task:
                losses_per_task[task] = dict()

            if config_subpath not in losses_per_task[task]:
                losses_per_task[task][config_subpath] = config, [(train_loss.numpy(), test_loss.numpy())]
            else:
                losses_per_task[task][config_subpath][1].append((train_loss.numpy(), test_loss.numpy()))

    if args.plot:
        plot_losses(results_base_dir, losses_per_task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('CONFIG', help='path to config file')
    parser.add_argument('--clean', action='store_true',
                        help='WARNING: deletes all result directories and starts a clean run')
    parser.add_argument('--plot', action='store_true', help='Run plotting')
    main(parser.parse_args())
