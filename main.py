import argparse
import traceback
from os import path, makedirs

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
    config_file_paths = args.c
    losses_per_task = {}
    for config_path in config_file_paths:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config_id = config_path
            seeds = config['seeds']
            for seed in seeds:
                if 'results_dir' not in config:
                    config['results_dir'] = path.splitext(config_path)[0].replace('config', 'results')
                config['results_dir'] = path.join(config['results_dir'], str(seed))

                dev = 'cpu'
                if 'device' in config and torch.cuda.is_available():
                    dev = config['device']
                config['device'] = torch.device(dev)

                if args.clean:
                    clean(**config)

                task = config.get('task', None)
                results_dir = config['results_dir']
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

                if config_id not in losses_per_task[task]:
                    losses_per_task[task][config_id] = config, [(train_loss.numpy(), test_loss.numpy())]
                else:
                    losses_per_task[task][config_id][1].append((train_loss.numpy(), test_loss.numpy()))
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

    plot_losses('results', losses_per_task)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('-c', default=[], help='path to config file(s)', nargs='*')
    parser.add_argument('--clean', action='store_true',
                        help='WARNING: deletes all result directories and starts a clean run')
    main(parser.parse_args())
