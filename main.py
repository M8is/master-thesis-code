import argparse
import traceback
from os import path

import torch
import yaml

from tasks.plotting import plot_losses
from tasks.train_log_reg import train_log_reg
from tasks.train_polynomial import train_polynomial
from tasks.train_vae import train_vae
from utils.clean import clean


def main(args):
    config_file_paths = args.c
    if args.cs is not None:
        with open('config/configs.yaml', 'r') as f:
            index = yaml.safe_load(f)
        config_file_paths.extend(index[args.cs])

    train_losses = {}
    test_losses = {}
    for config_path in config_file_paths:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if 'results_dir' not in config:
                config['results_dir'] = path.splitext(config_path)[0]

            dev = 'cpu'
            if 'device' in config and torch.cuda.is_available():
                dev = config['device']
            config['device'] = torch.device(dev)

            if args.clean:
                clean(**config)

            task = config.get('task', None)
            if task == 'vae':
                train_loss, test_loss = train_vae(**config)
            elif task == 'logreg':
                train_loss, test_loss = train_log_reg(**config)
            elif task == 'polynomial':
                train_loss, test_loss = train_polynomial(**config)
            else:
                raise ValueError(f"Unknown task '{task}'.")

            if task not in train_losses:
                train_losses[task] = [(config, train_loss)]
                test_losses[task] = [(config, test_loss)]
            else:
                train_losses[task].append((config, train_loss))
                test_losses[task].append((config, test_loss))
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
    plot_losses(args.plots, train_losses, test_losses, args.logscale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('-c', default=[], help='path to config file(s)', nargs='*')
    parser.add_argument('-cs', default=None, help='use a set of config files from config/configs.yaml')
    parser.add_argument('-plots', default=path.join('config', 'plots'), help='where general plots will be output')
    parser.add_argument('--clean', action='store_true',
                        help='WARNING: deletes all result directories and starts a clean run')
    parser.add_argument('--logscale', action='store_true', help='plot losses with logarithmic scale')
    main(parser.parse_args())
