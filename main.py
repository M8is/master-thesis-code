import argparse
import traceback
from os import path

import torch
import yaml

from modes.estimator_stds import get_estimator_stds
from modes.generate_images import generate_images_vae
from modes.plot_losses import plot_losses
from modes.train_log_reg import train_log_reg
from modes.train_vae import train_vae
from utils.clean import clean


def main(args):
    config_file_paths = args.c
    if args.cs is not None:
        with open('config/configs.yaml', 'r') as f:
            index = yaml.safe_load(f)
        config_file_paths.extend(index[args.cs])

    configs = []
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

            if args.mode == 'vae':
                train_vae(**config)
                generate_images_vae(**config)
            elif args.mode == 'logreg':
                train_log_reg(**config)
            elif args.mode == 'gradstds':
                print(get_estimator_stds(**config))

            configs.append(config)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

    if args.mode == 'vae' or args.mode == 'logreg':
        plot_losses(configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('mode', help='the main task to run', choices=['vae', 'logreg', 'gradstds'], default='vae')
    parser.add_argument('-c', default=[], help='path to config file(s)', nargs='*')
    parser.add_argument('-cs', default=None, help='use a set of config files from config/configs.yaml')
    parser.add_argument('--clean', action='store_true',
                        help='WARNING: deletes all result directories and starts a clean run')
    main(parser.parse_args())
