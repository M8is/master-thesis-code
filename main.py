import argparse
import traceback
from os import path

import torch
import yaml

from tasks.generate_images import generate_images_vae
from tasks.train_log_reg import train_log_reg
from tasks.train_polynomial import train_parabola
from tasks.train_vae import train_vae, get_estimator_stds
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

            if args.gradstds == 'gradstds':
                print(get_estimator_stds(**config))
            elif config['task'] == 'vae':
                train_vae(**config)
                generate_images_vae(**config)
            elif config['task'] == 'logreg':
                train_log_reg(**config)
            elif config['task'] == 'polynomial':
                train_parabola(**config)

            configs.append(config)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('-c', default=[], help='path to config file(s)', nargs='*')
    parser.add_argument('-cs', default=None, help='use a set of config files from config/configs.yaml')
    parser.add_argument('--clean', action='store_true',
                        help='WARNING: deletes all result directories and starts a clean run')
    parser.add_argument('--gradstds', action='store_true',
                        help='Calculate gradient standard deviations for the given configurations')
    main(parser.parse_args())
