import argparse
import traceback
from os import path

import yaml

from utils.generate_images import generate_images
from utils.plot_losses import plot_losses
from utils.train_vae import train_vae
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

            if args.clean:
                clean(**config)

            train_vae(**config)
            generate_images(**config)
            configs.append(config)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

    plot_losses(configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('-c', default=[], help='path to config file(s)', nargs='*')
    parser.add_argument('-cs', default=None, help='use a set of config files from config/configs.yaml')
    parser.add_argument('--clean', action='store_true',
                        help='WARNING: deletes all result directories and starts a clean run')
    main(parser.parse_args())
