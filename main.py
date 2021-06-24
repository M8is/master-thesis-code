import argparse
import shutil
import subprocess
from os import path, makedirs
from shutil import copyfile
from typing import List, Dict, Any

import torch
import yaml

from meta_vars import META_FILE_NAME
from tasks.train_log_reg import TrainLogReg
from tasks.train_polynomial import train_polynomial
from tasks.train_vae import TrainVAE
from utils.tensor_holders import TensorHolder
from utils.seeds import fix_random_seed


def main(args):
    config_path = args.CONFIG
    with open(config_path, 'r') as f:
        meta_config = yaml.safe_load(f)

    # Update meta config with run configs, or run meta config as-is, if no runs are defined.
    runs = meta_config.pop('runs', [{}])
    configs = [{**meta_config, **run_config} for run_config in runs]

    results_base_dir = path.splitext(config_path)[0].replace('config', 'results', 1)

    if args.clean and path.exists(results_base_dir):
        shutil.rmtree(results_base_dir)
        print(f"Cleaned '{results_base_dir}'.")

    # Copy config file to results dir
    config_results_path = path.join(results_base_dir, path.basename(config_path))
    makedirs(results_base_dir, exist_ok=True)
    copyfile(config_path, config_results_path)
    training(configs, results_base_dir)


def training(configs: List[Dict[str, Any]], results_base_dir: str):
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    for i, config in enumerate(configs):
        config['revision'] = git_revision
        results_subpath = path.join(results_base_dir, *[str(config[key]) for key in config['subpath_keys']])
        with open(path.join(results_subpath, META_FILE_NAME), 'w+') as f:
            yaml.safe_dump(config, f)

        seeds = config['seeds']
        for j, seed in enumerate(seeds):
            results_dir = path.join(results_subpath, str(seed))
            config['results_dir'] = results_dir

            # Replace device string with device class
            config['device'] = torch.device(config.get('device', 'cpu'))

            task = config.get('task')
            if not TensorHolder(results_dir, 'train_loss').is_empty():
                print(f"Skipping training; Loading existing results from '{results_dir}'...")
            else:
                print(f"=== Training {i * len(seeds) + j}/{len(seeds) * len(configs)} ===")
                fix_random_seed(seed)
                if task == 'vae':
                    TrainVAE(**config).train()
                elif task == 'logreg':
                    TrainLogReg(**config).train()
                elif task == 'polynomial':
                    train_polynomial(**config)
                else:
                    raise ValueError(f"Unknown task '{task}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probabilistic Gradient Estimators')
    parser.add_argument('CONFIG', help='Path to config file')
    parser.add_argument('--clean', action='store_true',
                        help='WARNING: deletes the config directory and starts a clean run')
    main(parser.parse_args())
