import argparse
import shutil
import subprocess
from os import path
from typing import List, Dict, Any

import yaml

from tasks.train_log_reg import TrainLogReg
from tasks.train_polynomial import train_polynomial
from tasks.train_vae import TrainVAE
from utils.meta_util import save_meta_info, meta_exists
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

    training(configs, results_base_dir)


def training(configs: List[Dict[str, Any]], results_base_dir: str):
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    for i, config in enumerate(configs):
        config['revision'] = git_revision
        results_subpath = path.join(results_base_dir, *[str(config[key]) for key in config['subpath_keys']])
        seeds = config.pop('seeds') if 'seeds' in config else [config['seed']]
        for j, seed in enumerate(seeds):
            results_dir = path.join(results_subpath, str(seed))
            if meta_exists(results_dir):
                print(f"Skipping training; meta file already exists in '{results_dir}'.")
            else:
                print(f"=== Training {i * len(seeds) + j}/{len(seeds) * len(configs)} ===")
                config['seed'] = seed
                fix_random_seed(seed)
                task = config['task']
                if task == 'vae':
                    saved_metrics = TrainVAE(**config, results_dir=results_dir).train()
                elif task == 'logreg':
                    saved_metrics = TrainLogReg(**config, results_dir=results_dir).train()
                elif task == 'polynomial':
                    saved_metrics = train_polynomial(**config, results_dir=results_dir)
                else:
                    raise ValueError(f"Unknown task '{task}'.")
                config['saved_metrics'] = saved_metrics
                save_meta_info(config, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probabilistic Gradient Estimators')
    parser.add_argument('CONFIG', help='Path to config file')
    parser.add_argument('--clean', action='store_true',
                        help='WARNING: deletes the config directory and starts a clean run')
    main(parser.parse_args())
