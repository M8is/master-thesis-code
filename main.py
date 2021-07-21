import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any

import yaml

from tasks.train_log_reg import TrainLogReg
from tasks.train_polynomial import TrainPolynomial
from tasks.train_vae import TrainVAE
from utils.meta_util import save_meta_info, meta_exists
from utils.seeds import fix_random_seed


def main(args):
    config_path = Path(args.CONFIG)
    with open(config_path, 'r') as f:
        meta_config = yaml.safe_load(f)

    # Update meta config with run configs, or run meta config as-is, if no runs are defined.
    runs = meta_config.pop('runs', [{}])
    configs = [{**meta_config, **run_config} for run_config in runs]

    results_base_dir = Path(str(config_path.parent / config_path.stem).replace('config', 'results', 1))

    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    for i, config in enumerate(configs):
        config['revision'] = git_revision
        results_subpath = Path(results_base_dir, *[str(config[key]) for key in config['subpath_keys']])
        seeds = config.pop('seeds') if 'seeds' in config else [config['seed']]
        for j, seed in enumerate(seeds):
            results_dir = results_subpath / str(seed)
            if meta_exists(results_dir):
                print(f"Skipping training; meta file already exists in '{results_dir}'.")
            else:
                print(f"=== Training {i * len(seeds) + j + 1}/{len(seeds) * len(configs)} ===")
                config['seed'] = seed
                config['results_dir'] = str(results_dir)
                train(config)


def train(config: Dict[str, Any]):
    fix_random_seed(config['seed'])
    task = config['task']
    if task == 'vae':
        saved_metrics = TrainVAE(**config).train()
    elif task == 'logreg':
        saved_metrics = TrainLogReg(**config).train()
    elif task == 'polynomial':
        saved_metrics = TrainPolynomial(**config).train()
    else:
        raise ValueError(f"Unknown task '{task}'.")
    config['saved_metrics'] = saved_metrics
    save_meta_info(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probabilistic Gradient Estimators')
    parser.add_argument('CONFIG', help='Path to config file')
    main(parser.parse_args())
