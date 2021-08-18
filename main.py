import argparse
import subprocess
from pathlib import Path

import yaml
from joblib import delayed, Parallel

from tasks import get_trainer
from utils.meta_util import save_meta_info, meta_exists
from utils.seeds import fix_random_seed


def main(args):
    config_path = Path(args.CONFIG)
    with open(config_path, 'r') as f:
        meta_config = yaml.safe_load(f)

    # Update meta config with run configs, or run meta config as-is, if no runs are defined.
    runs = meta_config.pop('runs', [{}])
    configs = [{**meta_config, **run_config} for run_config in runs]

    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    results_base_dir = Path(str(config_path.parent / config_path.stem).replace('config', 'results', 1))

    new_configs = []
    for i, config in enumerate(configs):
        results_subpath = Path(results_base_dir, *[str(config[key]) for key in config['subpath_keys']])
        seeds = config.pop('seeds') if 'seeds' in config else [config['seed']]
        for j, seed in enumerate(seeds):
            results_dir = results_subpath / str(seed)
            if not meta_exists(results_dir):
                config['seed'] = seed
                config['results_dir'] = str(results_dir)
                config['revision'] = git_revision
                new_configs.append(dict(config))
            else:
                print(f"Skipping '{results_dir}'; meta file already exists.")

    # Run seeds in parallel
    def training(config):
        print(f"=== Training {i + 1}/{len(new_configs)} ===")
        fix_random_seed(config['seed'])
        trainer = get_trainer(config['task'], config)
        trainer.train()
        config['saved_metrics'] = [m.name for m in trainer.metrics]
        save_meta_info(config)

    Parallel(n_jobs=meta_config['joblib_jobs'])(
        delayed(training)(params)
        for params in new_configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Probabilistic Gradient Estimators')
    parser.add_argument('CONFIG', help='Path to config file')
    main(parser.parse_args())
