import argparse
import traceback

import torch
import torch.utils.data
import yaml

import mc_estimators
import models.vae
import train.vae
from train.seeds import fix_random_seed
from train.utils import DataHolder
import matplotlib.pyplot as plt
import os


def get_stds(seed, dataset, latent_dim, sample_size, learning_rate, mc_estimator,
             distribution, batch_size, **_):
    fix_random_seed(seed)
    grad_samples = 100
    data_holder = DataHolder(dataset, batch_size)
    grads = []
    for hidden_dim in range(1, 21):
        print(f'Generating grads for {mc_estimator}, {hidden_dim} hidden dimension(s).')
        estimator = mc_estimators.get_estimator(mc_estimator, distribution, sample_size)
        encoder = models.vae.Encoder(data_holder.height * data_holder.width, hidden_dim, (latent_dim, latent_dim))
        decoder = models.vae.Decoder(data_holder.height * data_holder.width, hidden_dim, latent_dim)
        vae_network = models.vae.VAE(encoder, decoder, estimator)
        vae = train.vae.VAE(vae_network, data_holder, optimizer=torch.optim.Adam, learning_rate=learning_rate)
        for i, grad in enumerate(vae.get_grads(grad_samples)):
            try:
                grads[i].append(grad)
            except IndexError:
                grads.append([grad])
    return grads


def plot_stds(sample_size, mc_estimator, **kwargs):
    stds = get_stds(sample_size=sample_size, mc_estimator=mc_estimator, **kwargs)
    estimator = f'{mc_estimator} {sample_size} sample(s)'
    print(f'Plotting {estimator} grad standard deviation')
    plt.xlabel('hidden dim')
    for i, std in enumerate(stds):
        x = range(1, len(std)+1)
        plt.plot(x, std, '-', label=f'param {i}')
        plt.xticks(x)
    plt.legend()

    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join('plots', f'stds_{mc_estimator}_{sample_size}.svg'))
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gradient Estimator Standard Deviation Plotting Util')
    parser.add_argument('-c', default=[], help='path to config file(s)', nargs='*')
    args = parser.parse_args()
    config_file_paths = args.c
    for config_file_path in config_file_paths:
        try:
            print(f"Reading '{config_file_path}'.")
            with open(config_file_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            plot_stds(**loaded_config)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
