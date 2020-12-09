import argparse
import os
import traceback

import numpy as np
import torch
import torch.utils.data
import yaml

import mc_estimators
import models.vae
import train.vae
from train.seeds import fix_random_seed
from train.utils import DataHolder, LossHolder


def main(seed, results_dir, dataset, hidden_dim, latent_dim, epochs, sample_size, learning_rate, mc_estimator,
         distribution, batch_size):
    fix_random_seed(seed)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Load data
    data_holder = DataHolder()
    data_holder.load_datasets(dataset, batch_size)

    # Create model
    estimator = mc_estimators.get_estimator(mc_estimator, distribution, sample_size)
    encoder = models.vae.Encoder(data_holder.height * data_holder.width, hidden_dim, (latent_dim, latent_dim))
    decoder = models.vae.Decoder(data_holder.height * data_holder.width, hidden_dim, latent_dim)
    vae_network = models.vae.VAE(encoder, decoder, estimator)

    # Train
    vae = train.vae.VAE(vae_network, data_holder, optimizer=torch.optim.Adam, learning_rate=learning_rate)
    losses = LossHolder()
    for epoch in range(1, epochs + 1):
        train_loss = vae.train_epoch()
        test_loss = vae.test_epoch()
        print(f"===> Epoch: {epoch}/{epochs}")
        print(f"     Train Loss: {train_loss.mean():.3f}")
        print(f"     Test Loss: {test_loss.mean():.3f}", flush=True)
        losses.add(train_loss, test_loss)
        file_name = os.path.join(results_dir, f'{mc_estimator}_{epoch}.pt')
        losses.save(os.path.join(results_dir, 'loss.pkl'))
        torch.save(vae_network, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('-c', default=os.path.join('configs', 'default.yaml'), help='path to config file')
    parser.add_argument('-cs', default=None, help='use a set of config files from config/configs.yaml')
    args = parser.parse_args()

    configs = [args.c]
    if args.cs is not None:
        with open('config/configs.yaml', 'r') as f:
            index = yaml.safe_load(f)
        configs = index[args.cs]

    for config_path in configs:
        try:
            print(f"Running '{config_path}'.")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            main(**config)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue
