import argparse
import os
import traceback

import torch
import torch.utils.data
import yaml

import mc_estimators
import models.vae
import train.vae
from train.seeds import fix_random_seed
from train.utils import DataHolder, LossHolder
from scripts.generate_images import generate_images
from scripts.plot_losses import plot_losses


def main(args):
    config_file_paths = args.c
    if args.cs is not None:
        with open('config/configs.yaml', 'r') as f:
            index = yaml.safe_load(f)
        config_file_paths.extend(index[args.cs])

    configs = []
    for config_path in config_file_paths:
        try:
            print(f"Training with '{config_path}'.")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            train_vae(**config)
            generate_images(**config)
            configs.append(config)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

    plot_losses(configs)


def train_vae(seed, results_dir, dataset, hidden_dim, latent_dim, epochs, sample_size, learning_rate, mc_estimator,
              distribution, batch_size):
    fix_random_seed(seed)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    else:
        print(f"Skipping: '{results_dir}' already exists.")
        return

    data_holder = DataHolder(dataset, batch_size)

    # Create model
    estimator = mc_estimators.get_estimator(mc_estimator, distribution, sample_size)
    encoder = models.vae.Encoder(data_holder.height * data_holder.width, hidden_dim, (latent_dim, latent_dim))
    decoder = models.vae.Decoder(data_holder.height * data_holder.width, hidden_dim, latent_dim)
    vae_network = models.vae.VAE(encoder, decoder, estimator)

    # Train
    vae = train.vae.VAE(vae_network, data_holder, optimizer=torch.optim.Adam, learning_rate=learning_rate)
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    for epoch in range(1, epochs + 1):
        train_loss, test_loss = vae.train_epoch()
        print(f"Epoch: {epoch}/{epochs}", flush=True)
        train_losses.add(train_loss)
        test_losses.add(test_loss)
        file_name = os.path.join(results_dir, f'{mc_estimator}_{epoch}.pt')
        train_losses.save()
        test_losses.save()
        torch.save(vae_network, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('-c', default=[], help='path to config file(s)', nargs='*')
    parser.add_argument('-cs', default=None, help='use a set of config files from config/configs.yaml')
    main(parser.parse_args())
