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


def main(args):
    config_file_paths = [args.c]
    if args.cs is not None:
        with open('config/configs.yaml', 'r') as f:
            index = yaml.safe_load(f)
        config_file_paths = index[args.cs]

    for config_path in config_file_paths:
        try:
            print(f"Training with '{config_path}'.")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            train_vae(**config)
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue


def train_vae(seed, results_dir, dataset, hidden_dim, latent_dim, epochs, sample_size, learning_rate, mc_estimator,
              distribution, batch_size):
    fix_random_seed(seed)

    loss_file_path = os.path.join(results_dir, 'loss.pkl')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    else:
        if os.path.exists(loss_file_path):
            print(f"Skipping: '{loss_file_path}' already exists.")
            return

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
        train_loss, test_loss = vae.train_epoch()
        print(f"Epoch: {epoch}/{epochs}", flush=True)
        losses.add(train_loss, test_loss)
        file_name = os.path.join(results_dir, f'{mc_estimator}_{epoch}.pt')
        losses.save(loss_file_path)
        torch.save(vae_network, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('-c', default=os.path.join('configs', 'default.yaml'), help='path to config file')
    parser.add_argument('-cs', default=None, help='use a set of config files from config/configs.yaml')
    main(parser.parse_args())
