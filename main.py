import argparse
import os

import numpy as np
import torch
import torch.utils.data
import yaml

import mc_estimators
import models.vae
import train.vae


def main(seed, results_dir, dataset, hidden_dim, latent_dim, epochs, sample_size, learning_rate, mc_estimator,
         distribution):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Load data
    data_holder = train.vae.DataHolder()
    data_holder.load_datasets(dataset)

    # Create model
    estimator = mc_estimators.get_estimator(mc_estimator, distribution, sample_size)
    encoder = models.vae.Encoder(data_holder.height * data_holder.width, hidden_dim, (latent_dim, latent_dim))
    decoder = models.vae.Decoder(data_holder.height * data_holder.width, hidden_dim, latent_dim)
    vae_network = models.vae.VAE(encoder, decoder, estimator)

    # Train
    vae = train.vae.VAE(vae_network, data_holder, optimizer=torch.optim.Adam, learning_rate=learning_rate)
    losses = train.vae.LossHolder()
    for epoch in range(1, epochs+1):
        train_loss = vae.train_epoch()
        test_loss = vae.test_epoch()
        print(f"===> Epoch: {epoch}/{epochs}")
        print(f"     Train Loss: {train_loss:.3f}")
        print(f"     Test Loss: {test_loss:.3f}", flush=True)
        losses.add(train_loss, test_loss)
        file_name = os.path.join(results_dir, f'{mc_estimator}_{epoch}.pt')
        torch.save(vae_network, file_name)
    losses.save(os.path.join(results_dir, 'loss.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Variational Auto Encoder')
    parser.add_argument('-c', default=os.path.join('configs', 'default.yaml'), help='path to config file')
    args = parser.parse_args()
    with open(args.c, 'r') as f:
        config = yaml.safe_load(f)
    main(**config)
