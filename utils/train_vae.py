from os import path, makedirs

import torch
import torch.utils.data

import mc_estimators
import models.vae
import train.vae
from train.seeds import fix_random_seed
from train.utils import DataHolder, LossHolder


def train_vae(seed, results_dir, dataset, device, hidden_dim, param_dims, latent_dim, epochs, sample_size,
              learning_rate, mc_estimator, distribution, batch_size):
    fix_random_seed(seed)

    if not path.exists(results_dir):
        makedirs(results_dir)
    else:
        print(f"Skipping: '{results_dir}' already exists.")
        return

    data_holder = DataHolder(dataset, batch_size)

    # Create model
    estimator = mc_estimators.get_estimator(mc_estimator, distribution, sample_size, device)
    encoder = models.vae.Encoder(data_holder.height * data_holder.width, hidden_dim, param_dims)
    decoder = models.vae.Decoder(data_holder.height * data_holder.width, hidden_dim, latent_dim)
    vae_network = models.vae.VAE(encoder, decoder, estimator)
    vae_network.to(device)

    print(f'Training with {estimator}.')

    # Train
    vae = train.vae.VAE(vae_network, data_holder, device, torch.optim.Adam, learning_rate)
    train_losses = LossHolder(results_dir, train=True)
    test_losses = LossHolder(results_dir, train=False)
    for epoch in range(1, epochs + 1):
        train_loss, test_loss = vae.train_epoch()
        print(f"Epoch: {epoch}/{epochs}", flush=True)
        train_losses.add(train_loss)
        test_losses.add(test_loss)
        file_name = path.join(results_dir, f'{mc_estimator}_{epoch}.pt')
        train_losses.save()
        test_losses.save()
        torch.save(vae_network, file_name)
