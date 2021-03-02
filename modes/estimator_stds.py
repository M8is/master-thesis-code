import torch
import torch.utils.data

import mc_estimators
import models.vae
import train.vae
from utils.data_holder import DataHolder
from utils.seeds import fix_random_seed


def get_estimator_stds(seed, dataset, device, latent_dim, sample_size, learning_rate, mc_estimator, distribution,
                       batch_size, hidden_dim, **_):
    print(f'Generating standard deviation for {mc_estimator}.')
    fix_random_seed(seed)
    data_holder = DataHolder(dataset, batch_size)
    num_gradients = 100
    estimator = mc_estimators.get_estimator(mc_estimator, distribution, sample_size, device)
    encoder = models.vae.Encoder(data_holder.height * data_holder.width, hidden_dim, (latent_dim, latent_dim))
    decoder = models.vae.Decoder(data_holder.height * data_holder.width, hidden_dim, latent_dim)
    vae_network = models.vae.VAE(encoder, decoder, estimator)
    vae = train.vae.VAE(vae_network, data_holder, device, torch.optim.Adam, learning_rate)
    return vae.get_estimator_stds(num_gradients)
