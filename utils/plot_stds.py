import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data

import mc_estimators
import models.vae
import train.vae
from train.seeds import fix_random_seed
from train.utils import DataHolder


def get_stds(seed, dataset, device, latent_dim, sample_size, learning_rate, mc_estimator,
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
        vae = train.vae.VAE(vae_network, data_holder, device, torch.optim.Adam, learning_rate)
        for i, grad in enumerate(vae.get_grads(grad_samples)):
            try:
                grads[i].append(grad)
            except IndexError:
                grads.append([grad])
    return grads


def plot_stds(mc_estimator, distribution, sample_size, **kwargs):
    stds = get_stds(mc_estimator=mc_estimator, distribution=distribution, sample_size=sample_size, **kwargs)
    print(f'Plotting {mc_estimator} grad standard deviation')
    plt.xlabel('hidden dim')
    for i, std in enumerate(stds):
        x = range(1, len(std)+1)
        plt.plot(x, std, '-', label=f'param {i}')
        plt.xticks(x)
    plt.legend()

    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(os.path.join('plots', f'stds_{mc_estimator}_{distribution}_{sample_size}.svg'))
    plt.clf()
