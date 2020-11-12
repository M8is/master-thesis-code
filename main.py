from __future__ import print_function
import torch
import torch.utils.data

from torch import optim

import models.vae
import train.vae
from mc_estimators.pathwise import Pathwise


def main():
    torch.manual_seed(423456986)

    # Load the data
    data_holder = train.vae.DataHolder()
    data_holder.load_datasets()

    # Train the Variational Auto Encoder
    encoder = models.vae.Encoder([data_holder.height, data_holder.width], 200, [20, 20])
    decoder = models.vae.Decoder([data_holder.height, data_holder.width], 200, 20)
    vae_network = models.vae.VAE(encoder, decoder, Pathwise(10, torch.distributions.Normal))

    vae = train.vae.VAE(vae_network, data_holder, optimizer=optim.Adam(encoder.parameters(), lr=1e-2))
    vae.train(100)


if __name__ == '__main__':
    main()
