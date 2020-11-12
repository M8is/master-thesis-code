import unittest

import torch

import models.vae
import train.vae
from mc_estimators.pathwise_gradient import Pathwise


class TestVAE(unittest.TestCase):
    def test_vae(self):
        torch.manual_seed(1231532)

        def pp(mean):
            # sigma = sigma.reshape((latent_dim, latent_dim))
            # sigma.T @ sigma
            return mean.squeeze(), .1 * torch.eye(latent_dim)

        # Load the data
        data_holder = train.vae.DataHolder()
        data_holder.load_datasets()

        # Train the Variational Auto Encoder
        latent_dim = 20
        hidden_dim = 200
        data_dim = data_holder.height * data_holder.width
        encoder = models.vae.Encoder(data_dim, hidden_dim, (latent_dim,), post_processor=pp)
        decoder = models.vae.Decoder(data_dim, hidden_dim, latent_dim)
        vae_model = models.vae.VAE(encoder, decoder, Pathwise(1000, torch.distributions.MultivariateNormal))

        vae = train.vae.VAE(vae_model, data_holder, torch.optim.Adam)
        vae.train(3)


if __name__ == '__main__':
    unittest.main()
