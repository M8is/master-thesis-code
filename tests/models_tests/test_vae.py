import unittest

import torch

import models.vae
import train.vae
from mc_estimators.pathwise import MultivariateNormalPathwise
from mc_estimators.reinforce import MultivariateNormalReinforce
from mc_estimators.measure_valued_derivative import MultivariateNormalMVD


class TestVAE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

        self.data_holder = train.vae.DataHolder()
        self.data_holder.load_datasets('mnist')

        self.latent_dim = 5
        self.hidden_dim = 10
        self.data_dim = self.data_holder.height * self.data_holder.width

    def test_vae_pathwise(self):
        encoder = models.vae.Encoder(self.data_dim, self.hidden_dim, (self.latent_dim, self.latent_dim))
        decoder = models.vae.Decoder(self.data_dim, self.hidden_dim, self.latent_dim)
        vae_model = models.vae.VAE(encoder, decoder, MultivariateNormalPathwise(sample_size=2))

        vae = train.vae.VAE(vae_model, self.data_holder, torch.optim.SGD, 1e-2)
        vae.train_epoch()
        vae.test_epoch()

    def test_vae_reinforce(self):
        encoder = models.vae.Encoder(self.data_dim, self.hidden_dim, (self.latent_dim, self.latent_dim))
        decoder = models.vae.Decoder(self.data_dim, self.hidden_dim, self.latent_dim)
        vae_model = models.vae.VAE(encoder, decoder, MultivariateNormalReinforce(sample_size=10))

        vae = train.vae.VAE(vae_model, self.data_holder, torch.optim.SGD, 1e-4)
        vae.train_epoch()
        vae.test_epoch()

    def test_vae_mvd(self):
        encoder = models.vae.Encoder(self.data_dim, self.hidden_dim, (self.latent_dim, self.latent_dim))
        decoder = models.vae.Decoder(self.data_dim, self.hidden_dim, self.latent_dim)
        vae_model = models.vae.VAE(encoder, decoder, MultivariateNormalMVD(sample_size=2))

        vae = train.vae.VAE(vae_model, self.data_holder, torch.optim.SGD, 1e-3)
        vae.train_epoch()
        vae.test_epoch()


if __name__ == '__main__':
    unittest.main()
