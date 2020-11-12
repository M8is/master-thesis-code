import unittest

import torch

import models.vae
import train.vae
from mc_estimators.pathwise import Pathwise
from mc_estimators.reinforce import Reinforce
from mc_estimators.measure_valued_derivative import MVD


class TestVAE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1231532)

        self.data_holder = train.vae.DataHolder()
        self.data_holder.load_datasets()

        self.latent_dim = 5
        self.hidden_dim = 10
        self.data_dim = self.data_holder.height * self.data_holder.width

        self.episode_size = 20

    def pp(self, mean):
        return mean.squeeze(), .1 * torch.eye(self.latent_dim)

    def test_vae_pathwise(self):
        encoder = models.vae.Encoder(self.data_dim, self.hidden_dim, (self.latent_dim,), post_processor=self.pp)
        decoder = models.vae.Decoder(self.data_dim, self.hidden_dim, self.latent_dim)
        vae_model = models.vae.VAE(encoder, decoder,
                                   Pathwise(self.episode_size, torch.distributions.MultivariateNormal))

        vae = train.vae.VAE(vae_model, self.data_holder, torch.optim.Adam)
        vae.train(1)

    def test_vae_reinforce(self):
        encoder = models.vae.Encoder(self.data_dim, self.hidden_dim, (self.latent_dim,), post_processor=self.pp)
        decoder = models.vae.Decoder(self.data_dim, self.hidden_dim, self.latent_dim)
        vae_model = models.vae.VAE(encoder, decoder,
                                   Reinforce(self.episode_size, torch.distributions.MultivariateNormal))

        vae = train.vae.VAE(vae_model, self.data_holder, torch.optim.Adam)
        vae.train(1)

    @unittest.skip("not yet implemented")
    def test_vae_mvd(self):
        encoder = models.vae.Encoder(self.data_dim, self.hidden_dim, (self.latent_dim,), post_processor=self.pp)
        decoder = models.vae.Decoder(self.data_dim, self.hidden_dim, self.latent_dim)
        vae_model = models.vae.VAE(encoder, decoder,
                                   MVD(self.episode_size, self.latent_dim, torch.distributions.MultivariateNormal))

        vae = train.vae.VAE(vae_model, self.data_holder, torch.optim.Adam)
        vae.train(1)


if __name__ == '__main__':
    unittest.main()
