from typing import List

from mc_estimators.estimator_base import MCEstimator
from models import vae

vaes = {
    'fc': (vae.FCEncoder, vae.FCDecoder),
    'conv2d': (vae.Conv2DEncoder, vae.Conv2DDecoder)
}


def get_vae(vae_type: str, estimator: MCEstimator, data_dims: List[int], hidden_dims: List[int], latent_dim: int, *_,
            **__):
    """ Instantiate a new VAE

    :param vae_type: Which VAE architecture to use. Check `vaes.keys()` for valid values.
    :param estimator: MC estimator for the probabilistic layer
    :param data_dims: Input size
    :param hidden_dims: List of hidden dimensions the model should use (mirrored in decoder)
    :param latent_dim: Size of the latent dimension
    :return: Model instance
    """
    vae_type = vae_type.lower()
    if vae_type not in vaes:
        raise ValueError(f'Model {vae_type} not available.')
    encoder, decoder = vaes[vae_type]
    return vae.VAE(encoder(data_dims, hidden_dims, estimator.distribution_type.param_dims(latent_dim)),
                   decoder(latent_dim, hidden_dims[::-1], data_dims),
                   estimator)
