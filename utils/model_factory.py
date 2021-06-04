from typing import List

from mc_estimators.estimator_base import MCEstimator
from models import vae

vaes = {
    'fc': (vae.FCEncoder, vae.FCDecoder),
    'conv2d': (vae.Conv2DEncoder, vae.Conv2DDecoder)
}


def get_vae(architecture_tag: str, estimator: MCEstimator, data_dims: List[int], hidden_dims: List[int]):
    """ Instantiate a new VAE

    :param architecture_tag: Which VAE architecture to use. Check `vaes.keys()` for valid values.
    :param estimator: MC estimator for the probabilistic layer
    :param data_dims: Input size
    :param hidden_dims: List of hidden dimensions the model should use (mirrored in decoder)
    :return: Model instance
    """
    architecture_tag = architecture_tag.lower()
    if architecture_tag not in vaes:
        raise ValueError(f'Model {architecture_tag} not available.')
    encoder, decoder = vaes[architecture_tag]
    return vae.VAE(encoder(data_dims, hidden_dims, sum(estimator.param_dims)),
                   decoder(estimator.distribution.latent_dim, hidden_dims[::-1], data_dims),
                   estimator)
