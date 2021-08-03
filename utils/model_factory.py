from pathlib import Path
from typing import List, Iterable, Tuple

import torch

from models import vae, vae_rnn
from utils.distribution_factory import get_distribution_type

vaes = {
    'fc': (vae.FCEncoder, vae.FCDecoder),
    'conv2d': (vae.Conv2DEncoder, vae.Conv2DDecoder)
}

vaes_rnn = {
    'fc': (vae_rnn.FCEncoderRNN, vae_rnn.FCDecoderRNN),
}

def get_vae(vae_type: str, data_dims: Tuple[int], hidden_dims: List[int], distribution: str, latent_dim: int, *_, **__):
    vae_type = vae_type.lower()
    if vae_type not in vaes:
        raise ValueError(f'Model {vae_type} not available. Available VAE types are `{(k for k in vaes.keys())}`.')
    encoder, decoder = vaes[vae_type]
    distribution_type = get_distribution_type(distribution)
    return vae.VAE(encoder(data_dims, hidden_dims, distribution_type.param_dims(latent_dim)),
                   decoder(latent_dim, hidden_dims[::-1], data_dims), distribution_type)


def get_vae_rnn(vae_type: str, data_dims: Tuple[int], hidden_dim: int, rnn_n_layers: int, distribution: str,
                latent_dim: int, rnn_block: str, dropout: bool, bidirectional: bool, sequence_length: int,
                batch_size: int, rnn_nonlinearity: 'str', device: 'str', *_, **__):
    vae_type = vae_type.lower()
    if vae_type not in vaes_rnn:
        raise ValueError(f'Model {vae_type} not available. Available VAE types are `{(k for k in vaes_rnn.keys())}`.')
    encoder, decoder = vaes_rnn[vae_type]
    distribution_type = get_distribution_type(distribution)
    return vae_rnn.VAERNN(encoder(data_dims, hidden_dim, distribution_type.param_dims(latent_dim),
                                  rnn_n_layers, rnn_block, dropout, bidirectional),
                          decoder(data_dims, latent_dim, hidden_dim, data_dims,
                                  sequence_length, batch_size,
                                  rnn_n_layers, rnn_block, dropout, bidirectional, rnn_nonlinearity, device),
                          distribution_type)


def load_models(base_dir: Path) -> Iterable[Tuple[Path, torch.nn.Module]]:
    return [(file_path, torch.load(file_path)) for file_path in base_dir.rglob('*.pt')]
