from pathlib import Path
from typing import List, Iterable, Tuple

import torch

from models import vae
from utils.distribution_factory import get_distribution_type

vaes = {
    'fc': (vae.FCEncoder, vae.FCDecoder),
    'conv2d': (vae.Conv2DEncoder, vae.Conv2DDecoder)
}


def get_vae(vae_type: str, data_dims: List[int], hidden_dims: List[int], distribution: str, latent_dim: int, *_, **__):
    vae_type = vae_type.lower()
    if vae_type not in vaes:
        raise ValueError(f'Model {vae_type} not available. Available VAE types are `{(k for k in vaes.keys())}`.')
    encoder, decoder = vaes[vae_type]
    distribution_type = get_distribution_type(distribution)
    return vae.VAE(encoder(data_dims, hidden_dims, distribution_type.param_dims(latent_dim)),
                   decoder(latent_dim, hidden_dims[::-1], data_dims), distribution_type)


def load_models(base_dir: Path) -> Iterable[Tuple[Path, torch.nn.Module]]:
    return [(file_path, torch.load(file_path)) for file_path in base_dir.rglob('*.pt')]
