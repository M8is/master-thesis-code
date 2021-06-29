from functools import reduce
from operator import mul
from pathlib import Path
from typing import Type, Tuple, Iterable
import torch
from torch import nn
from torchvision.utils import save_image

from distributions.distribution_base import Distribution
from models.stochastic_model import StochasticModel
from utils.data_holder import DataHolder
from utils.eval_util import eval_mode


class VAE(StochasticModel):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, distribution_type: Type[Distribution]):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.distribution_type = distribution_type

    def encode(self, data: torch.Tensor) -> Distribution:
        return self.distribution_type(self.encoder(data))

    def interpret(self, sample: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return self.decoder(sample)

    def generate_images(self, output_dir: Path, data_holder: DataHolder, limit: int) -> None:
        with eval_mode(self):
            output_dir.mkdir(exist_ok=True)
            for batch_id, (x_batch, _) in enumerate(data_holder.test):
                x_batch = x_batch.to(next(self.parameters()).device)
                x_pred_batch = self(x_batch)
                comparison = torch.cat((x_batch, x_pred_batch.view(x_batch.shape)))
                save_image(comparison, output_dir / f'recon_{batch_id}.png', nrow=x_batch.size()[0])
                if batch_id == limit:
                    break


class FCEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], hidden_dims: Iterable[int], output_sizes: Iterable[int]):
        super().__init__()
        self.input_shape = input_shape
        self.layers = nn.ModuleList([])
        in_features = reduce(mul, input_shape)
        for h_dim in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.LeakyReLU())
            )
            in_features = h_dim
        self.output_layer = nn.Linear(in_features, sum(output_sizes))

    def forward(self, x):
        x = x.flatten(start_dim=-len(self.input_shape))
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class FCDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Iterable[int], output_shape: Tuple[int, ...]):
        super().__init__()
        self.output_shape = output_shape

        self.layers = nn.ModuleList([])
        in_features = latent_dim
        for h_dim in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.LeakyReLU())
            )
            in_features = h_dim

        output_size = reduce(mul, output_shape)
        self.output_layer = nn.Sequential(nn.Linear(in_features,  output_size),
                                          nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x.view(*x.shape[:-1], *self.output_shape)


class Conv2DEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], hidden_dims: Iterable[int], output_sizes: Iterable[int]):
        super().__init__()

        conv_kw = {
            'kernel_size': 3,
            'stride': 2,
            'padding': 1,
        }

        layers = []
        in_channels, input_size, _ = input_shape
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, **conv_kw),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            input_size = int(1 + (input_size + 2 * conv_kw['padding'] - conv_kw['kernel_size']) / conv_kw['stride'])
        self.model = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_channels * input_size ** 2, sum(output_sizes))

    def forward(self, x):
        x = self.model(x)
        return self.output_layer(x.flatten(start_dim=1))


class Conv2DDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: Iterable[int], output_shape: Tuple[int, ...]):
        super().__init__()
        self.decoder_start_dims = (hidden_dims[0], 2, 2)
        self.input_layer = nn.Linear(latent_dim, reduce(mul, self.decoder_start_dims))

        layers = []

        conv_kw = {
            'kernel_size': 3,
            'stride': 2,
            'padding': 1,
            'output_padding': 1
        }
        in_channels, input_size, _ = self.decoder_start_dims
        for h_dim in hidden_dims[1:]:
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, h_dim, **conv_kw),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            input_size = (input_size - 1) * conv_kw['stride'] - 2 * conv_kw['padding'] + conv_kw['kernel_size'] + \
                         conv_kw['output_padding']

        data_channels, output_size, _ = output_shape
        if output_size > input_size:
            kernel_size = output_size - input_size + 1
            self.output_layer = nn.ConvTranspose2d(in_channels, data_channels, kernel_size=kernel_size)
        else:
            kernel_size = input_size - output_size + 1
            self.output_layer = nn.Conv2d(in_channels, data_channels, kernel_size=kernel_size)

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        sample_shape = x.shape[:-1]
        x = self.input_layer(x)
        x = self.decoder(x.view(-1, *self.decoder_start_dims))
        x = torch.sigmoid(self.output_layer(x))
        return x.view(*sample_shape, *x.shape[1:])
