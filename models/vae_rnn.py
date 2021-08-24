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


RNN_CELLS = {
    'rnn': nn.RNN,
    'lstm': nn.LSTM,
    'gru': nn.GRU
}


class VAERNN(StochasticModel):
    """
    Variational Recurrent Autoencoder.
    https://arxiv.org/pdf/1412.6581.pdf
    """

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


class FCEncoderRNN(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], hidden_dim: int, output_sizes: Iterable[int],
                 rnn_n_layers=1, rnn_block='rnn', dropout=0.0, bidirectional=False, rnn_nonlinearity='tanh'):
        super().__init__()
        self._rnn_block = rnn_block
        self.input_shape = input_shape
        in_features = reduce(mul, input_shape)
        if rnn_block in RNN_CELLS:
            if rnn_block in ['rnn']:
                self.rnn_cell = RNN_CELLS[rnn_block](
                    in_features, hidden_dim, rnn_n_layers,
                    dropout=dropout, bidirectional=bidirectional, nonlinearity=rnn_nonlinearity)
            elif rnn_block in ['lstm', 'gru']:
                self.rnn_cell = RNN_CELLS[rnn_block](
                    in_features, hidden_dim, rnn_n_layers,
                    dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError

        self.hidden_to_latent_params = nn.Linear(hidden_dim, sum(output_sizes))
        nn.init.xavier_uniform_(self.hidden_to_latent_params.weight)

    def forward(self, x):
        # By default our data is in shape (batch_size, sequence_length, input_dim)
        # rnn in pytorch expects shape (sequence_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)
        if isinstance(self.rnn_cell, nn.LSTM):
            h_all, (h_end, c_end) = self.rnn_cell(x)
        elif isinstance(self.rnn_cell, nn.RNN) or isinstance(self.rnn_cell, nn.GRU):
            h_all, h_end = self.rnn_cell(x)
        else:
            raise NotImplementedError

        h_end = h_end[-1, :, :]

        output = self.hidden_to_latent_params(h_end)
        return output


class FCDecoderRNN(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], latent_dim: int, hidden_dim: int, output_shape: Tuple[int, ...],
                 sequence_length: int, batch_size: int,
                 rnn_n_layers=1, rnn_block='rnn', dropout=0.0, bidirectional=False, rnn_nonlinearity='tanh',
                 device='cpu'):
        super().__init__()
        self._rnn_block = rnn_block
        self._rnn_n_layers = rnn_n_layers
        self.output_shape = output_shape
        in_features = reduce(mul, input_shape)
        if rnn_block in RNN_CELLS:
            if rnn_block in ['rnn']:
                self.rnn_cell = RNN_CELLS[rnn_block](
                    in_features, hidden_dim, rnn_n_layers,
                    dropout=dropout, bidirectional=bidirectional, nonlinearity=rnn_nonlinearity)
            elif rnn_block in ['lstm', 'gru']:
                self.rnn_cell = RNN_CELLS[rnn_block](
                    in_features, hidden_dim, rnn_n_layers,
                    dropout=dropout, bidirectional=bidirectional)
        else:
            raise ValueError

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)

        output_size = reduce(mul, output_shape)
        self.hidden_to_output = nn.Linear(hidden_dim, output_size)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

        self.decoder_inputs = torch.zeros(sequence_length, batch_size, output_size, requires_grad=True,
                                          dtype=torch.float32, device=torch.device(device))
        self.c_0 = torch.zeros(rnn_n_layers, batch_size, hidden_dim, requires_grad=True, dtype=torch.float32,
                               device=torch.device(device))

    def forward(self, latent):
        if latent.ndim == 2:
            h_state = self.latent_to_hidden(latent)
            h_0 = torch.stack([h_state for _ in range(self._rnn_n_layers)])
            if isinstance(self.rnn_cell, nn.LSTM):
                h = (h_0, self.c_0)
            elif isinstance(self.rnn_cell, nn.GRU) or isinstance(self.rnn_cell, nn.RNN):
                h = h_0
            else:
                raise NotImplementedError

            decoder_output, _ = self.rnn_cell(self.decoder_inputs, h)

            out = self.hidden_to_output(decoder_output)
            out = out.permute(1, 0, 2)
            return out
        elif latent.ndim == 3:  # TODO: this case covers pathwise and SF place it in a separate function
            N, B, D = latent.shape
            latent = latent.view(N*B, D)
            h_state = self.latent_to_hidden(latent)
            h_0 = torch.stack([h_state for _ in range(self._rnn_n_layers)])
            if isinstance(self.rnn_cell, nn.LSTM):
                c_0 = self.c_0.repeat(1, N, 1)
                h = (h_0, c_0)
            elif isinstance(self.rnn_cell, nn.GRU) or isinstance(self.rnn_cell, nn.RNN):
                h = h_0
            else:
                raise NotImplementedError

            decoder_inputs = self.decoder_inputs.repeat(1, N, 1)
            decoder_output, _ = self.rnn_cell(decoder_inputs, h)

            out = self.hidden_to_output(decoder_output)
            out = out.permute(1, 0, 2)
            NB, L, D = out.shape
            out = out.reshape(N, B, L, D)
            return out
        elif latent.ndim == 6:  # TODO: this case covers MVD
            Nparams, PosNeg, N, Dparams, B, D = latent.shape
            latent = latent.view(Nparams * PosNeg * N * Dparams * B, D)
            h_state = self.latent_to_hidden(latent)
            h_0 = torch.stack([h_state for _ in range(self._rnn_n_layers)])
            if isinstance(self.rnn_cell, nn.LSTM):
                c_0 = self.c_0.repeat(1, Nparams*PosNeg*N*Dparams, 1)
                h = (h_0, c_0)
            elif isinstance(self.rnn_cell, nn.GRU) or isinstance(self.rnn_cell, nn.RNN):
                h = h_0
            else:
                raise NotImplementedError

            decoder_inputs = self.decoder_inputs.repeat(1, Nparams*PosNeg*N*Dparams, 1)
            decoder_output, _ = self.rnn_cell(decoder_inputs, h)

            out = self.hidden_to_output(decoder_output)
            out = out.permute(1, 0, 2)
            Nparams_PosNeg_N_Dparams_B, L, D = out.shape
            out = out.reshape(Nparams, PosNeg, N, Dparams, B, L, D)
            return out
        else:
            raise NotImplementedError
