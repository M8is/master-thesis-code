import torch

from tasks.trainer import StochasticTrainer
from utils.model_factory import get_vae_rnn


class TrainVAERNN(StochasticTrainer):
    def __init__(self, learning_rate: float, distribution: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs['sequence_length'] = self.data_holder.train.dataset.tensors[0].shape[1]
        self.__model = get_vae_rnn(data_dims=self.data_holder.dims, distribution=distribution, *args, **kwargs)
        self.__optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        # x (and x_recon) are vectors with shape (batch_size, sequence_length, input_dim)
        x, x_recon = inputs, outputs
        # Use no reduction to get separate losses for each input
        mse = torch.nn.MSELoss(reduction='none')
        n_data_dims = len(x.size()) - 2
        mse_raw = mse(x_recon, x.expand_as(x_recon)).flatten(start_dim=-n_data_dims)
        # sum over input dimension, sum over the trajectory
        # TODO: is this loss correct?
        return mse_raw.sum(-1).sum(-1)
