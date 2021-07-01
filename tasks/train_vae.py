import torch

from tasks.trainer import StochasticTrainer
from utils.model_factory import get_vae


class TrainVAE(StochasticTrainer):
    def __init__(self, learning_rate: float, distribution: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__model = get_vae(data_dims=self.data_holder.dims, distribution=distribution, *args, **kwargs)
        self.__optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @property
    def variance_interval(self) -> int:
        return 20

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        x, x_recon = inputs, outputs
        n_data_dims = len(x.size()) - 1
        # Use no reduction to get separate losses for each image
        binary_cross_entropy = torch.nn.BCELoss(reduction='none')
        return binary_cross_entropy(x_recon, x.expand_as(x_recon)).flatten(start_dim=-n_data_dims).sum(dim=-1)
