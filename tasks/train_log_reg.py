import torch
import torch.utils.data

from models.logistic_regression import LogisticRegressionClassifier
from tasks.trainer import StochasticTrainer
from utils.distribution_factory import get_distribution_type
from utils.eval_util import eval_mode
from utils.tensor_holders import TensorHolder


class TrainLogReg(StochasticTrainer):
    def __init__(self, learning_rate: float, distribution: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        latent_dim = self.data_holder.dims[-1]
        self.__model = LogisticRegressionClassifier(latent_dim, get_distribution_type(distribution))
        self.__optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.test_accuracies = TensorHolder(self.results_dir, 'test_accuracies')
        self.metrics.add(self.test_accuracies)

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.__optimizer

    def loss(self, inputs: torch.Tensor, labels: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        y, y_pred = labels.double(), outputs
        # Use no reduction to get separate losses for each input
        bce = torch.nn.BCELoss(reduction='none')
        return bce(y_pred, y.expand_as(y_pred))

    def post_epoch(self, epoch: int) -> None:
        self.__add_test_accuracy()
        super().post_epoch(epoch)

    def __add_test_accuracy(self):
        with eval_mode(self.model):
            matches = []
            for x, y_true in self.data_holder.test:
                y_true = y_true.to(self.device)
                y_pred = self.model(x.to(self.device)).round().int()
                matches.append(torch.eq(y_true, y_pred))
            accuracy = torch.cat(matches).float().mean()
            self.test_accuracies.add(accuracy)
            print(f'-> Test Accuracy: {accuracy:.4f}')
