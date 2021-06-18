import torch
import torch.utils.data

import models.logistic_regression
from utils.estimator_factory import get_estimator
from utils.trainer import Trainer


class TrainLogReg(Trainer):
    def __init__(self, learning_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        latent_dim = self.data_holder.dims[-1]
        estimator = get_estimator(latent_dim=latent_dim, *args, **kwargs)
        self.__model = models.logistic_regression.LinearLogisticRegression(latent_dim, estimator).to(self.device)
        self.__optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    @property
    def variance_interval(self):
        return 5

    @property
    def model(self):
        return self.__model

    @property
    def optimizer(self):
        return self.__optimizer

    def loss(self, inputs, outputs):
        y, y_pred = inputs, outputs
        # Use no reduction to get separate losses for each image
        binary_cross_entropy = torch.nn.BCELoss(reduction='none')
        return binary_cross_entropy(y_pred, y.expand_as(y_pred))
