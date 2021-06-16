import torch
import torch.utils.data

import models.logistic_regression
from utils.estimator_factory import get_estimator
from utils.trainer import Trainer


class TrainLogReg(Trainer):
    def __init__(self, sample_size, learning_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        estimator_tag = kwargs['mc_estimator']
        distribution_tag = kwargs['distribution']
        n_features = self.data_holder.dims[-1]
        estimator = get_estimator(estimator_tag, distribution_tag, sample_size, self.device, n_features, **kwargs)
        self.__model = models.logistic_regression.LinearLogisticRegression(sum(estimator.param_dims), estimator).to(self.device)
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

    def post_training(self):
        pass
