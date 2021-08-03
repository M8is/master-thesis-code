import torch

from models.linear_regression import LinearRegressor


class LogisticRegressionClassifier(LinearRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def interpret(self, sample: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().interpret(sample, data))
