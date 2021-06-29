from abc import ABC, abstractmethod

import torch

from distributions.distribution_base import Distribution


class StochasticModel(ABC, torch.nn.Module):
    def forward(self, x):
        return self.interpret(self.encode(x).sample(), x)

    @abstractmethod
    def encode(self, data: torch.Tensor) -> Distribution:
        """ Encodes the given data into a probability distribution.

        :param data: Input data
        :return: Encoding as probability distribution
        """
        pass

    @abstractmethod
    def interpret(self, sample: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """ Interprets the sample using the given data.

        :param sample: Sample to interpret
        :param data: Data point the encoding was based on.
        :return: Interpretation, e.g., label prediction, reconstruction, ...
        """
        pass
