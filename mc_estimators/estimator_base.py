from abc import ABC, abstractmethod


class MCEstimator(ABC):
    def __init__(self, distribution, sample_size):
        self.distribution = distribution
        self.sample_size = sample_size

    @abstractmethod
    def sample(self, params):
        pass

    @abstractmethod
    def backward(self, params, losses):
        pass

    def __str__(self):
        return f'{type(self).__name__} {type(self.distribution).__name__} {self.sample_size} sample(s)'
