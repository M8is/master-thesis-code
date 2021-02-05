from abc import ABC, abstractmethod


class Distribution(ABC):
    @abstractmethod
    def sample(self, params, size=1, with_grad=False):
        pass

    @abstractmethod
    def mvd_sample(self, params, size):
        pass

    @abstractmethod
    def mvd_constant(self, params):
        pass

    @abstractmethod
    def kl(self, params):
        pass

    @abstractmethod
    def log_prob(self, params, samples):
        pass
