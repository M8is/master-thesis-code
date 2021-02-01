from .estimator_base import MCEstimator


class Reinforce(MCEstimator):
    def __init__(self, distribution, sample_size):
        super().__init__(distribution, sample_size)
        self.samples = None

    def sample(self, params):
        self.samples = self.distribution.sample(params, self.sample_size, with_grad=False)
        return self.samples

    def backward(self, params, losses):
        if self.samples is None:
            raise ValueError("No forward call or multiple backward calls.")
        log_probs = self.distribution.log_prob(params, self.samples)
        assert losses.shape == log_probs.shape
        (losses * log_probs).mean().backward()
        self.samples = None
