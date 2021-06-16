from .estimator_base import MCEstimator


class Pathwise(MCEstimator):
    def backward(self, distribution, loss_fn, retain_graph=False):
        samples = distribution.rsample((self.sample_size,))
        losses = loss_fn(samples)
        losses.mean().backward(retain_graph=retain_graph)
