from .estimator_base import MCEstimator


class PathwiseEstimator(MCEstimator):
    @staticmethod
    def name() -> str:
        return 'pathwise'

    def backward(self, distribution, loss_fn, sample_size, retain_graph):
        samples = distribution.rsample((sample_size,))
        losses = loss_fn(samples)
        losses.mean().backward(retain_graph=retain_graph)
