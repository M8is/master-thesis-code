from .estimator_base import MCEstimator


class MVEstimator(MCEstimator):
    @property
    def name(self) -> str:
        return 'measure-valued'

    def backward(self, distribution, loss_fn, sample_size, retain_graph):
        mvd_samples = distribution.mvsample(sample_size)
        losses = loss_fn(mvd_samples).detach()
        return distribution.mvd_backward(losses, retain_graph)
