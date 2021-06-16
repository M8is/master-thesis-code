from .estimator_base import MCEstimator


class MVD(MCEstimator):
    def backward(self, distribution, loss_fn, retain_graph=False):
        mvd_samples = distribution.mvd_sample(self.sample_size)
        losses = loss_fn(mvd_samples).detach()
        return distribution.mvd_backward(losses, retain_graph)
