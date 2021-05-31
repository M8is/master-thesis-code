from .estimator_base import MCEstimator


class MVD(MCEstimator):
    def _sample(self, raw_params):
        return self.distribution.sample(raw_params, with_grad=False)

    def backward(self, raw_params, loss_fn, retain_graph=False):
        mvd_samples = self.distribution.mvd_sample(raw_params, self.sample_size)
        losses = loss_fn(mvd_samples).detach()
        self.distribution.mvd_backward(raw_params, losses, retain_graph)
