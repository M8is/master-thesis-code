from .estimator_base import MCEstimator


class MVD(MCEstimator):
    def backward(self, raw_params, loss_fn, retain_graph=False, return_grad=False):
        mvd_samples = self.distribution.mvd_sample(raw_params, self.sample_size)
        losses = loss_fn(mvd_samples).detach()
        return self.distribution.mvd_backward(raw_params, losses, retain_graph)
