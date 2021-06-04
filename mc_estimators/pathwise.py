from .estimator_base import MCEstimator


class Pathwise(MCEstimator):
    def backward(self, raw_params, loss_fn, retain_graph=False):
        samples = self.distribution.sample(raw_params, self.sample_size, with_grad=True)
        losses = loss_fn(samples)
        losses.mean().backward(retain_graph=retain_graph)
