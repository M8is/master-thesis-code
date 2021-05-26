from .estimator_base import MCEstimator


class Pathwise(MCEstimator):
    def _sample(self, raw_params):
        return self.distribution.sample(raw_params, self.sample_size, with_grad=True)

    def backward(self, raw_params, loss_fn, retain_graph=False):
        # Gradients set already, since the samples were not detached.
        pass
