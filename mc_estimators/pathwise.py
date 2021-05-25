from .estimator_base import MCEstimator


class Pathwise(MCEstimator):
    def _sample(self, params):
        return self.distribution.sample(params, self.sample_size, with_grad=True)

    def _backward(self, params, loss_fn, retain_graph):
        # Gradients set already, since the samples were not detached.
        pass
