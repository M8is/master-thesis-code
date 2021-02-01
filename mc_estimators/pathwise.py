from mc_estimators.base import MCEstimator


class Pathwise(MCEstimator):
    def sample(self, params):
        return self.distribution.sample(params, self.sample_size, with_grad=True)

    def backward(self, params, losses):
        # Gradients set already, since the samples were not detached.
        pass
