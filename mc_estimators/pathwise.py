from .base import MultivariateNormalProbabilistic


class MultivariateNormalPathwise(MultivariateNormalProbabilistic):
    def grad_samples(self, params):
        return self.sample(params, self.sample_size, with_grad=True)

    def backward(self, params, losses):
        # Gradients from losses are set already, since the samples were not detached.
        pass
