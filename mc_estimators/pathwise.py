from . import base


class MultivariateNormalPathwise(base.MultivariateNormalProbabilistic):
    def grad_samples(self, params):
        return self.sample(params, self.sample_size, with_grad=True)

    def backward(self, params, losses):
        # Gradients set already, since the samples were not detached.
        pass


class ExponentialPathwise(base.ExponentialProbabilistic):
    def grad_samples(self, params):
        return self.sample(params, self.sample_size, with_grad=True)

    def backward(self, params, losses):
        # Gradients set already, since the samples were not detached.
        pass


class PoissonPathwise(base.PoissonProbabilistic):
    def grad_samples(self, params):
        return self.sample(params, self.sample_size, with_grad=True)

    def backward(self, params, losses):
        # Gradients set already, since the samples were not detached.
        pass
