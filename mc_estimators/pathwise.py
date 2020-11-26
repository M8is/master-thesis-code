from mc_estimators.probabilistic_objective_gradient import Probabilistic


class Pathwise(Probabilistic):
    def grad_samples(self, params):
        return self.distribution(*params).rsample((self.episode_size,))

    def backward(self, losses):
        return losses.mean().backward()
