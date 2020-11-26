from mc_estimators.probabilistic_objective_gradient import Probabilistic


class Reinforce(Probabilistic):
    def __init__(self, episode_size, distribution):
        super().__init__(episode_size, distribution)

    def grad_samples(self, params):
        samples = self.distribution(*params).sample((self.episode_size,))
        self._to_backward((params, samples))
        return samples

    def backward(self, losses):
        params, samples = self._from_forward()
        log_probs = self.distribution(*params).log_prob(samples).reshape(-1, 1)
        return (losses * log_probs).mean().backward()
