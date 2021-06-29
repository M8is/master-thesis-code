from .estimator_base import MCEstimator


class SFEstimator(MCEstimator):
    __LOSS_AVG_KEY = '__loss_avg'

    @property
    def name(self) -> str:
        return 'score-function'

    def backward(self, distribution, loss_fn, sample_size, retain_graph):
        samples = distribution.sample((sample_size,))
        losses = loss_fn(samples).detach()
        log_probs = distribution.log_prob(samples)
        baseline = self.__get_baseline(losses)
        ((losses - baseline) * log_probs).mean().backward(retain_graph=retain_graph)

    def __get_baseline(self, losses):
        baseline = getattr(self, self.__LOSS_AVG_KEY, 0)
        setattr(self, self.__LOSS_AVG_KEY, .9 * baseline + .1 * losses.mean())
        return baseline
