from .estimator_base import MCEstimator


class SFEstimator(MCEstimator):
    @staticmethod
    def name() -> str:
        return 'score-function'

    def __init__(self, no_baseline: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if no_baseline:
            self.__get_baseline = lambda *_: 0
        else:
            self.__loss_avg = 0

    def backward(self, distribution, loss_fn, sample_size, retain_graph):
        samples = distribution.sample((sample_size,))
        losses = loss_fn(samples).detach()
        log_probs = distribution.log_prob(samples).squeeze()
        baseline = self.__get_baseline(losses)
        ((losses - baseline) * log_probs).mean().backward(retain_graph=retain_graph)

    def __get_baseline(self, losses):
        baseline = self.__loss_avg
        if not self._frozen:
            self.__loss_avg = .9 * baseline + .1 * losses.mean()
        return baseline
