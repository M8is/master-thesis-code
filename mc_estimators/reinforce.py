from .estimator_base import MCEstimator


class Reinforce(MCEstimator):
    def __init__(self, baseline=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.baseline = baseline
        if self.baseline:
            self.__loss_avg = None

    def backward(self, distribution, loss_fn, retain_graph=False):
        samples = distribution.sample((self.sample_size,))
        losses = loss_fn(samples).detach()
        log_probs = distribution.log_prob(samples)
        baseline = self.__get_baseline(losses) if self.baseline and self.sample_size > 1 else 0
        ((losses - baseline) * log_probs).mean().backward(retain_graph=retain_graph)

    def __get_baseline(self, losses):
        self.__loss_avg = losses.mean() if self.__loss_avg is None else .9 * self.__loss_avg + .1 * losses.mean()
        return self.__loss_avg
