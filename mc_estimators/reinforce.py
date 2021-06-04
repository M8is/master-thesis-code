from .estimator_base import MCEstimator


class Reinforce(MCEstimator):
    def __init__(self, distribution, sample_size, baseline=True, *args, **kwargs):
        super().__init__(distribution, sample_size, *args, **kwargs)
        self.baseline = baseline and self.sample_size > 1
        self.__loss_avg = None

    def backward(self, raw_params, loss_fn, retain_graph=False):
        samples = self.distribution.sample(raw_params, self.sample_size, with_grad=False)
        losses = loss_fn(samples).detach()
        log_probs = self.distribution.log_prob(raw_params, samples)
        baseline = self._get_baseline(losses) if self.baseline else 0
        ((losses - baseline) * log_probs).mean().backward(retain_graph=retain_graph)

    def _get_baseline(self, losses):
        self.__loss_avg = losses.mean() if self.__loss_avg is None else .9 * self.__loss_avg + .1 * losses.mean()
        return self.__loss_avg
