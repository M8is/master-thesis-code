from .estimator_base import MCEstimator


class Reinforce(MCEstimator):
    def __init__(self, distribution, sample_size, baseline=True, *args, **kwargs):
        super().__init__(distribution, sample_size, *args, **kwargs)
        self.baseline = baseline and self.sample_size > 1
        self._samples = None
        self.__loss_avg = None

    def _sample(self, raw_params):
        self._samples = self.distribution.sample(raw_params, self.sample_size, with_grad=False)
        return self._samples

    def _backward(self, raw_params, losses, retain_graph):
        if self._samples is None:
            raise ValueError("No forward call or multiple backward calls.")
        losses = losses.squeeze()
        log_probs = self.distribution.log_prob(raw_params, self._samples)
        baseline = self._get_baseline(losses) if self.baseline else 0
        if len(log_probs.shape) > 1:
            log_probs = log_probs.mean(dim=-1)
        ((losses - baseline) * log_probs).mean().backward(retain_graph=retain_graph)
        self._samples = None

    def _get_baseline(self, losses):
        self.__loss_avg = losses.mean() if self.__loss_avg is None else .9 * self.__loss_avg + .1 * losses.mean()
        return self.__loss_avg
