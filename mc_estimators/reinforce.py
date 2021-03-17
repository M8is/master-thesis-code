from .estimator_base import MCEstimator


class Reinforce(MCEstimator):
    def __init__(self, distribution, sample_size, baseline=True, *args, **kwargs):
        super().__init__(distribution, sample_size, *args, **kwargs)
        self.with_baseline = baseline and self.sample_size > 1
        self._samples = None
        self.__loss_sum = None
        self.__grad_count = 0

    def sample(self, params):
        self._samples = self.distribution.sample(params, self.sample_size, with_grad=False)
        return self._samples

    def backward(self, params, losses):
        if self._samples is None:
            raise ValueError("No forward call or multiple backward calls.")
        log_probs = self.distribution.log_prob(params, self._samples)
        baseline = self._get_baseline(losses)
        ((losses - baseline) * log_probs).mean().backward()
        self._samples = None

    def _get_baseline(self, losses):
        mean_loss = losses.mean(-1, keepdim=True)
        self.__loss_sum = mean_loss if self.__loss_sum is None else self.__loss_sum + mean_loss
        self.__grad_count += 1
        return self.__loss_sum / self.__grad_count
