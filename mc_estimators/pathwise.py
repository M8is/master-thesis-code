from .estimator_base import MCEstimator


class Pathwise(MCEstimator):
    def backward(self, raw_params, loss_fn, retain_graph=False, return_grad=False):
        samples, params = self.distribution.sample(raw_params, self.sample_size, with_grad=True)
        if return_grad:
            params.retain_grad()
        losses = loss_fn(samples)
        losses.mean().backward(retain_graph=retain_graph)
        return params.grad if return_grad else None
