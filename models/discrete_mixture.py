import torch
from mc_estimators.estimator_base import MCEstimator


class DiscreteMixture(torch.nn.Module):
    def __init__(self, selector: MCEstimator, component: MCEstimator, *_, **kwargs):
        super().__init__()
        self.with_kl = kwargs["with_kl"] if "with_kl" in kwargs else True
        self.selector = selector
        self.component = component
        self.distribution = component.distribution
        self.param_dims = list(selector.distribution.param_dims)
        for _ in range(sum(selector.distribution.param_dims)):
            self.param_dims.append(sum(component.distribution.param_dims))
        # Disable separate KL divergence, as mixtures require a combined KL divergence.
        self.selector.with_kl = False
        self.component.with_kl = False

    def forward(self, raw_params):
        raw_params = raw_params.split(self.param_dims, dim=-1)
        raw_selector_params = raw_params[0]
        selector_params, selected_components = self.selector(raw_selector_params)
        raw_component_params = torch.stack(raw_params[1:])
        selected_raw_component_params = raw_component_params[selected_components]
        component_params, samples = self.component(selected_raw_component_params)
        return (selector_params, component_params), samples

    def backward(self, params, losses, retain_graph=False):
        selector_params, component_params = params
        if self.with_kl:
            self.kl(params).mean().backward(retain_graph=True)
        if losses.requires_grad:
            losses.backward()
        self.component.backward(component_params, losses.detach(), retain_graph=True)
        self.selector.backward(selector_params, losses.detach(), retain_graph=retain_graph)

    def kl(self, params):
        # TODO: implement KL divergence
        return torch.zeros_like(params, requires_grad=True)
