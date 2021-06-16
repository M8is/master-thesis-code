import torch
from mc_estimators.estimator_base import MCEstimator


# FIXME: this is probably completely broken at the moment
class DiscreteMixture(torch.nn.Module):
    def __init__(self, selector: MCEstimator, component: MCEstimator, *_, **__):
        super().__init__()
        self.selector = selector
        self.component = component
        self.distribution_type = component.distribution_type
        self.param_dims = list(selector.distribution_type.param_dims)
        # FIXME: selector.distribution_type.param_dims requires latent_dim parameter
        for _ in range(sum(selector.distribution_type.param_dims)):
            self.param_dims.append(sum(component.distribution_type.param_dims))

    def forward(self, raw_params):
        raw_params = raw_params.split(self.param_dims, dim=-1)
        raw_selector_params = raw_params[0]
        selector_dist, selected_components = self.selector(raw_selector_params)
        raw_component_params = torch.stack(raw_params[1:])
        component_dist, selected_raw_component_params = raw_component_params[selected_components]
        samples = self.component(selected_raw_component_params)
        return (selector_dist, component_dist), samples

    def backward(self, distribution, losses, retain_graph=False):
        selector_dist, component_dist = distribution
        self.component.backward(component_dist, losses, retain_graph=True)
        self.selector.backward(selector_dist, losses, retain_graph=retain_graph)

    def kl(self, params):
        # TODO: implement KL divergence
        return torch.zeros_like(params, requires_grad=True)
