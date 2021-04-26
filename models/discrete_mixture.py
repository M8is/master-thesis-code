import torch
from mc_estimators.estimator_base import MCEstimator


class DiscreteMixture(torch.nn.Module):
    def __init__(self, selector: MCEstimator, component: MCEstimator):
        super().__init__()
        self.selector = selector
        self.component = component
        self.param_dims = list(selector.distribution.param_dims)
        for _ in range(sum(selector.distribution.param_dims)):
            self.param_dims.append(sum(component.distribution.param_dims))

    def forward(self, params):
        params = params.split(self.param_dims, dim=-1)
        raw_selector_params = params[0]
        selector_params, selected_components = self.selector(raw_selector_params)
        raw_component_params = torch.stack(params[1:])[selected_components]
        component_params, samples = self.component(raw_component_params)
        return (selector_params, component_params), samples

    def backward(self, params, losses):
        params = params.split(self.param_dims, dim=-1)
        self.component.backward(torch.stack(params[1:]), losses.detach(), retain_graph=True)
        self.selector.backward(torch.stack(params[:1]), losses.detach().mean(0).sum(-1))
