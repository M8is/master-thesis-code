import torch
from mc_estimators.estimator_base import MCEstimator


class DiscreteMixture(torch.nn.Module):
    def __init__(self, selector: MCEstimator, component: MCEstimator):
        super().__init__()
        self.selector = selector
        self.component = component

    def forward(self, params):
        selector_params, component_params = params
        selected_components = self.selector(selector_params)
        return self.component(component_params[selected_components])

    def backward(self, params, losses):
        selector_params, component_params = params
        self.selector.backward(selector_params, losses.detach())
        self.component.backward(component_params, losses.detach())
