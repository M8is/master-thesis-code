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
        selector_params = params[0]
        selected_components = self.selector(selector_params)
        component_params = torch.stack(params[1:])[selected_components]
        return self.component(component_params)

    def backward(self, params, losses):
        params = params.split(self.param_dims, dim=-1)
        self.selector.backward(params[:1], losses.detach().sum(-1).mean(0), retain_graph=True)
        self.component.backward(params[1:], losses.detach())
