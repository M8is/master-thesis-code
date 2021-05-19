from typing import List

from mc_estimators.distributions import normal, exponential, poisson, bernoulli, categorical
from mc_estimators.measure_valued_derivative import MVD
from mc_estimators.pathwise import Pathwise
from mc_estimators.reinforce import Reinforce
from models.discrete_mixture import DiscreteMixture

estimators = {
    'pathwise': Pathwise,
    'reinforce': Reinforce,
    'mvd': MVD,
    'gaussianmixture': None  # Mixture requires separate instantiation
}

distributions = {
    'multivariatenormal': normal.MultivariateNormal,
    'normal': normal.MultivariateNormal,
    'exponential': exponential.Exponential,
    'poisson': poisson.Poisson,
    'bernoulli': bernoulli.Bernoulli,
    'categorical': categorical.Categorical
}


def get_estimator(estimator_tag: str, distribution_tag: str, sample_size: int, device: str, param_dims: List[int],
                  *args, **kwargs):
    """ Create a new estimator

    :param estimator_tag: How the gradient will be estimated. Check `estimators.keys()` for valid values.
    :param distribution_tag: Which distribution to use. Check `distributions.keys()` for valid values.
                             Note: Some estimator/distribution combinations do NOT WORK, even if there are no errors.
    :param sample_size: Number of MC samples used by the returned estimator
    :param device: Device on which to put created tensors
    :param param_dims: Dimensions of the probabilistic layer.
    :param args: Additional args for the algorithm
    :param kwargs: Additional kwargs for the algorithm
    :return: Estimator instance
    """
    estimator_tag = estimator_tag.lower()
    distribution_tag = distribution_tag.lower()
    if estimator_tag not in estimators:
        raise ValueError(f'Algorithm {estimator_tag} not available.')

    if distribution_tag == 'gaussianmixture':
        selector_tag = kwargs['selector_estimator'].lower()
        selector = estimators[selector_tag](distributions['categorical'](param_dims[:1], device), 1, **kwargs)
        component = estimators[estimator_tag](distributions['multivariatenormal'](param_dims[1:], device), sample_size,
                                              device, **kwargs)
        return DiscreteMixture(selector, component, *args, **kwargs)
    else:
        estimator = estimators[estimator_tag]
        if distribution_tag not in distributions:
            raise ValueError(f'Distribution {distribution_tag} not available for estimator {estimator_tag}.')
        distribution = distributions[distribution_tag](param_dims, device)
        return estimator(distribution, sample_size, *args, **kwargs)
