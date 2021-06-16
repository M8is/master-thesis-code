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


def get_estimator(mc_estimator: str, distribution: str, *args, **kwargs):
    """ Create a new estimator

    :param mc_estimator: How the gradient will be estimated. Check `estimators.keys()` for valid values.
    :param distribution: Which distribution to use. Check `distributions.keys()` for valid values.
    :param args: Additional args for the algorithm
    :param kwargs: Additional kwargs for the algorithm
    :return: Estimator instance
    """
    mc_estimator = mc_estimator.lower()
    distribution = distribution.lower()
    if mc_estimator not in estimators:
        raise ValueError(f'Algorithm {mc_estimator} not available.')

    if distribution == 'gaussianmixture':
        selector_tag = kwargs['selector_estimator'].lower()
        selector = estimators[selector_tag](distributions['categorical'], 1, **kwargs)
        component = estimators[mc_estimator](distributions['multivariatenormal'], **kwargs)
        return DiscreteMixture(selector, component, *args, **kwargs)
    else:
        estimator = estimators[mc_estimator]
        if distribution not in distributions:
            raise ValueError(f'Distribution {distribution} not available for estimator {mc_estimator}.')
        return estimator(distribution_type=distributions[distribution], *args, **kwargs)
