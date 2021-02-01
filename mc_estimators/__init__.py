from .measure_valued_derivative import MVD
from .pathwise import Pathwise
from .reinforce import Reinforce

from .distributions import MultivariateNormal, Exponential, Poisson


estimators = {
    'pathwise': Pathwise,
    'reinforce': Reinforce,
    'mvd': MVD,
}

distributions = {
    'multivariatenormal': MultivariateNormal,
    'exponential': Exponential,
    'poisson': Poisson
}


def get_estimator(estimator_tag: str, distribution_tag: str, sample_size: int, *args, **kwargs):
    """ Create a new estimator

    :param estimator_tag: How the gradient will be estimated. Check `estimators.keys()` for valid values.
    :param distribution_tag: Which distribution to use. Check `estimators[estimator].keys()` for valid values.
    :param sample_size: The number of MC samples used by the returned estimator
    :param args: Additional args for the algorithm
    :param kwargs: Additional kwargs for the algorithm
    :return: Estimator instance
    """
    estimator_tag = estimator_tag.lower()
    distribution_tag = distribution_tag.lower()
    if estimator_tag not in estimators:
        raise ValueError(f'Algorithm {estimator_tag} not available.')
    estimator = estimators[estimator_tag]

    if distribution_tag not in distributions:
        raise ValueError(f'Distribution {distribution_tag} not available for estimator {estimator_tag}.')
    distribution = distributions[distribution_tag]()

    return estimator(distribution, sample_size, *args, **kwargs)
