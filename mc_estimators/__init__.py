from .measure_valued_derivative import MultivariateNormalMVD
from .pathwise import MultivariateNormalPathwise
from .reinforce import MultivariateNormalReinforce


estimators = {
    'pathwise': {'multivariatenormal': MultivariateNormalPathwise},
    'reinforce': {'multivariatenormal': MultivariateNormalReinforce},
    'mvd': {'multivariatenormal': MultivariateNormalMVD}
}


def get_estimator(estimator: str, distribution: str, sample_size: int, *args, **kwargs):
    """ Create a new estimator

    :param estimator: How the gradient will be estimated. Check `estimators.keys()` for valid values.
    :param distribution: Which distribution to use. Check `estimators[estimator].keys()` for valid values.
    :param sample_size: The number of MC samples used by the returned estimator
    :param args: Additional args for the algorithm
    :param kwargs: Additional kwargs for the algorithm
    :return: Estimator instance
    """
    estimator = estimator.lower()
    distribution = distribution.lower()
    if estimator not in estimators:
        raise ValueError(f'Algorithm {estimator} not available.')
    distributions = estimators[estimator]

    if distribution not in distributions:
        raise ValueError(f'Distribution {distribution} not available for estimator {estimator}.')

    return distributions[distribution](sample_size, *args, **kwargs)
