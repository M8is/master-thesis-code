from typing import Type

from distributions import normal, bernoulli, exponential, categorical, poisson
from distributions.distribution_base import Distribution

distributions = {
    'multivariatenormal': normal.MultivariateNormal,
    'normal': normal.MultivariateNormal,
    'exponential': exponential.Exponential,
    'poisson': poisson.Poisson,
    'bernoulli': bernoulli.Bernoulli,
    'categorical': categorical.Categorical
}


def get_distribution_type(distribution: str) -> Type[Distribution]:
    """ Get a distribution class from a string identifier.

    :param distribution: Distribution identifier
    :return: Distribution class (subclass of `distributions.distribution_base.Distribution`).
    """
    distribution = distribution.lower()
    if distribution not in distributions:
        raise ValueError(f'Distribution {distribution} not available. '
                         f'Available distributions are {(k for k in distributions.keys())}')
    return distributions[distribution]
