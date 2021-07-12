from distributions.mc_estimators.convex_combination import MVSFEstimator
from distributions.mc_estimators.measure_valued import MVEstimator
from distributions.mc_estimators.pathwise import PathwiseEstimator
from distributions.mc_estimators.score_function import SFEstimator

estimators = {
    'pathwise': PathwiseEstimator,
    'reptrick': PathwiseEstimator,
    'score-function': SFEstimator,
    'log-ratio': SFEstimator,
    'sf': SFEstimator,
    'reinforce': SFEstimator,
    'mv': MVEstimator,
    'mvd': MVEstimator,
    'measure-valued': MVEstimator,
    'mvsf': MVSFEstimator
}


def get_estimator(mc_estimator: str, *estimator_args, **estimator_kwargs):
    """ Create a new estimator

    :param mc_estimator: How the gradient will be estimated. Check `estimators.keys()` for valid values.
    :param estimator_args: Additional args for estimator construction.
    :param estimator_kwargs: Additional kwargs for estimator construction.
    :return: Estimator instance
    """
    mc_estimator = mc_estimator.lower()
    if mc_estimator not in estimators:
        raise ValueError(f'Estimator `{mc_estimator}` not available.')
    return estimators[mc_estimator](*estimator_args, **estimator_kwargs)
