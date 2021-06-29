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
}


def get_estimator(mc_estimator: str):
    """ Create a new estimator

    :param mc_estimator: How the gradient will be estimated. Check `estimators.keys()` for valid values.
    :return: Estimator instance
    """
    mc_estimator = mc_estimator.lower()
    if mc_estimator not in estimators:
        raise ValueError(f'Estimator `{mc_estimator}` not available.')
    return estimators[mc_estimator]()
