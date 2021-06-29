from distributions.mc_estimators.measure_valued_derivative import MVD
from distributions.mc_estimators.pathwise import Pathwise
from distributions.mc_estimators.reinforce import Reinforce

estimators = {
    'pathwise': Pathwise,
    'reinforce': Reinforce,
    'mvd': MVD,
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
