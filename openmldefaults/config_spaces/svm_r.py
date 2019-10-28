import ConfigSpace


def get_hyperparameter_search_space(seed):
    """
    Ranger config space

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations
    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('mlr.svm', seed)

    # fixed to deviance, as exponential requires two classes
    imputer = ConfigSpace.CategoricalHyperparameter(
        name='num.impute.selected.cpo', choices=['impute.hist', 'impute.median', 'impute.mean'],
        default_value='impute.hist')
    kernel = ConfigSpace.CategoricalHyperparameter(
        name='kernel', choices=['linear', 'polynomial', 'radial'])
    cost = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='cost', lower=0, upper=4096, log = True)
    gamma = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='gamma', lower=0, upper=4096, log = True)
    degree = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='degree', lower=2, upper=5)
    tolerance = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='tolerance', lower=1/4096, upper=1/8)
    shrinking = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name='shrinking', choices=['TRUE', 'FALSE'])

    cs.add_hyperparameters([
        imputer,
        kernel,
        cost,
        gamma,
        degree,
        tolerance,
        shrinking
    ])

    poly_ker = ConfigSpace.InCondition(degree, kernel, ['polynomial'])
    cs.add_condition(poly_ker)
    gamma_ker = ConfigSpace.InCondition(gamma, kernel, ['radial'])
    cs.add_condition(gamma_ker)

    return cs
