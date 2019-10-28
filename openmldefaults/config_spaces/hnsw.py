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
    cs = ConfigSpace.ConfigurationSpace('mlr.RcppHNSW', seed)

    # fixed to deviance, as exponential requires two classes
    imputer = ConfigSpace.CategoricalHyperparameter(
        name='num.impute.selected.cpo', choices=['impute.hist', 'impute.median', 'impute.mean'],
        default_value='impute.hist')
    distance = ConfigSpace.CategoricalHyperparameter(
        name='distance', choices=['l2', 'cosine', 'ip'])
    k = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='k', lower=1, upper=50)
    M = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='M', lower=18, upper=50)
    ef = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='ef', lower=8, upper=256)
    ef_construction = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='ef_construction', lower=16, upper=512)

    cs.add_hyperparameters([
        imputer,
        distance,
        k,
        M,
        ef,
        ef_construction
    ])

    return cs
