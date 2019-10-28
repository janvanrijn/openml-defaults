import ConfigSpace


def get_hyperparameter_search_space(seed):
    """
    Ranger config space

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations
    @attribute 'num.impute.selected.cpo' string
    @attribute 'num.trees' numeric
    @attribute 'replace' string
    @attribute 'sample.fraction' numeric
    @attribute 'mtry.power' numeric
    @attribute 'respect.unordered.factors' string
    @attribute 'min.node.size' numeric
    @attribute 'splitrule' string
    @attribute 'num.random.splits' numeric

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('mlr.glmnet', seed)

    imputer = ConfigSpace.CategoricalHyperparameter(
        name='num.impute.selected.cpo', choices=['impute.hist', 'impute.median', 'impute.mean'],
        default_value='impute.hist')
    alpha = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='alpha', lower=0, upper=1, default_value=0.1)
    s = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='s', lower=0, upper=1, default_value=0, log=True)

    cs.add_hyperparameters([
        imputer,
        alpha,
        s
    ])

    return cs
