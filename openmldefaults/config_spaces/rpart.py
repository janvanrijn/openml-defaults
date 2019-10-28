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
    cs = ConfigSpace.ConfigurationSpace('mlr.rpart', seed)

    # fixed to deviance, as exponential requires two classes
    imputer = ConfigSpace.CategoricalHyperparameter(
        name='num.impute.selected.cpo', choices=['impute.hist', 'impute.median', 'impute.mean'],
        default_value='impute.hist')
    num_trees = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='cp', lower=0, upper=1, default_value=0.01, log = True)
    maxdepth = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='maxdepth', lower=1, upper=30)
    minbucket = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='minbucket', lower=1, upper=100)
    minsplit = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='minsplit', lower=1, upper=100)


    cs.add_hyperparameters([
        imputer,
        num_trees,
        maxdepth,
        minbucket,
        minsplit
    ])

    return cs
