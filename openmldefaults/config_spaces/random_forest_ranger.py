import ConfigSpace


def get_hyperparameter_search_space(seed):
    """
    Ranger config space

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurationsrpart.py

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('mlr.ranger', seed)

    # fixed to deviance, as exponential requires two classes
    imputer = ConfigSpace.CategoricalHyperparameter(
        name='num.impute.selected.cpo', choices=['impute.hist', 'impute.median', 'impute.mean'], default_value='impute.hist')
    num_trees = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='num.trees', lower=1, upper=2000, default_value=100)
    replace = ConfigSpace.CategoricalHyperparameter(
        name = "replace", choices = ['TRUE', 'FALSE'], default_value = 'TRUE')
    sample_fraction = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='sample.fraction', lower=0.1, upper=1, default_value=0.1, log=True)
    mtry_power = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='mtry.pow', lower=0, upper=1)
    respect_unordered_factors = ConfigSpace.CategoricalHyperparameter(
        name="respect.unordered.factors", choices=['ignore', 'order', 'partition'], default_value='ignore')
    min_node_size = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='min.node.size', lower=2, upper=20, default_value=2)
    splitrule = ConfigSpace.CategoricalHyperparameter(
        name='splitrule', choices=['gini', 'extratrees'], default_value='gini')
    num_random_splits = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='num.random.splits', lower=1, upper=100, default_value=3)

    cs.add_hyperparameters([
        imputer,
        replace,
        sample_fraction,
        mtry_power,
        respect_unordered_factors,
        min_node_size,
        splitrule,
        num_random_splits
    ])

    pars_extra = ConfigSpace.InCondition(num_random_splits, splitrule, ['extratrees'])
    cs.add_condition(pars_extra)

    return cs
