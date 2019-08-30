import ConfigSpace


def get_hyperparameter_search_space(seed):
    """
    Full xgboost config space, featuring important hyperparameters,

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('mlr.xgboost', seed)

    imputer = ConfigSpace.CategoricalHyperparameter(
        name='num.impute.selected.cpo', choices=['impute.hist', 'impute.median', 'impute.mean'], default_value='impute.hist')
    booster = ConfigSpace.CategoricalHyperparameter(
        name='booster', choices=['gbtree', 'dart', 'gblinear'], default_value='gbtree')
    nrounds = ConfigSpace.UniformIntegerHyperparameter(
        name='nrounds', lower=1, upper=5973, default_value=100)
    eta = ConfigSpace.UniformFloatHyperparameter(
        name='eta', lower=2**-12, upper=2**0, log=True, default_value=0.1)
    lmbda = ConfigSpace.UniformFloatHyperparameter(
        name='lambda', lower=2**-12, upper=2**12, log=True, default_value=0.1)
    gamma = ConfigSpace.UniformFloatHyperparameter(
        name='gamma', lower=0.0, upper=32768, log=True, default_value=0.0)
    alpha = ConfigSpace.UniformFloatHyperparameter(
        name='alpha', lower=2**-12, upper=2**12, log=True, default_value=0.1)
    subsample = ConfigSpace.UniformFloatHyperparameter(
        name='subsample', lower=0.0003, upper=1, default_value=0.8)
    max_depth = ConfigSpace.UniformIntegerHyperparameter(
        name='max_depth', lower=1, upper=36, default_value=3)
    min_child_weight = ConfigSpace.UniformIntegerHyperparameter(
        name='min_child_weight', lower=0, upper=2^7, default_value=1)
    colsample_bytree = ConfigSpace.UniformFloatHyperparameter(
        name='colsample_bytree', lower=0.0001, upper=1, default_value=0.8)
    colsample_bylevel = ConfigSpace.UniformFloatHyperparameter(
        name='colsample_bylevel', lower=0.0001, upper=1, default_value=0.8)
    rate_drop = ConfigSpace.UniformFloatHyperparameter(
        name='rate_drop', lower=0.0003, upper=1, default_value=0.8)
    skip_drop = ConfigSpace.UniformFloatHyperparameter(
        name='skip_drop', lower=0.0003, upper=1, default_value=0.8)

    cs.add_hyperparameters([
        imputer,
        booster,
        nrounds,
        eta,
        lmbda,
        gamma,
        alpha,
        subsample,
        max_depth,
        min_child_weight,
        colsample_bytree,
        colsample_bylevel,
        rate_drop,
        skip_drop
    ])

    pars_depth = ConfigSpace.InCondition(max_depth, booster , ['gbtree', 'dart'])
    pars_cw = ConfigSpace.InCondition(min_child_weight, booster , ['gbtree', 'dart'])
    pars_eta = ConfigSpace.InCondition(eta, booster, ['gbtree', 'dart'])
    pars_cs_bt = ConfigSpace.InCondition(colsample_bylevel, booster, ['gbtree', 'dart'])
    pars_cs_bl = ConfigSpace.InCondition(colsample_bytree, booster, ['gbtree', 'dart'])
    pars_cs_ga = ConfigSpace.InCondition(gamma, booster, ['gbtree', 'dart'])
    skip = ConfigSpace.EqualsCondition(skip_drop, booster, 'dart')
    rate = ConfigSpace.EqualsCondition(rate_drop, booster, 'dart')
    cs.add_condition(pars_depth)
    cs.add_condition(pars_cw)
    cs.add_condition(pars_eta)
    cs.add_condition(pars_cs_bt)
    cs.add_condition(pars_cs_bl)
    cs.add_condition(pars_cs_ga)
    cs.add_condition(skip)
    cs.add_condition(rate)
    return cs


def get_hyperparameter_search_space_gbtree(seed):
    """
    Small version of xgboost config space, featuring important hyperparameters


    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('mlr.xgboost', seed)

    imputer = ConfigSpace.CategoricalHyperparameter(
        name='num.impute.selected.cpo', choices=['impute.hist', 'impute.median', 'impute.mean'], default_value='impute.hist')
    booster = ConfigSpace.CategoricalHyperparameter(
        name='booster', choices=['gbtree', 'dart', 'gblinear'], default_value='gbtree')
    nrounds = ConfigSpace.UniformIntegerHyperparameter(
        name='nrounds', lower=1, upper=5973, default_value=100)
    eta = ConfigSpace.UniformFloatHyperparameter(
        name='eta', lower=2**-12, upper=2**0, log=True, default_value=0.1)
    lmbda = ConfigSpace.UniformFloatHyperparameter(
        name='lambda', lower=2**-12, upper=2**12, log=True, default_value=0.1)
    gamma = ConfigSpace.UniformFloatHyperparameter(
        name='gamma', lower=0.0, upper=32768, log=True, default_value=0.0)
    alpha = ConfigSpace.UniformFloatHyperparameter(
        name='alpha', lower=2**-12, upper=2**12, log=True, default_value=0.1)
    subsample = ConfigSpace.UniformFloatHyperparameter(
        name='subsample', lower=0.0003, upper=1, default_value=0.8)
    max_depth = ConfigSpace.UniformIntegerHyperparameter(
        name='max_depth', lower=1, upper=36, default_value=3)
    min_child_weight = ConfigSpace.UniformIntegerHyperparameter(
        name='min_child_weight', lower=0, upper=2^7, default_value=1)
    colsample_bytree = ConfigSpace.UniformFloatHyperparameter(
        name='colsample_bytree', lower=0.0001, upper=1, default_value=0.8)
    colsample_bylevel = ConfigSpace.UniformFloatHyperparameter(
        name='colsample_bylevel', lower=0.0001, upper=1, default_value=0.8)

    cs.add_hyperparameters([
        imputer,
        booster,
        nrounds,
        eta,
        lmbda,
        gamma,
        alpha,
        subsample,
        max_depth,
        min_child_weight,
        colsample_bytree,
        colsample_bylevel,
    ])

    pars_depth = ConfigSpace.InCondition(max_depth, booster , ['gbtree', 'dart'])
    pars_cw = ConfigSpace.InCondition(min_child_weight, booster , ['gbtree', 'dart'])
    pars_eta = ConfigSpace.InCondition(eta, booster, ['gbtree', 'dart'])
    pars_cs_bt = ConfigSpace.InCondition(colsample_bylevel, booster, ['gbtree', 'dart'])
    pars_cs_bl = ConfigSpace.InCondition(colsample_bytree, booster, ['gbtree', 'dart'])
    pars_cs_ga = ConfigSpace.InCondition(gamma, booster, ['gbtree', 'dart'])
    cs.add_condition(pars_depth)
    cs.add_condition(pars_cw)
    cs.add_condition(pars_eta)
    cs.add_condition(pars_cs_bt)
    cs.add_condition(pars_cs_bl)
    cs.add_condition(pars_cs_ga)
    return cs