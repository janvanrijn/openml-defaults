import ConfigSpace


def get_hyperparameter_search_space_small(seed):
    """
    Small version of svm config space, featuring important hyperparameters
    based on https://arxiv.org/abs/1710.04725

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', seed)

    C = ConfigSpace.UniformFloatHyperparameter(
        name='svc__C', lower=0.03125, upper=32768, log=True, default_value=1.0)
    kernel = ConfigSpace.CategoricalHyperparameter(
        name='svc__kernel', choices=['rbf', 'poly', 'sigmoid'], default_value='rbf')
    degree = ConfigSpace.UniformIntegerHyperparameter(
        name='svc__degree', lower=1, upper=5, default_value=3)
    gamma = ConfigSpace.UniformFloatHyperparameter(
        name='svc__gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)
    coef0 = ConfigSpace.UniformFloatHyperparameter(
        name='svc__coef0', lower=-1, upper=1, default_value=0)

    cs.add_hyperparameters([
        C,
        kernel,
        degree,
        gamma,
        coef0
    ])

    degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, 'poly')
    coef0_condition = ConfigSpace.InCondition(coef0, kernel, ['poly', 'sigmoid'])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs


def get_hyperparameter_search_space_micro(seed):
    """
    Small version of svm config space, featuring important hyperparameters
    as used by:
    http://metalearning.ml/2018/papers/metalearn2018_paper70.pdf

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', seed)

    kernel = ConfigSpace.Constant(name='svc__kernel', value='rbf')
    C = ConfigSpace.UniformFloatHyperparameter(name='svc__C', lower=0.03125, upper=32768, log=True, default_value=1.0)
    gamma = ConfigSpace.UniformFloatHyperparameter(
        name='svc__gamma', lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)

    cs.add_hyperparameters([
        kernel,
        C,
        gamma
    ])

    return cs
