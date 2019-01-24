import ConfigSpace


def get_hyperparameter_search_space_sklearn_0_19(seed):
    """
    Configspace based on scikit-learn 0.19 (pre-ColumnTransformer)

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    imputation = ConfigSpace.CategoricalHyperparameter('imputation__strategy', ['mean', 'median', 'most_frequent'])

    C = ConfigSpace.UniformFloatHyperparameter("classifier__C", 0.03125, 32768, log=True, default_value=1.0)
    # No linear kernel here, because we have liblinear
    kernel = ConfigSpace.CategoricalHyperparameter(name="classifier__kernel", choices=["rbf", "poly", "sigmoid"],
                                       default_value="rbf")
    degree = ConfigSpace.UniformIntegerHyperparameter("classifier__degree", 2, 5, default_value=3)
    gamma = ConfigSpace.UniformFloatHyperparameter("classifier__gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
    # TODO this is totally ad-hoc
    coef0 = ConfigSpace.UniformFloatHyperparameter("classifier__coef0", -1, 1, default_value=0)
    # probability is no hyperparameter, but an argument to the SVM algo
    shrinking = ConfigSpace.CategoricalHyperparameter("classifier__shrinking", [True, False], default_value=True)
    tol = ConfigSpace.UniformFloatHyperparameter("classifier__tol", 1e-5, 1e-1, default_value=1e-3, log=True)
    # cache size is not a hyperparameter, but an argument to the program!
    max_iter = ConfigSpace.UnParametrizedHyperparameter("classifier__max_iter", -1)

    cs = ConfigSpace.ConfigurationSpace('sklearn.svm.SVC', seed)
    cs.add_hyperparameters([imputation, C, kernel, degree, gamma, coef0, shrinking, tol, max_iter])

    degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, "poly")
    coef0_condition = ConfigSpace.InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs


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
