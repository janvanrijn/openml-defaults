import ConfigSpace

from openmldefaults.config_spaces.misc import prefix


def get_libsvm_svc_default_search_space(show_prefix=True):
    imputation = ConfigSpace.CategoricalHyperparameter(prefix('imputation', show_prefix) + 'strategy', ['mean', 'median', 'most_frequent'])
    C = ConfigSpace.UniformFloatHyperparameter(prefix('classifier', show_prefix) + "C", 0.03125, 32768, log=True, default_value=1.0)
    kernel = ConfigSpace.CategoricalHyperparameter(name=prefix('classifier', show_prefix) + "kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf")
    degree = ConfigSpace.UniformIntegerHyperparameter(prefix('classifier', show_prefix) + "degree", 1, 5, default_value=3)
    gamma = ConfigSpace.UniformFloatHyperparameter(prefix('classifier', show_prefix) + "gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
    coef0 = ConfigSpace.UniformFloatHyperparameter(prefix('classifier', show_prefix) + "coef0", -1, 1, default_value=0)
    shrinking = ConfigSpace.CategoricalHyperparameter(prefix('classifier', show_prefix) + "shrinking", ["True", "False"], default_value="True")
    tol = ConfigSpace.UniformFloatHyperparameter(prefix('classifier', show_prefix) + "tol", 1e-5, 1e-1, default_value=1e-3, log=True)
    max_iter = ConfigSpace.UnParametrizedHyperparameter(prefix('classifier', show_prefix) + "max_iter", -1)

    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameters([imputation, C, kernel, degree, gamma, coef0, shrinking, tol, max_iter])

    degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, "poly")
    coef0_condition = ConfigSpace.InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs


def get_libsvm_svc_small_search_space(show_prefix=True):
    C = ConfigSpace.UniformFloatHyperparameter(prefix('classifier', show_prefix) + "C", 0.03125, 32768, log=True, default_value=1.0)
    kernel = ConfigSpace.CategoricalHyperparameter(name=prefix('classifier', show_prefix) + "kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf")
    degree = ConfigSpace.UniformIntegerHyperparameter(prefix('classifier', show_prefix) + "degree", 1, 5, default_value=3)
    gamma = ConfigSpace.UniformFloatHyperparameter(prefix('classifier', show_prefix) + "gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
    coef0 = ConfigSpace.UniformFloatHyperparameter(prefix('classifier', show_prefix) + "coef0", -1, 1, default_value=0)

    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameters([C, kernel, degree, gamma, coef0])

    degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, "poly")
    coef0_condition = ConfigSpace.InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs
