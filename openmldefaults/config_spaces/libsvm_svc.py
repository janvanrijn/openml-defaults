import ConfigSpace


def get_libsvm_svc_default_search_space():
    imputation = ConfigSpace.CategoricalHyperparameter(
        'strategy', ['mean', 'median', 'most_frequent'], meta={'component': 'imputation'})
    C = ConfigSpace.UniformFloatHyperparameter(
        "C", 0.03125, 32768, log=True, default_value=1.0, meta={'component': 'classifier'})
    kernel = ConfigSpace.CategoricalHyperparameter(
        "kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf", meta={'component': 'classifier'})
    degree = ConfigSpace.UniformIntegerHyperparameter(
        "degree", 1, 5, default_value=3, meta={'component': 'classifier'})
    gamma = ConfigSpace.UniformFloatHyperparameter(
        "gamma", 3.0517578125e-05, 8, log=True, default_value=0.1, meta={'component': 'classifier'})
    coef0 = ConfigSpace.UniformFloatHyperparameter(
        "coef0", -1, 1, default_value=0, meta={'component': 'classifier'})
    shrinking = ConfigSpace.CategoricalHyperparameter(
        "shrinking", ["True", "False"], default_value="True", meta={'component': 'classifier'})
    tol = ConfigSpace.UniformFloatHyperparameter(
        "tol", 1e-5, 1e-1, default_value=1e-3, log=True, meta={'component': 'classifier'})
    max_iter = ConfigSpace.UnParametrizedHyperparameter(
        "max_iter", -1, meta={'component': 'classifier'})

    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameters([imputation, C, kernel, degree, gamma, coef0, shrinking, tol, max_iter])

    degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, "poly")
    coef0_condition = ConfigSpace.InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs


def get_libsvm_svc_small_search_space():
    C = ConfigSpace.UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0, meta={'component': 'classifier'})
    kernel = ConfigSpace.CategoricalHyperparameter("kernel", choices=["rbf", "poly", "sigmoid"], default_value="rbf", meta={'component': 'classifier'})
    degree = ConfigSpace.UniformIntegerHyperparameter("degree", 1, 5, default_value=3, meta={'component': 'classifier'})
    gamma = ConfigSpace.UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True, default_value=0.1, meta={'component': 'classifier'})
    coef0 = ConfigSpace.UniformFloatHyperparameter("coef0", -1, 1, default_value=0, meta={'component': 'classifier'})

    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameters([C, kernel, degree, gamma, coef0])

    degree_depends_on_poly = ConfigSpace.EqualsCondition(degree, kernel, "poly")
    coef0_condition = ConfigSpace.InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs


def get_libsvm_svc_micro_search_space():
    C = ConfigSpace.UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0, meta={'component': 'classifier'})
    kernel = ConfigSpace.Constant("kernel", value="rbf", meta={'component': 'classifier'})
    gamma = ConfigSpace.UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True, default_value=0.1, meta={'component': 'classifier'})

    cs = ConfigSpace.ConfigurationSpace()
    cs.add_hyperparameters([C, kernel,  gamma])
    return cs
