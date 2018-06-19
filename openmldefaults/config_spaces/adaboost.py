import ConfigSpace


def get_adaboost_default_search_space():
    cs = ConfigSpace.ConfigurationSpace()
    imputation = ConfigSpace.hyperparameters.CategoricalHyperparameter('strategy', ['mean', 'median', 'most_frequent'])

    n_estimators = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name="n_estimators", lower=50, upper=500, default_value=50, log=False)
    learning_rate = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    algorithm = ConfigSpace.hyperparameters.CategoricalHyperparameter(
        name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
    max_depth = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name="max_depth", lower=1, upper=10, default_value=1, log=False)

    cs.add_hyperparameters([imputation, n_estimators, learning_rate, algorithm, max_depth])

    return cs
