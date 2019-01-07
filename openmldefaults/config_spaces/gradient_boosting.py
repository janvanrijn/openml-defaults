import ConfigSpace


def get_hyperparameter_search_space_small(seed):
    """
    Small version of gradient boosting config space

    Parameters
    ----------
    seed: int
        Random seed that will be used to sample random configurations

    Returns
    -------
    cs: ConfigSpace.ConfigurationSpace
        The configuration space object
    """
    cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.GradientBoostingClassifier', seed)

    # fixed to deviance, as exponential requires two classes
    learning_rate = ConfigSpace.hyperparameters.UniformFloatHyperparameter(
        name='gradientboostingclassifier__learning_rate', lower=0.01, upper=2, default_value=0.1, log=True)
    n_estimators = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='gradientboostingclassifier__n_estimators', lower=64, upper=512, default_value=100, log=False)
    subsample = ConfigSpace.UniformFloatHyperparameter(
        name='gradientboostingclassifier__subsample', lower=0.0, upper=1.0, default_value=1.0)
    min_samples_split = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='gradientboostingclassifier__min_samples_split', lower=2, upper=20, default_value=2)
    max_depth = ConfigSpace.hyperparameters.UniformIntegerHyperparameter(
        name='gradientboostingclassifier__max_depth', lower=1, upper=10, default_value=3)

    cs.add_hyperparameters([

        learning_rate,
        n_estimators,
        subsample,
        min_samples_split,
        max_depth,
    ])

    return cs
