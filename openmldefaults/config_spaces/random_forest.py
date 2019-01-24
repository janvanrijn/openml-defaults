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
    cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.RandomForestClassifier', seed)
    imputation = ConfigSpace.CategoricalHyperparameter('imputation__strategy', ['mean', 'median', 'most_frequent'])
    n_estimators = ConfigSpace.Constant("classifier__n_estimators", 100)
    criterion = ConfigSpace.CategoricalHyperparameter(
        "classifier__criterion", ["gini", "entropy"], default_value="gini")

    # The maximum number of features used in the forest is calculated as m^max_features, where
    # m is the total number of features, and max_features is the hyperparameter specified below.
    # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
    # corresponds with Geurts' heuristic.
    max_features = ConfigSpace.UniformFloatHyperparameter(
        "classifier__max_features", 0., 1., default_value=0.5)

    # max_depth = UnParametrizedHyperparameter("classifier__max_depth", None)
    min_samples_split = ConfigSpace.UniformIntegerHyperparameter(
        "classifier__min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter(
        "classifier__min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = ConfigSpace.UnParametrizedHyperparameter("classifier__min_weight_fraction_leaf", 0.)
    bootstrap = ConfigSpace.CategoricalHyperparameter(
        "classifier__bootstrap", [True, False], default_value=True)
    cs.add_hyperparameters([imputation, n_estimators, criterion, max_features,
                            min_samples_split, min_samples_leaf,
                            bootstrap, min_weight_fraction_leaf])
