import ConfigSpace


def get_random_forest_default_search_space():

    cs = ConfigSpace.ConfigurationSpace()
    imputation = ConfigSpace.CategoricalHyperparameter('strategy', ['mean', 'median', 'most_frequent'],
                                                       meta={'component': 'imputation'})
    n_estimators = ConfigSpace.Constant("n_estimators", 100,
                                        meta={'component': 'classifier'})
    criterion = ConfigSpace.CategoricalHyperparameter("criterion", ["gini", "entropy"], default_value="gini",
                                                      meta={'component': 'classifier'})

    # The maximum number of features used in the forest is calculated as m^max_features, where
    # m is the total number of features, and max_features is the hyperparameter specified below.
    # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
    # corresponds with Geurts' heuristic.
    # XXX: Can't use 0 as minimal value, as sklearn forbids this value
    max_features = ConfigSpace.UniformFloatHyperparameter(
        "max_features", 0.001, 1., default_value=0.5, meta={'component': 'classifier'})
    max_depth = ConfigSpace.UnParametrizedHyperparameter(
        "max_depth", "None", meta={'component': 'classifier'})
    min_samples_split = ConfigSpace.UniformIntegerHyperparameter(
        "min_samples_split", 2, 20, default_value=2, meta={'component': 'classifier'})
    min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 20, default_value=1, meta={'component': 'classifier'})
    min_weight_fraction_leaf = ConfigSpace.UnParametrizedHyperparameter(
        "min_weight_fraction_leaf", 0., meta={'component': 'classifier'})
    max_leaf_nodes = ConfigSpace.UnParametrizedHyperparameter(
        "max_leaf_nodes", "None", meta={'component': 'classifier'})
    bootstrap = ConfigSpace.CategoricalHyperparameter(
        "bootstrap", ["True", "False"], default_value="True", meta={'component': 'classifier', 'true_type': bool})

    cs.add_hyperparameters([imputation, n_estimators, criterion, max_features,
                            max_depth, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_leaf_nodes,
                            bootstrap])
    return cs


def get_random_forest_micro_search_space():

    cs = ConfigSpace.ConfigurationSpace(meta={'flow_id': 6969})

    n_estimators = ConfigSpace.Constant("classifier__n_estimators", 100)
    max_features = ConfigSpace.UniformFloatHyperparameter(
        "classifier__max_features", 0.001, 1., default_value=0.5)
    min_samples_split = ConfigSpace.UniformIntegerHyperparameter(
        "classifier__min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = ConfigSpace.UniformIntegerHyperparameter(
        "classifier__min_samples_leaf", 1, 20, default_value=1)

    cs.add_hyperparameters([n_estimators, max_features,
                            min_samples_split, min_samples_leaf])
    return cs
