import ConfigSpace
import sklearn


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
    cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.AdaBoostClassifier',
                                        seed,
                                        meta={"adaboostclassifier__base_estimator": sklearn.tree.DecisionTreeClassifier()})
    imputation = ConfigSpace.CategoricalHyperparameter('imputation__strategy', ['mean', 'median', 'most_frequent'])
    n_estimators = ConfigSpace.UniformIntegerHyperparameter(
        name="classifier__n_estimators", lower=50, upper=500, default_value=50, log=False)
    learning_rate = ConfigSpace.UniformFloatHyperparameter(
        name="classifier__learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    algorithm = ConfigSpace.CategoricalHyperparameter(
        name="classifier__algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")
    max_depth = ConfigSpace.UniformIntegerHyperparameter(
        name="classifier__base_estimator__max_depth", lower=1, upper=10, default_value=1, log=False)

    cs.add_hyperparameters([imputation, n_estimators, learning_rate, algorithm, max_depth])

    return cs
