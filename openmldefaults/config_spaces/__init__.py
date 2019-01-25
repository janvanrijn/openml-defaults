from . import adaboost
from . import gradient_boosting
from . import random_forest
from . import svc

import ConfigSpace
import importlib
import openmldefaults
import sklearn
import sklearnbot
import typing


def get_config_space(classifier_name: str, random_seed: int, space_type: typing.Optional[str]) -> \
        ConfigSpace.ConfigurationSpace:
    if space_type is None:
        return sklearnbot.config_spaces.get_config_space(classifier_name, random_seed)
    else:
        cs = getattr(getattr(openmldefaults.config_spaces, classifier_name),
                     'get_hyperparameter_search_space_%s' % space_type)
        return cs(random_seed)


def get_config_spaces(classifier_names: typing.List[str], random_seed: int, space_type: typing.Optional[str]) -> \
        ConfigSpace.ConfigurationSpace:
    config_spaces = dict()
    for classifier_name in classifier_names:
        config_spaces[classifier_name] = get_config_space(classifier_name, random_seed, space_type)

    cs = ConfigSpace.ConfigurationSpace('mixed', random_seed)
    classifier_choice = ConfigSpace.CategoricalHyperparameter(name='classifier', choices=classifier_names)
    cs.add_hyperparameter(classifier_choice)

    for identifier, sub_space in config_spaces.items():
        parent_hyperparameter = {'parent': classifier_choice, 'value': identifier}
        cs.add_configuration_space(identifier,
                                   sub_space,
                                   parent_hyperparameter=parent_hyperparameter)
    return cs


def reinstantiate_model(classifier_name: str, search_space_identifier: typing.Optional[str],
                        numeric_indices: typing.List[int],
                        nominal_indices: typing.List[int]) -> sklearn.base.BaseEstimator:
    config_space = get_config_space(classifier_name, 0, search_space_identifier)
    if search_space_identifier == 'sklearn_0_19':
        import openmlstudy14
        module_name = config_space.name.rsplit('.', 1)
        model_class = getattr(importlib.import_module(module_name[0]),
                              module_name[1])
        clf = openmlstudy14.pipeline.EstimatorFactory._get_pipeline(nominal_indices=nominal_indices,
                                                                    numeric_indices=numeric_indices,
                                                                    model_class=model_class)
        if config_space.meta is not None:
            clf.set_params(**config_space.meta)
        return clf
    else:
        return sklearnbot.sklearn.as_estimator(config_space, numeric_indices, nominal_indices)
