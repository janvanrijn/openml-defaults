from . import gradient_boosting
from . import resnet
from . import svc
from . import text_classification

import ConfigSpace
import openmldefaults
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

