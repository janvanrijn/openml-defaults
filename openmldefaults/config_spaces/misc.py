import ConfigSpace
import json
import numpy as np
import openmlcontrib
import openmldefaults

from typing import Dict


def remove_hyperparameter(config_space: ConfigSpace.ConfigurationSpace,
                          hyperparameter_name: str) -> ConfigSpace.ConfigurationSpace:
    config_space_prime = ConfigSpace.ConfigurationSpace(meta=config_space.meta)
    for hyperparameter in config_space.get_hyperparameters():
        if hyperparameter.name != hyperparameter_name:
            config_space_prime.add_hyperparameter(hyperparameter)
    for condition in config_space.get_conditions():
        if condition.parent.name != hyperparameter_name and condition.child.name != hyperparameter_name:
            config_space_prime.add_condition(condition)
        else:
            raise ValueError()
    return config_space_prime


def get_search_space(search_space, type):
    name = 'get_' + search_space + '_' + type + '_search_space'
    return getattr(openmldefaults.config_spaces, name)()


def post_process(value):
    # TODO: get this info from config space?!
    if value is "None":
        value = None
    elif value is "True":
        value = True
    elif value is "False":
        value = False

    if not (isinstance(value, (str, bool, int, type(None))) or np.issubdtype(type(value), np.number)):
        raise ValueError('unsupported type: %s' % type(value))

    if isinstance(value, (str, bool, type(None))):
        value = json.dumps(value)
    return value


def reinstantiate_parameter_value(value):
    if not (isinstance(value, str) or np.issubdtype(type(value), np.number)):
        raise ValueError('unsupported type: %s' % type(value))

    if np.issubdtype(type(value), np.int):
        return int(value)
    elif np.issubdtype(type(value), np.float):
        return float(value)
    return json.loads(value)


def prefix_hyperparameter_name(hyperparameter: ConfigSpace.hyperparameters.Hyperparameter):
    return hyperparameter.meta['component'] + '__' + hyperparameter.name


def dict_to_prefixed_dict(hyperparameter_value, config_space):
    result = dict()
    for hyperparameter, value in hyperparameter_value.items():
        param = config_space.get_hyperparameter(hyperparameter)
        result[prefix_hyperparameter_name(param)] = value
    return result

