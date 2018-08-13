import json
import numpy as np
import openmldefaults


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


def prefix(prefix_str, show):
    if show:
        return prefix_str + "__"
    else:
        return ""


def prefix_mapping(config_space_fn):
    """
    Returns a dict mapping names of config space without prefix to names of config space with prefix
    """
    config_space_with_prefix = config_space_fn(True)
    config_space_without_prefix = config_space_fn(False)

    mapping = dict()
    for hyperparameter in config_space_with_prefix.get_hyperparameters():
        for alternative in config_space_without_prefix.get_hyperparameters():
            if hyperparameter.name.endswith(alternative.name):
                if alternative.name in mapping:
                    raise ValueError()
                else:
                    mapping[alternative.name] = hyperparameter.name

    return mapping
