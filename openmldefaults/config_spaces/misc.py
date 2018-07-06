

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
