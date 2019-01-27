import openmldefaults
import typing
import sklearn.pipeline
import sklearnbot


def convert_defaults_to_param_grid(defaults: typing.List[typing.Dict[str, typing.Union[str, int, bool, float]]]) \
        -> typing.List[typing.Dict[str, typing.List]]:
    """
    Converts the default format into the parameter grid format required to run
    the sklearn.model_selection.GridSearchCV on it.


    Parameters
    ----------
    defaults: list[dict[str, mixed]]
        A list containing various dictionaries, each dictionary representing
        a single configuration
    """
    param_grid = list()
    for default in defaults:
        param_grid.append({k: [v] for k, v in default.items()})
    return param_grid


def convert_defaults_to_multiple_param_grids(defaults: typing.List[typing.Dict[str, typing.Union[str, int, bool, float]]],
                                             classifier_identifying_param: typing.Optional[str],
                                             search_space_identifier: str,
                                             numeric_indices: typing.List[int], nominal_indices: typing.List[int]) \
        -> typing.Tuple[typing.Dict[str, sklearn.base.BaseEstimator],
                        typing.Dict[str, typing.List[typing.Dict[str, typing.List]]]]:
    classifiers = dict()
    param_grids = dict()
    for default in defaults:
        classifier = default[classifier_identifying_param]
        current = {key.split(':')[1]: value for key, value in default.items() if key.split(':')[0] == classifier}
        if classifier not in param_grids:
            param_grids[classifier] = list()
        param_grids[classifier].append(current)
    for key in param_grids:
        config_space = openmldefaults.config_spaces.get_config_space(key, 0, search_space_identifier)
        param_grids[key] = convert_defaults_to_param_grid(param_grids[key])
        classifiers[key] = sklearnbot.sklearn.as_estimator(config_space, numeric_indices, nominal_indices)
    return classifiers, param_grids
