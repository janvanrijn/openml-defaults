import typing


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
