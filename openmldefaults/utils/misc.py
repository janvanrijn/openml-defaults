import ConfigSpace
import csv
import os
import pandas as pd
import typing


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


def _traverse_run_folders(folder: str, n_defaults: int, traversed_directories: typing.List[str],
                          constraints: typing.Optional[typing.Dict[int, typing.List[str]]]) \
        -> typing.List[typing.List[str]]:
    folder_content = os.listdir(folder)
    if 'results_%d_1.csv' % n_defaults in folder_content:
        return [traversed_directories + ['results_%d_1.csv' % n_defaults]]
    elif 'results_%d_0.csv' % n_defaults in folder_content:
        return [traversed_directories + ['results_%d_0.csv' % n_defaults]]
    else:
        depth = len(traversed_directories)
        results = []
        for item in folder_content:
            if constraints is not None and depth in constraints:
                if item not in constraints[depth]:
                    # skip this folder
                    continue
            subfolder = os.path.join(folder, item)
            if os.path.isdir(subfolder):
                results += _traverse_run_folders(subfolder, n_defaults, traversed_directories + [item], constraints)
        return results


def results_from_folder_to_df(folder: str, n_defaults_in_file: int, budget: int,
                              constraints: typing.Optional[typing.Dict[int, typing.List[str]]],
                              raise_if_not_enough: bool) \
        -> pd.DataFrame:
    """
    Traverses all subdirecties, and obtains stored runs with evaluation scores.

    Parameters
    ----------
    folder: str
        The folder that should be traversed (recursively)

    n_defaults_in_file: str
        max number of defaults to be found

    budget: int
        Number of iterations

    constraints: dict (optional)
        dict mapping from a integer, representing the directory level at which
        a filter is active to a list of allowed values

    raise_if_not_enough: bool
        If true, it raises an error when there are not enough results

    Returns
    -------
    df : pd.DataFrame
        a data frame with columns for each folder level and the metric_fn as y
    """
    list_dirs_runs = _traverse_run_folders(folder, n_defaults_in_file, list(), constraints)
    results = []
    for dirs in list_dirs_runs:
        current = {'folder_depth_%d' % idx: folder for idx, folder in enumerate(dirs[:-1])}
        current_path = '/'.join(dirs)
        with open(os.path.join(folder, current_path), 'r') as fp:
            reader = csv.DictReader(fp)
            row = next(reader)  # if budget were to be zero
            n_defaults = 0
            for _ in range(budget):
                try:
                    row = next(reader)
                    n_defaults += 1
                except StopIteration as e:
                    if raise_if_not_enough:
                        raise e
            current.update(row)
            current['n_defaults'] = n_defaults
        results.append(current)
    return pd.DataFrame(results)
