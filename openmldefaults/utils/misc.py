import ConfigSpace
import csv
import logging
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
            raise ValueError('Hyperparameter %s can not be removed as it is part of a condition' % hyperparameter_name)
    return config_space_prime


def _traverse_run_folders(folder: str, n_defaults: int, traversed_directories: typing.List[str],
                          constraints: typing.Optional[typing.Dict[int, typing.List[str]]], log: bool) \
        -> typing.List[typing.List[str]]:
    if log:
        depth = len(traversed_directories) + 1
        logging.info('-' * depth + ': ' + folder)
    folder_content = os.listdir(folder)
    if 'surrogated_%d_1.csv' % n_defaults in folder_content:
        return [traversed_directories + ['surrogated_%d_1.csv' % n_defaults]]
    elif 'surrogated_%d_0.csv' % n_defaults in folder_content:
        return [traversed_directories + ['surrogated_%d_0.csv' % n_defaults]]
    else:
        depth = len(traversed_directories)
        results = []
        for item in folder_content:
            subfolder = os.path.join(folder, item)
            if constraints is not None and depth in constraints:
                if item not in constraints[depth]:
                    # skip this folder
                    logging.info('skipping folder %s' % subfolder)
                    continue
            if os.path.isdir(subfolder):
                results += _traverse_run_folders(subfolder, n_defaults, traversed_directories + [item], constraints, log)
        return results


def results_from_folder_to_df(folder: str, n_defaults_in_file: int, budget: int,
                              constraints: typing.Optional[typing.Dict[int, typing.List[str]]],
                              raise_if_not_enough: bool, log: bool) \
        -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
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

    log: bool
        if set, the script outputs each traversed directory

    Returns
    -------
    pd.DataFrame
        a data frame with columns for each folder level and the metric_fn as y
    pd.DataFrame
        a data frame with columns for each folder level and the loss curve (in a
        single cell)
    """
    list_dirs_runs = _traverse_run_folders(folder, n_defaults_in_file, list(), constraints, log)
    results_vanilla = []
    results_curves = []
    for dirs in list_dirs_runs:
        current_vanilla = {'folder_depth_%d' % idx: folder for idx, folder in enumerate(dirs[:-1])}
        current_path = '/'.join(dirs)
        with open(os.path.join(folder, current_path), 'r') as fp:
            reader = csv.DictReader(fp)
            curve = list()
            for idx, row in enumerate(reader):
                if idx > budget:
                    # only break if curve is bigger than budget (since the n-th
                    # line represents the case of "budget" = n, in particular
                    # line 0 represents budget = 0)
                    break
                curve.append(row)
                current_curve_point = dict(current_vanilla)
                current_curve_point.update(row)
                results_curves.append(current_curve_point)
        current_vanilla.update(curve[-1])
        current_vanilla['n_defaults'] = len(curve) - 1  # same reason
        if len(curve) - 1 > budget:
            # Should never happen
            raise ValueError()
        if len(curve) - 1 < budget and raise_if_not_enough:
            raise ValueError('Not enough curve points')
        results_vanilla.append(current_vanilla)
    return pd.DataFrame(results_vanilla), pd.DataFrame(results_curves)


def openml_measure_to_sklearn(openml_measure: str) -> typing.Tuple[str, bool]:
    if openml_measure == 'predictive_accuracy':
        return 'accuracy', True
    else:
        raise ValueError('Unimplemented measure')
