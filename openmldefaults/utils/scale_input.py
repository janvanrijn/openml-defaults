import numpy as np
import pandas as pd

from typing import List, Tuple


def selected_set(df: pd.DataFrame, defaults: List[Tuple], column_slice: List=None):
    # filters out only the algorithms that we have in the 'set of defaults'
    df = df.loc[list(defaults)]
    if column_slice is not None:
        df = df.iloc[:, column_slice]
    # df.min(axis=0) returns per dataset the minimum score obtained by 'set of defaults'
    # then we take the median of this
    result = df.min(axis=0)
    if np.isnan(sum(result)):
        raise ValueError('None of the results of this function should be NaN')
    return result


def selected_set_index(df: pd.DataFrame, indices: List[int], minimize: bool) -> List[float]:
    """
    Convenience function. Returns per row the (minimum, maximum) of a selected
    set of columns.

    Parameters
    ----------
    df: pd.DataFrame
        A dataframe with each column representing a dataset, and each row
        representing a configuration.
    indices: List
        The rows to select
    minimize: bool
        Whether to return the sum of column-wise minimum or the sum of
        column-wise maximum

    Returns
    -------
    List[float]
         per column (minimum, maximum) of the selected rows
    """
    is_series = isinstance(df, pd.Series)
    # filters out only the algorithms that we have in the 'set of defaults'
    df = df.iloc[indices]
    # df.min(axis=0) returns per dataset the minimum score obtained by 'set of defaults'
    # then we take the median of this
    if minimize:
        result = df.min(axis=0)
    else:
        result = df.max(axis=0)
    if is_series:
        result = [result]
    if np.isnan(sum(result)):
        raise ValueError('None of the results of this function should be NaN')
    return result


def reshape_configs(df, params, resized_grid_size):
    # subsample the hyperparameter grid
    for param in params:
        unique = np.array(df[param].unique())
        if len(unique) > resized_grid_size:
            interval = int(np.ceil(len(unique) / resized_grid_size))
            resized = unique[0::interval]
            assert len(resized) <= resized_grid_size, 'Param %s, originally %d, new size: %d' % (
            param, len(unique), len(resized))
            df = df.loc[df[param].isin(resized)]
    return df
