import numpy as np
import pandas as pd

from typing import List, Tuple


def selected_set(df: pd.DataFrame, defaults: List[Tuple]):
    # filters out only the algorithms that we have in the 'set of defaults'
    df = df.loc[list(defaults)]
    # df.min(axis=0) returns per dataset the minimum score obtained by 'set of defaults'
    # then we take the median of this
    return df.min(axis=0)


def selected_set_index(df: pd.DataFrame, indices: List[int]):
    # filters out only the algorithms that we have in the 'set of defaults'
    df = df.iloc[indices]
    # df.min(axis=0) returns per dataset the minimum score obtained by 'set of defaults'
    # then we take the median of this
    return df.min(axis=0)


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
