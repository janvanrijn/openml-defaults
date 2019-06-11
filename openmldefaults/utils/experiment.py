import ConfigSpace
import numpy as np
import openmlcontrib
import pandas as pd
import typing


def selected_row_to_config_dict(df: pd.DataFrame, row_idx: int, config_space: ConfigSpace.ConfigurationSpace) -> typing.Dict:
    values = df.index[row_idx]
    if isinstance(df.index, pd.core.index.MultiIndex):
        keys = df.index.names
    else:
        values = [values]
        keys = [df.index.name]
    if not isinstance(keys, list):
        raise ValueError('data frame index not interpreted properly')
    if len(keys) != len(values):
        raise ValueError()
    result = dict()
    for name, value in zip(keys, values):
        dtype_callable = openmlcontrib.legacy.get_hyperparameter_datatype(config_space.get_hyperparameter(name))
        if isinstance(value, float) and np.isnan(value):
            # JvR: I think it's never good to add a NaN to a parameter grid
            # It usually indicates a inactive hyperparameter, and when added
            # explicitly it can crash other libraries (SVC degree)
            # Although there are exceptions where the NaN is actually an
            # active hyperparameter (Imputer) we won't need it on the param grid
            pass
        else:
            result[name] = dtype_callable(value)
    return result
