import hashlib
import logging
import numpy as np
import pandas as pd
import sklearn.base
import sklearn.preprocessing
import typing


def hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def get_scaler(scaling_type: typing.Optional[str]) -> sklearn.base.TransformerMixin:
    legal_types = {
        'MinMaxScaler': sklearn.preprocessing.MinMaxScaler(),
        'StandardScaler': sklearn.preprocessing.StandardScaler()
    }
    if scaling_type not in legal_types:
        raise ValueError('Can not find scaling type: %s' % scaling_type)
    return legal_types[scaling_type]


def normalize_df_columnwise(df: pd.DataFrame, scaling_type: typing.Optional[str]) -> pd.DataFrame:
    logging.info('scaling dataframe (no specific measure indicated) with %s' % scaling_type)
    if scaling_type is None:
        return df

    scaler = get_scaler(scaling_type)
    for column in df.columns.values:
        res = scaler.fit_transform(df[column].values.reshape(-1, 1))[:, 0]
        df[column] = res
    return df


def normalize_df_conditioned_on(df_orig: pd.DataFrame,
                                scaling_type: typing.Optional[str],
                                scaling_col: str,
                                conditioned_on: str) -> pd.DataFrame:
    logging.info('scaling dataframe (no specific measure indicated) with %s' % scaling_type)
    df_copy = df_orig.copy(True)
    if scaling_type is None:
        return df_copy
    if scaling_col == conditioned_on:
        raise ValueError('Scaling column and conditioned on column can not be the same. ')
    scaler = get_scaler(scaling_type)
    for value in df_orig[conditioned_on].unique():
        array = df_orig.loc[df_orig[conditioned_on] == value][scaling_col].values
        res = scaler.fit_transform(array.reshape(-1, 1))[:, 0]
        df_copy.loc[df_orig[conditioned_on] == value, scaling_col] = res
    return df_copy


def create_a3r_frame(scoring_frame: pd.DataFrame, runtime_frame: pd.DataFrame, a3r_r: int) -> pd.DataFrame:
    """
    Replaces all occurrences of zero in the runtime frame with the min val (to
    prevent division by zero) and uses accuracy and runtime frames to create an
    a3r frame.
    """
    min_val = np.min(runtime_frame.values[np.nonzero(runtime_frame.values)])
    runtime_frame = runtime_frame.replace(0, min_val)
    assert(np.array_equal(scoring_frame.columns.values, runtime_frame.columns.values))
    assert(np.array_equal(scoring_frame.shape, runtime_frame.shape))
    runtime_frame = runtime_frame ** (1/a3r_r)
    return scoring_frame / runtime_frame
