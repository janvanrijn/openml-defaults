import hashlib
import numpy as np
import pandas as pd
import sklearn.preprocessing


def hash_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def normalize_df_columnwise(df: pd.DataFrame) -> pd.DataFrame:
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np_scaled)
    return df_normalized


def create_a3r_frame(scoring_frame: pd.DataFrame, runtime_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces all occurrences of zero in the runtime frame with the min val (to
    prevent division by zero) and uses this frame to create a a3r frame.
    """
    min_val = np.min(runtime_frame.values[np.nonzero(runtime_frame.values)])
    runtime_frame = runtime_frame.replace(0, min_val)
    return scoring_frame.divide(runtime_frame)
