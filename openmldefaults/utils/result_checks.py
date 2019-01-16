import logging
import numpy as np
import pandas as pd
import typing


def check_budget_curves(df: pd.DataFrame,
                        index_columns: typing.List,
                        values_column: str,
                        pivot_column: str,
                        expected_minimum: typing.Optional[float],
                        expected_maximum: typing.Optional[float]):
    """
    Checks a dataframe with resulting columns whether the result curves are
    strictly increasing (can only be guaranteed on validation set, not on test
    set.) Also checks whether all values are within expected ranges.
    """
    results_pivot = df.pivot_table(
        values=values_column,
        index=index_columns,
        columns=pivot_column,
        aggfunc=np.mean)

    last_column = 0.0
    for column in results_pivot.columns.values:
        if last_column > column:
            logging.warning('Weird column order: %s' % results_pivot.columns.values)
        last_column = int(column)

    for index, row in results_pivot.iterrows():
        last_value = 0.0
        for column, value in row.iteritems():
            if expected_minimum is not None and value < expected_minimum:
                msg = 'Value (%f) lower than expected minimum (%f) budget %d for %s with %s.' \
                      % (value, expected_minimum, column, values_column, index)
                raise ValueError(msg)
            if expected_maximum is not None and value > expected_maximum:
                msg = 'Value (%f) higher than expected maximum (%f) budget %d for %s with %s.' \
                      % (value, expected_maximum, column, values_column, index)
                raise ValueError(msg)
            if value < last_value:
                msg = 'Series not strictly improving at budget %d for %s with %s. Previous %f, current %f' \
                      % (column, values_column, index, last_value, value)
                raise ValueError(msg)
            last_value = value
