import logging
import numpy as np
import pandas as pd
import typing


def check_budget_curves(df: pd.DataFrame, index_columns: typing.List, values_column: str, pivot_column: str):
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
            if value < last_value:
                msg = 'Series not strictly improving at budget %d for %s with %s. Previous %f, current %f' \
                      % (column, values_column, index, last_value, value)
                raise ValueError(msg)
            last_value = value
