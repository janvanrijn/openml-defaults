import ConfigSpace
import logging
import numpy as np
import openmldefaults
import pandas as pd
import time
import typing

from openmldefaults.models.defaults_generator_interface import DefaultsGenerator


class AverageRankDefaults(DefaultsGenerator):

    def __init__(self):
        self.name = 'average_rank'

    def generate_defaults_discretized(self, df: pd.DataFrame, num_defaults: int,
                                      minimize: bool, aggregate: typing.Callable,
                                      config_space: ConfigSpace.ConfigurationSpace,
                                      raise_no_improvement: bool):
        """
        Takes a data frame with a discretized set of defaults and returns the
        average rank. The data frame should be structured as follows: each
        column represents a task, each row represents a configuration. As such,
        each cell represents the evaluation of that configuration on that task.
        The sum of the selected configurations across tasks is to be minimized
        or maximized

        Parameters
        ----------
        df: pd.DataFrame
            The data frame as described above

        num_defaults: int
            The number of configurations to be selected

        minimize: bool
            Will minimize the objective function iff this is true.

        aggregate: callable
            function to aggregate per task results

        config_space: ConfigurationSpace
            the configuration space object corresponding to the defaults. Will
            be used casting defaults to the right data type.

        raise_no_improvement: bool
            if true, an error will be raised if no improvement is obtained
            before the correct number of default was obtained

        Returns
        -------
        results_dict: dict
            A dictionary containing the defaults, indices of the defaults,
            objective score and the run time of this algorithm.
        """
        logging.info('Started %s, dimensions config frame %s' % (self.name,
                                                                 str(df.shape)))
        if num_defaults < 1:
            raise ValueError()
        start_time = time.time()

        avg_ranks = df.rank(axis=0, method='average', ascending=(not minimize)).sum(axis=1) / df.shape[1]
        selected_indices = np.argsort(avg_ranks.values)

        selected_defaults = [openmldefaults.utils.selected_row_to_config_dict(df, idx, config_space) for idx in selected_indices]
        logging.info(selected_defaults)

        runtime = time.time() - start_time
        results_dict = {
            'defaults': selected_defaults,
            'indices': selected_indices,
            'run_time': runtime,
        }
        return results_dict
