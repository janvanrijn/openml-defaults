import ConfigSpace
import logging
import openmldefaults
import pandas as pd
import time
import typing

from openmldefaults.models.defaults_generator_interface import DefaultsGenerator


class ActiveTestingDefaults(DefaultsGenerator):

    def __init__(self):
        self.name = 'active_testing'

    def generate_defaults_discretized(self, df: pd.DataFrame, num_defaults: int,
                                      minimize: bool, aggregate: typing.Callable,
                                      config_space: ConfigSpace.ConfigurationSpace,
                                      raise_no_improvement: bool) \
            -> typing.Tuple[typing.List, typing.Dict[str, typing.Any]]:
        """
        Takes a data frame with a discretized set of defaults and returns the
        result of active testing (i.e., the greedy approach until that is ex-
        hausted, continuing with the average rank)

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
        selected_indices: List[int]
            List of indices, as given by the dataframe
        results_dict: Dict[str, Any]
            Additional meta-information. Containing at least the key 'run_time',
            but potentially more information
        """
        logging.info('Started %s, dimensions config frame %s' % (self.name,
                                                                 str(df.shape)))
        if num_defaults < 1:
            raise ValueError()
        start_time = time.time()

        selected_indices, _ = openmldefaults.models.GreedyDefaults().generate_defaults_discretized(
            df, num_defaults, minimize, aggregate, config_space, raise_no_improvement)
        ar_indices, _ = openmldefaults.models.AverageRankDefaults().generate_defaults_discretized(
            df, num_defaults, minimize, aggregate, config_space, raise_no_improvement)

        for index in ar_indices:
            if index not in selected_indices:
                selected_indices.append(index)

        results_dict = {
            'run_time': time.time() - start_time,
        }
        return selected_indices, results_dict
