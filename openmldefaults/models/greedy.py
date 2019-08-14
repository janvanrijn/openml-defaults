import ConfigSpace
import logging

import openmldefaults
import pandas as pd
import time
import typing

from openmldefaults.models.defaults_generator_interface import DefaultsGenerator


class GreedyDefaults(DefaultsGenerator):

    def __init__(self):
        self.name = 'greedy'

    @staticmethod
    def find_best_competitor(df: pd.DataFrame,
                             best_score: float,
                             selected_indices: typing.List[int],
                             aggregate: typing.Callable,
                             minimize: bool) -> typing.Tuple[float, int]:
        best_index = None
        for idx, current_config in enumerate(df.index.values):
            per_task_scores = openmldefaults.utils.selected_set_index(df, selected_indices + [idx], minimize)
            current_score = aggregate(per_task_scores)
            if best_score is None \
                    or (minimize and current_score < best_score) \
                    or ((not minimize) and current_score > best_score):
                best_score = current_score
                best_index = idx
        return best_score, best_index

    def generate_defaults_discretized(self, df: pd.DataFrame, num_defaults: int,
                                      minimize: bool, aggregate: typing.Callable,
                                      config_space: ConfigSpace.ConfigurationSpace,
                                      raise_no_improvement: bool) \
            -> typing.Tuple[typing.List, typing.Dict[str, typing.Any]]:
        """
        Takes a data frame with a discretized set of defaults and returns the
        greedy defaults. The data frame should be structured as follows: each
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

        selected_indices = []
        best_score = None
        for itt_defaults in range(num_defaults):
            best_score, best_index = GreedyDefaults.find_best_competitor(df,
                                                                         best_score,
                                                                         selected_indices,
                                                                         aggregate,
                                                                         minimize)
            if best_index is None:
                if raise_no_improvement:
                    raise ValueError('Could not add default, as there were no '
                                     'configurations that yield improvement after '
                                     '%d defaults' % len(selected_indices))
            else:
                selected_indices.append(best_index)

        results_dict = {
            'run_time': time.time() - start_time,
        }
        return selected_indices, results_dict
