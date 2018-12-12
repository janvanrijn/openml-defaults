import logging
import openmldefaults
import pandas as pd
import time


class GreedyDefaults(object):

    def __init__(self):
        self.name = 'greedy'

    def generate_defaults(self, df: pd.DataFrame, num_defaults: int, minimize: bool):
        """
        Takes a data frame and returns the greedy defaults. The data frame
        should be structured as follows: each column represents a task, each row
        represents a configuration. As such, each cell represents the evaluation
        of that configuration on that task. The sum of the selected configurations
        across tasks is to be minimized or maximized

        Parameters
        ----------
        df: pd.DataFrame
            The data frame as described above

        num_defaults: int
            The number of configurations to be selected

        minimize: bool
            Will minimize the objective function iff this is true.

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

        selected_configs = []
        selected_indices = []
        best_score = None
        for itt_defaults in range(num_defaults):
            best_addition = None
            best_index = None
            for idx, current_config in enumerate(df.index.values):
                current_score = sum(openmldefaults.utils.selected_set_index(df, selected_indices + [idx], minimize))
                if best_score is None \
                        or (minimize and current_score < best_score) \
                        or ((not minimize) and current_score > best_score):
                    best_score = current_score
                    best_addition = current_config
                    best_index = idx
            if best_addition is None:
                raise ValueError('Could not add default, as there were no '
                                 'configurations that yield improvement after '
                                 '%d defaults' % len(selected_configs))
            selected_configs.append(best_addition)
            selected_indices.append(best_index)

        selected_defaults = [openmldefaults.utils.selected_row_to_config_dict(df, idx) for idx in selected_indices]
        logging.info(selected_defaults)

        runtime = time.time() - start_time
        results_dict = {
            'defaults': selected_defaults,
            'indices': selected_indices,
            'objective': best_score,
            'run_time': runtime,
        }
        return results_dict
