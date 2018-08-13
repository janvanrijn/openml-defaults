import openmldefaults
import time


class GreedyDefaults(object):

    def __init__(self):
        self.name = 'greedy'

    def generate_defaults(self, df, num_defaults):
        print(openmldefaults.utils.get_time(), 'Started %s' % self.name)
        start_time = time.time()

        selected_configs = []
        selected_indices = []
        for itt_defaults in range(num_defaults):
            best_addition = None
            best_score = None
            best_index = None
            for idx, current_config in enumerate(df.index.values):
                current_score = sum(openmldefaults.utils.selected_set_index(df, selected_indices + [idx]))
                if best_score is None or current_score < best_score:
                    best_score = current_score
                    best_addition = current_config
                    best_index = idx
            selected_configs.append(best_addition)
            selected_indices.append(best_index)

        selected_defaults = [openmldefaults.utils.selected_row_to_config_dict(df, idx) for idx in selected_indices]
        print(openmldefaults.utils.get_time(), selected_defaults)

        runtime = time.time() - start_time
        results_dict = {
            'defaults': selected_defaults,
            'indices': selected_indices,
            'objective': best_score,
            'run_time': runtime,
        }
        return results_dict
