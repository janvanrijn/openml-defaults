import argparse
import logging
import numpy as np
import openml
import openmlcontrib
import openmldefaults
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/openml_cache',
                        help='directory to store cache')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/symbolic_defaults/')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--classifier', type=str, default='libsvm_svc', help='scikit-learn flow name')
    parser.add_argument('--config_space', type=str, default='micro', help='config space type')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--num_runs', type=int, default=500, help='max runs to obtain from openml')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    return parser.parse_args()


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    study = openml.study.get_study(args.study_id, 'tasks')
    config_space_fn = getattr(openmldefaults.config_spaces,
                              'get_%s_%s_search_space' % (args.classifier,
                                                          args.config_space))
    config_space = config_space_fn()
    configurations = openmldefaults.utils.generate_grid_configurations(config_space, 0, args.resized_grid_size)

    config_frame = pd.DataFrame(configurations)
    surrogates = dict()
    for task_id in study.tasks:
        logging.info('Training surrogate on Task %d' % task_id)
        estimator, columns = openmldefaults.utils.train_surrogate_on_task(task_id,
                                                                          config_space.meta['flow_id'],
                                                                          args.num_runs,
                                                                          config_space,
                                                                          args.scoring,
                                                                          args.cache_directory)
        if not np.array_equal(config_frame.columns.values, columns):
            # if this goes wrong, it is due to the pd.get_dummies() fn
            raise ValueError('Column set not equal: %s vs %s' % (config_frame.columns.values, columns))
        surrogates[task_id] = estimator
    print(config_frame)


if __name__ == '__main__':
    run(parse_args())
