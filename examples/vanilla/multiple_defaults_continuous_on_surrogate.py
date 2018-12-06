import argparse
import ConfigSpace
import logging
import numpy as np
import openml
import openmlcontrib
import openmldefaults
import os
import pandas as pd
import pickle
import sklearn
import sklearnbot
import typing


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    metadata_file = '/home/janvanrijn/experiments/sklearn-bot/results/results__500__svc__predictive_accuracy.arff'
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/vanilla_defaults/')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--task_idx', type=int, default=None)
    parser.add_argument('--metadata_file', type=str, default=metadata_file)
    parser.add_argument('--classifier_name', type=str, default='svc', help='scikit-learn flow name')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    config_space = sklearnbot.config_spaces.get_config_space(args.classifier_name, args.random_seed)

    metadata_atts = openmldefaults.utils.get_dataset_metadata(args.metadata_file)
    if args.scoring not in metadata_atts['measure']:
        raise ValueError('Could not find measure: %s' % args.scoring)
    metadata_frame = openmldefaults.utils.metadata_file_to_frame(args.metadata_file, config_space, args.scoring)

    config_frame = openmldefaults.utils.generate_grid_dataset(metadata_frame,
                                                              config_space,
                                                              args.resized_grid_size,
                                                              args.scoring,
                                                              args.random_seed)
    model = openmldefaults.models.GreedyDefaults()
    model.generate_defaults(config_frame, 32, False)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
