import argparse
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import os
import pandas as pd
import typing

import sys
sys.path.append('/home/flo/Documents/projects/sklearn-bot')
sys.path.append('/home/flo/Documents/projects/openml-python-contrib')
sys.path.append('/home/flo/Documents/projects/openml-defaults')
import openmldefaults


# __file__ = ''

def parse_args():
    # metadata_file_svc = os.path.expanduser('~/Documents/projects/openml-defaults/data/svc.arff')
    # metadata_file_gb = os.path.expanduser('~/data/openml-defaults/gradient_boosting.arff')
    metadata_file_svm = os.path.expanduser('~/Documents/projects/openml-defaults/data/small/classif_svm.arff')
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/stats')
    parser.add_argument('--metadata_files', type=str, nargs='+', default=[metadata_file_svm])
    # parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--scoring', type=str, default='perf.mmce')
    parser.add_argument('--search_space_identifier', type=str, default=None)
    parser.add_argument('--task_id_column', default='task_id', type=str)
    parser.add_argument('--skip_row_check', action='store_true')
    args_ = parser.parse_args()
    return args_


def run(args):
    args = parse_args()
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.info('Started %s with args %s' % (os.path.basename(__file__), args))
    os.makedirs(args.output_directory, exist_ok=True)
    usercpu_time = 'usercpu_time_millis'
    identifier = args.search_space_identifier if args.search_space_identifier is not None else 'full'
    metadata_frame = openmldefaults.utils.metadata_files_to_frame(args.metadata_files,
                                                                  args.search_space_identifier,
                                                                  [args.scoring],
                                                                  task_id_column=args.task_id_column,
                                                                  skip_row_check=args.skip_row_check
                                                                  )
    best_frame = metadata_frame.groupby([args.task_id_column, 'classifier'])[args.scoring].max().reset_index()
    #  plots
    fig, ax = plt.subplots()
    sns.boxplot(x="classifier", y=args.scoring, data=best_frame, ax=ax)
    output_file = os.path.join(args.output_directory, 'performances_%s.png' % identifier)
    plt.savefig(output_file)
    logging.info('stored figure to %s' % output_file)
    fig, ax = plt.subplots(figsize=(48, 6))
    sns.boxplot(x=args.task_id_column, y=args.scoring, hue='classifier', data=metadata_frame, ax=ax)
    output_file = os.path.join(args.output_directory, 'spread_%s.png' % identifier)
    plt.savefig(output_file)
    logging.info('stored figure to %s' % output_file)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
