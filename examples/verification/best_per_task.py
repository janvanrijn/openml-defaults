import argparse
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import openmldefaults
import os
import pandas as pd
import typing


def parse_args():
    metadata_file_svc = os.path.expanduser('~/data/openml-defaults/svc.arff')
    metadata_file_gb = os.path.expanduser('~/data/openml-defaults/gradient_boosting.arff')
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/stats')
    parser.add_argument('--metadata_files', type=str, nargs='+', default=[metadata_file_svc, metadata_file_gb])
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--search_space_identifier', type=str, default=None)
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.info('Started %s' % os.path.basename(__file__))

    usercpu_time = 'usercpu_time_millis'
    metadata_frame = openmldefaults.utils.metadata_files_to_frame(args.metadata_files,
                                                                  args.search_space_identifier,
                                                                  [args.scoring, usercpu_time])
    print(metadata_frame.columns.values)
    metadata_frame = metadata_frame.groupby(['task_id', 'classifier'])[args.scoring].max().reset_index()
    fig, ax = plt.subplots()

    # absolute plots
    os.makedirs(args.output_file, exist_ok=True)
    sns.boxplot(x="classifier", y=args.scoring, data=metadata_frame, ax=ax)
    output_file = os.path.join(args.output_directory, 'performances.png')
    plt.savefig(output_file)
    logging.info('stored figure to %s' % output_file)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
