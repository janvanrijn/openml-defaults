import argparse
import logging

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import openmldefaults
import os
import pandas as pd
import seaborn as sns


def parse_args():
    metadata_file_text_classification = os.path.expanduser('../../data/text_classification.arff')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/vanilla_defaults_vs_rs/')
    parser.add_argument('--metadata_files', type=str, nargs='+', default=[metadata_file_text_classification])
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--plot_extension', type=str, default='pdf')
    parser.add_argument('--hyperparameter', type=str, default='text_classification:algorithm')
    parser.add_argument('--search_space_identifier', type=str, default='ferreira')
    parser.add_argument('--task_id_column', default='dataset', type=str)
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    metadata_frame = openmldefaults.utils.metadata_files_to_frame(args.metadata_files,
                                                                  args.search_space_identifier,
                                                                  [args.scoring],
                                                                  args.task_id_column,
                                                                  True)
    result_series = metadata_frame.groupby([args.hyperparameter, args.task_id_column])[args.scoring].mean()
    result_frame = result_series.reset_index()
    fig, ax1 = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=args.hyperparameter, y=args.scoring, data=result_frame, ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_xlabel(args.hyperparameter.split(':')[-1])
    ax1.set_ylabel(args.scoring.replace('_', ' '))
    output_file = os.path.join(args.output_directory, 'marginal_%s.%s' % (args.hyperparameter, args.plot_extension))
    plt.tight_layout()
    plt.savefig(output_file)
    logging.info('stored figure to %s' % output_file)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
