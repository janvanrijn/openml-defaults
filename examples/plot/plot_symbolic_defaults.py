import argparse
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import openmldefaults
import os
import pandas as pd
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--results_directory', type=str, default=os.path.expanduser('~/experiments/openml-defaults/symbolic_defaults/svc/'))
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--n_defaults_in_file', type=int, default=32)
    return parser.parse_args()


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for basefilename in os.listdir(args.results_directory):
        filepath = os.path.join(args.results_directory, basefilename)
        filename_base = os.path.splitext(basefilename)[0]
        task_id = filename_base.split('_')[-1]
        with open(filepath, 'rb') as fp:
            results = pickle.load(fp)

        best = None
        for res in results['symbolic_defaults']:
            if best is None:
                best = res
            elif res['avg_performance'] > best['avg_performance']:
                best = res
        print(best)

        if task_id == 'all':
            pass
        else:
            task_id = int(task_id)  # throws an error in case of an unexpected file


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
