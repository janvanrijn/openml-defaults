import argparse
import os
import pandas as pd
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--defaults_path', type=str,
                        default=os.path.expanduser('~/experiments/openml-defaults/defaults_32_0.pkl'))
    parser.add_argument('--num_configurations', type=int, default=5)
    parser.add_argument('--search_space', type=str, default='resnet')
    return parser.parse_args()


def run(args):
    with open(args.defaults_path, 'rb') as fp:
        defaults = pickle.load(fp)['defaults']
    results = []
    header = []
    col_format = 'X'
    for idx, default in enumerate(defaults):
        if idx >= args.num_configurations:
            continue
        default_renamed = dict()
        for key, value in default.items():
            splitted = key.split(':')
            if len(splitted) > 2:
                raise ValueError()
            if splitted[0] == args.search_space:
                param_name = splitted[1].replace('_', ' ').capitalize()
                default_renamed[param_name] = value
        results.append(default_renamed)
        header.append('Default #%d' % (idx + 1))
        col_format += '@{\hspace{0.4cm}}r'
    results = pd.DataFrame(results).T
    print(results.to_latex(float_format=lambda x: '%.3e' % x, header=header, column_format=col_format))
    pass


if __name__ == '__main__':
    run(parse_args())
