import argparse
import numpy as np
import feather
import os
import pandas as pd

from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=os.path.expanduser('~') + '/Downloads/mlr.classif.rpart.feather')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=5)
    return parser.parse_args()


def score_set(df: pd.DataFrame, indices: List[int]):
    # filters out only the algorithms that we have in the 'set of defaults'
    df = df.iloc[indices]
    # df.min(axis=0) returns per dataset the minimum score obtained by 'set of defaults'
    # then we take the median of this
    return df.min(axis=0).median()


def run(args):
    df = feather.read_dataframe(args.dataset_path)

    if args.resized_grid_size is not None:
        # subsample the hyperparameter grid
        for param in args.params:
            unique = np.array(df[param].unique())
            interval = int(np.ceil(len(unique) / args.resized_grid_size))
            resized = unique[0::interval]
            assert len(resized) == args.resized_grid_size
            df = df.loc[df[param].isin(resized)]

    df = df.set_index(args.params)
    for idx in range(len(df)):
        result = score_set(df, [idx])
        print(idx, result)


if __name__ == '__main__':
    run(parse_args())
