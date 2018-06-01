import argparse
import feather
import numpy as np
import openmldefaults
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/train_svm.feather')
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=5)
    parser.add_argument('--restricted_num_tasks', type=int, default=5)
    return parser.parse_args()


def run(args):
    df = feather.read_dataframe(args.dataset_path)
    print(openmldefaults.utils.get_time(), 'Original data frame dimensions:', df.shape)

    if args.resized_grid_size is not None:
        df = openmldefaults.utils.reshape_configs(df, args.params, args.resized_grid_size)

    # always set the index
    df = df.set_index(args.params)

    if args.restricted_num_tasks is not None:
        # subsample num tasks
        df = df.iloc[:, 0:args.restricted_num_tasks]

    df, dominated = openmldefaults.utils.simple_cull(df, openmldefaults.utils.dominates_min)
    print(openmldefaults.utils.get_time(),
          'Dominated Configurations: %d/%d' % (len(dominated), len(df) + len(dominated)))

    # sort configurations by 'good ones'
    df['sum_of_columns'] = df.apply(lambda row: sum(row), axis=1)
    df = df.sort_values(by=['sum_of_columns'])
    del df['sum_of_columns']

    models = [openmldefaults.models.CppDefaults(args.c_executable, False),
              openmldefaults.models.CppDefaults(args.c_executable, True)]

    for num_defaults in range(2, 5):
        result_a = models[0].generate_defaults(df, num_defaults)
        result_b = models[1].generate_defaults(df, num_defaults)
        assert result_a['branch_and_bound'] == 0
        assert result_b['branch_and_bound'] == 1
        assert result_a['defaults'] == result_b['defaults']
        assert result_a['objective'] == result_b['objective']
        assert result_a['nodes_visited'] > result_b['nodes_visited']


if __name__ == '__main__':
    run(parse_args())
