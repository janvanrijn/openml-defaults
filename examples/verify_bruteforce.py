import argparse
import feather
import openmldefaults
import os
import scipy.misc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/train_svm.feather')
    parser.add_argument('--flip_performances', action='store_true')
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=5)
    return parser.parse_args()


def run(args):
    df = openmldefaults.utils.load_dataset(args.dataset_path, args.params, args.resized_grid_size, args.flip_performances)

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
        c_choose_d = int(scipy.misc.comb(len(df), num_defaults))
        result_a = models[0].generate_defaults(df, num_defaults)
        result_b = models[1].generate_defaults(df, num_defaults)
        assert result_a['branch_and_bound'] == 0
        assert result_b['branch_and_bound'] == 1
        assert result_a['defaults'] == result_b['defaults']
        assert result_a['objective'] == result_b['objective']
        assert result_a['nodes_visited'] > result_b['nodes_visited']
        assert result_a['leafs_visited'] == c_choose_d


if __name__ == '__main__':
    run(parse_args())
