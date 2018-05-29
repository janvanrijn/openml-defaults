import argparse
import feather
import numpy as np
import openmldefaults
import os
import pickle


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/train_svm.feather')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--restricted_num_tasks', type=int, default=None)
    parser.add_argument('--num_defaults', type=int, default=2)
    return parser.parse_args()


def print_columns(df, params):
    for param in params:
        unique = np.array(df[param].unique())
        print(openmldefaults.utils.get_time(), '%s unique values: %s (%d)' % (param, unique, len(unique)))


def run(args):
    df = feather.read_dataframe(args.dataset_path)
    print(openmldefaults.utils.get_time(), 'Original data frame dimensions:', df.shape)

    for param in args.params:
        if param not in df.columns.values:
            raise ValueError('Param column not found. Columns %s, illegal: %s' % (df.columns.values, param))

    if not os.path.isfile(args.c_executable):
        raise ValueError('Please compile C program first')

    if args.resized_grid_size is not None:
        df = openmldefaults.utils.reshape_configs(df, args.params, args.resized_grid_size)

    print_columns(df, args.params)

    # always set the index
    df = df.set_index(args.params)

    if args.restricted_num_tasks is not None:
        # subsample num tasks
        df = df.iloc[:, 0:args.restricted_num_tasks]
    print(openmldefaults.utils.get_time(), 'Reshaped data frame dimensions:', df.shape)

    # pareto front
    df, dominated = openmldefaults.utils.simple_cull(df, openmldefaults.utils.dominates_min)
    print(openmldefaults.utils.get_time(), 'Dominated Configurations: %d/%d' % (len(dominated), len(df) + len(dominated)))

    # sort configurations by 'good ones'
    df['sum_of_columns'] = df.apply(lambda row: sum(row), axis=1)
    df = df.sort_values(by=['sum_of_columns'])
    del df['sum_of_columns']

    models = [openmldefaults.models.CppDefaults(args.c_executable)]

    results = {}

    for model in models:
        solver_dir = model.name
        dataset_dir = os.path.basename(args.dataset_path)
        setup_dir = openmldefaults.utils.get_setup_dirname(args)
        experiment_dir = os.path.join(args.output_dir, dataset_dir, solver_dir, setup_dir)
        experiment_file = os.path.join(experiment_dir, 'results.pkl')
        if os.path.isfile(experiment_file):
            with open(experiment_file, 'rb') as fp:
                results[model.name] = pickle.load(fp)
                continue

        results_dict = model.generate_defaults(df, args.num_defaults)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(experiment_file, 'wb') as fp:
            pickle.dump(results_dict, fp)

        sum_of_scores = sum(openmldefaults.utils.selected_set(df, results_dict['selected_defaults']))
        diff = abs(sum_of_scores - results_dict['score'])
        assert diff < 0.0001, 'Sum of scores does not equal score of solution: %f vs %f' % (sum_of_scores, results_dict['score'])

        results[model.name] = results_dict


if __name__ == '__main__':
    run(parse_args())
