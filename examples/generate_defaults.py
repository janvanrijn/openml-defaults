import argparse
import openmldefaults
import os
import pickle


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/train_svm.feather')
    parser.add_argument('--flip_performances', action='store_true')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--num_defaults', type=int, default=3)
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    return parser.parse_args()


def run(dataset_path, flip_performances, params, resized_grid_size, num_defaults, models, output_dir, holdout_indices=[]):
    if not isinstance(holdout_indices, list):
        raise ValueError('holdout_indices should be a list')

    print(openmldefaults.utils.get_time(), '=== NUM DEFAULTS: %d ===' % num_defaults)
    df = openmldefaults.utils.load_dataset(dataset_path, params, resized_grid_size, flip_performances)

    # pareto front
    df, dominated = openmldefaults.utils.simple_cull(df, openmldefaults.utils.dominates_min)
    print(openmldefaults.utils.get_time(), 'Dominated Configurations: %d/%d' % (len(dominated), len(df) + len(dominated)))

    # sort configurations by 'good ones'
    df['sum_of_columns'] = df.apply(lambda row: sum(row), axis=1)
    df = df.sort_values(by=['sum_of_columns'])
    del df['sum_of_columns']

    for column in df.columns.values[holdout_indices]:
        del df[column]

    results = {}

    for model in models:
        print(openmldefaults.utils.get_time(), 'Started on model: %s' % model.name)
        solver_dir = model.name
        dataset_dir = os.path.basename(dataset_path)
        setup_dir = openmldefaults.utils.get_setup_dirname(resized_grid_size, num_defaults)
        experiment_dir = os.path.join(output_dir, dataset_dir, solver_dir, setup_dir)
        if len(holdout_indices) > 0:
            experiment_dir = os.path.join(experiment_dir, str(sorted(holdout_indices)))
        experiment_file = os.path.join(experiment_dir, 'results.pkl')
        if os.path.isfile(experiment_file):
            with open(experiment_file, 'rb') as fp:
                print(openmldefaults.utils.get_time(), 'Already exists')
                results[model.name] = pickle.load(fp)
                continue

        results_dict = model.generate_defaults(df, num_defaults)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(experiment_file, 'wb') as fp:
            pickle.dump(results_dict, fp)

        sum_of_scores = sum(openmldefaults.utils.selected_set(df, results_dict['defaults']))
        diff = abs(sum_of_scores - results_dict['objective'])
        assert diff < 0.0001, 'Sum of scores does not equal score of solution: %f vs %f' % (sum_of_scores,
                                                                                            results_dict['score'])

        results[model.name] = results_dict
        print(results_dict)

    if 'cpp_bruteforce' in results and 'mip' in results:
        diff = abs(results['cpp_bruteforce']['objective'] - results['mip']['objective'])
        assert diff < 0.0001
        assert set(results['cpp_bruteforce']['defaults']) == set(results['mip']['defaults'])


if __name__ == '__main__':
    args = parse_args()

    models = [
        openmldefaults.models.CppDefaults(args.c_executable, True),
        openmldefaults.models.GreedyDefaults(),
        # openmldefaults.models.MipDefaults('GLPK_CMD')
    ]

    run(args.dataset_path, args.flip_performances, args.params, args.resized_grid_size, args.num_defaults,
        models, args.output_dir)
