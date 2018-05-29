import argparse
import feather
import json
import openmldefaults
import os
import pickle
import subprocess
import time


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/mlr.classif.rpart.feather')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=5)
    parser.add_argument('--restricted_num_tasks', type=int, default=None)
    parser.add_argument('--num_defaults', type=int, default=2)
    return parser.parse_args()


def run(args):
    df = feather.read_dataframe(args.dataset_path)
    print(openmldefaults.utils.get_time(), 'Original data frame dimensions:', df.shape)

    if not os.path.isfile(args.c_executable):
        raise ValueError('Please compile C program first')

    if args.resized_grid_size is not None:
        df = openmldefaults.utils.reshape_configs(df, args.params, args.resized_grid_size)

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

    print(openmldefaults.utils.get_time(), 'Started c program')
    num_configs, num_tasks = df.shape
    process = subprocess.Popen([args.c_executable], stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    process_input = [str(args.num_defaults), str(num_configs), str(num_tasks)]
    for iConf in range(num_configs):
        for iTask in range(num_tasks):
            process_input.append(str(df.iloc[iConf, iTask]))

    start_time = time.time()
    out, err = process.communicate("\n".join(process_input))
    runtime = time.time() - start_time
    if process.returncode != 0:
        raise ValueError('Process terminated with non-zero exit code. ')
    print(openmldefaults.utils.get_time(), 'Runtime: %d seconds' % runtime)

    for idx, line in enumerate(out.split("\n")):
        try:
            solution = json.loads(line)
        except json.decoder.JSONDecodeError:
            pass

    print(openmldefaults.utils.get_time(), solution)
    selected_defaults = [df.index[idx] for idx in solution['solution']]

    results_dict = {
        'objective': solution['score'],
        'run_time': runtime,
        'defaults': selected_defaults
    }
    solver_dir = 'cpp_bruteforce'
    dataset_dir = os.path.basename(args.dataset_path)
    experiment_dir = openmldefaults.utils.get_experiment_dir(args)
    os.makedirs(os.path.join(args.output_dir, dataset_dir, solver_dir, experiment_dir), exist_ok=True)
    with open(os.path.join(args.output_dir, dataset_dir, solver_dir, experiment_dir, 'results.pkl'), 'wb') as fp:
        pickle.dump(results_dict, fp)

    sum_of_scores = sum(openmldefaults.utils.selected_set(df, selected_defaults))
    diff = abs(sum_of_scores - solution['score'])
    assert diff < 0.00001, 'Sum of scores does not equal score of solution: %f vs %f' % (sum_of_scores, solution['score'])


if __name__ == '__main__':
    run(parse_args())
