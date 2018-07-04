import arff
import argparse
import json
import openml
import openmldefaults
import os
import pickle
import sklearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/surrogate_adaboost_c16.arff')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--dataset_prefix', type=str, default='predictive_accuracy_task_')
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--num_defaults', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='cpp_bruteforce')
    parser.add_argument('--defaults_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    return parser.parse_args()


def run(args):
    with open(args.dataset_path) as fp:
        first_line = fp.readline()
        if first_line[0] != '%':
            raise ValueError('arff data file should start with comment for meta-data')
        meta_data = json.loads(first_line[1:])
        column_idx_task_id = []
        for att_name, att_type in arff.load(fp)['attributes']:
            if att_name.startswith(args.dataset_prefix):
                column_idx_task_id.append(int(att_name[len(args.dataset_prefix):]))

    solver_dir = args.model_name
    dataset_dir = os.path.basename(args.dataset_path)
    setup_dir = openmldefaults.utils.get_setup_dirname(args.resized_grid_size, args.num_defaults)
    configuration_dir = os.path.join(args.defaults_dir, dataset_dir, solver_dir, setup_dir)
    if not os.path.isdir(configuration_dir):
        raise ValueError('Directory does not exists: %s' % configuration_dir)
    correct_holdout_task_dir = None
    for holdout_tasks_dir in os.listdir(configuration_dir):
        if not os.path.isdir(os.path.join(configuration_dir, holdout_tasks_dir)):
            continue
        try:
            holdout_tasks = json.loads(holdout_tasks_dir)
        except json.decoder.JSONDecodeError:
            continue
        if args.task_idx in holdout_tasks:
            correct_holdout_task_dir = holdout_tasks_dir
    if correct_holdout_task_dir is None:
        raise ValueError('Could not find holdout task dir for task: %d' % args.task_idx)
    experiment_dir = os.path.join(configuration_dir, correct_holdout_task_dir)
    experiment_file = os.path.join(experiment_dir, 'generated_defaults.pkl')
    if not os.path.isfile(experiment_file):
        raise ValueError('Defaults file not generated yet: %s' % experiment_file)
    print(openmldefaults.utils.get_time(), 'Experiments file: %s' % experiment_file)

    with open(experiment_file, 'rb') as fp:
        generated_defaults = pickle.load(fp)['defaults']

    flow = openml.flows.get_flow(meta_data['flow_id'])
    task_id = column_idx_task_id[args.task_idx]
    task = openml.tasks.get_task()
    estimator = openml.flows.flow_to_sklearn(flow)
    config_space = getattr(openmldefaults.config_spaces, 'get_%s_default_search_space' % meta_data['classifier'])()
    param_grid = openmldefaults.search.config_space_to_dist(config_space)

    output_dir = os.path.join(args.defaults_dir, dataset_dir, 'live_random_search', solver_dir, setup_dir, str(task_id))
    os.makedirs(output_dir, exist_ok=True)

    search_x1 = sklearn.model_selection.RandomSearchCV(estimator, param_grid, args.n_defaults)

    pass


if __name__ == '__main__':
    run(parse_args())
