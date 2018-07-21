import arff
import argparse
import json
import numpy as np
import openml
import openmldefaults
import os
import pickle
import sklearn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/surrogate_adaboost_c8.arff')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--dataset_prefix', type=str, default='predictive_accuracy_task_')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--num_defaults', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='greedy')
    parser.add_argument('--defaults_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    return parser.parse_args()


def json_loads_defaults(defaults):
    for idx, default in enumerate(defaults):
        for param, value in default.items():
            defaults[idx][param] = openmldefaults.config_spaces.reinstantiate_parameter_value(value)
    return defaults


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
    configuration_dir = os.path.join(args.defaults_dir, dataset_dir, 'generated_defaults', solver_dir, setup_dir)
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
    defaults_file = os.path.join(experiment_dir, 'generated_defaults.pkl')
    if not os.path.isfile(defaults_file):
        raise ValueError('Defaults file not generated yet: %s' % defaults_file)
    print(openmldefaults.utils.get_time(), 'Experiments file: %s' % defaults_file)

    with open(defaults_file, 'rb') as fp:
        generated_defaults = pickle.load(fp)['defaults']

    flow = openml.flows.get_flow(meta_data['flow_id'])
    task_id = column_idx_task_id[args.task_idx]
    task = openml.tasks.get_task(task_id)
    estimator = openml.flows.flow_to_sklearn(flow, initialize_with_defaults=True)
    categoricals = task.get_dataset().get_features_by_type('nominal', [task.target_name])
    categorical_params = {
        'hotencoding__categorical_features': categoricals,
        'imputation__categorical_features': categoricals
    }
    estimator.set_params(**categorical_params)
    config_space = getattr(openmldefaults.config_spaces, 'get_%s_default_search_space' % meta_data['classifier'])()
    param_grid = openmldefaults.search.config_space_to_dist(config_space)

    output_dir_defaults = os.path.join(args.defaults_dir, dataset_dir, 'live_random_search', solver_dir, setup_dir, str(task_id))
    os.makedirs(output_dir_defaults, exist_ok=True)

    if not os.path.isdir(output_dir_defaults):
        generated_defaults = json_loads_defaults(generated_defaults)
        print(openmldefaults.utils.get_time(), 'Starting default search, defaults: %s' % generated_defaults)
        search = openmldefaults.search.DefaultSearchCV(estimator, generated_defaults)
        run = openml.runs.run_model_on_task(search, task)
        run.to_filesystem(output_dir_defaults)

    for i in range(1, 5):
        print(openmldefaults.utils.get_time(), 'Starting random search x%d, param grid: %s' % (i, param_grid))

        output_dir_rs = os.path.join(args.defaults_dir, dataset_dir, 'live_random_search', 'random_search_x%d' % i, setup_dir, str(task_id))
        if os.path.isdir(output_dir_rs) and len(os.listdir(output_dir_rs)) > 0:
            print(openmldefaults.utils.get_time(), 'output dir already has content')
            continue
        search = sklearn.model_selection.RandomizedSearchCV(estimator, param_grid, args.num_defaults * i)
        run = openml.runs.run_model_on_task(search, task)
        run.to_filesystem(output_dir_rs)
        print(openmldefaults.utils.get_time(), 'Saved to: %s' % output_dir_rs)


if __name__ == '__main__':
    run(parse_args())
