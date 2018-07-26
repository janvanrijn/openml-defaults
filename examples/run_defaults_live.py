import arff
import argparse
import collections
import json
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
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='greedy')
    parser.add_argument('--defaults_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    return parser.parse_args()


def json_loads_defaults(defaults):
    for idx, default in enumerate(defaults):
        for param, value in default.items():
            defaults[idx][param] = openmldefaults.config_spaces.reinstantiate_parameter_value(value)
    return defaults


def run(args):
    meta_data = openmldefaults.utils.get_dataset_metadata(args.dataset_path)
    with open(args.dataset_path) as fp:
        column_idx_task_id = []
        for att_name, att_type in arff.load(fp)['attributes']:
            if att_name.startswith(args.dataset_prefix):
                column_idx_task_id.append(int(att_name[len(args.dataset_prefix):]))

    dataset_dir = os.path.basename(args.dataset_path)
    setup_dir = openmldefaults.utils.get_setup_dirname(args.resized_grid_size, args.num_defaults)
    configuration_dir = os.path.join(args.defaults_dir, dataset_dir, 'generated_defaults', args.model_name, setup_dir)
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
        'imputation__categorical_features': categoricals,
        'imputation__fill_empty': -1,
        'hotencoding__handle_unknown': 'ignore'
    }
    estimator.set_params(**categorical_params)
    config_space = getattr(openmldefaults.config_spaces, 'get_%s_default_search_space' % meta_data['classifier'])()
    param_grid = openmldefaults.search.config_space_to_dist(config_space)

    scheduled_strategies = collections.OrderedDict()
    generated_defaults = json_loads_defaults(generated_defaults)
    scheduled_strategies[args.model_name] = openmldefaults.search.DefaultSearchCV(estimator, generated_defaults)
    for i in range(1, 5):
        search_strategy = sklearn.model_selection.RandomizedSearchCV(estimator,
                                                                     param_grid,
                                                                     args.num_defaults * i,
                                                                     random_state=args.random_state)
        scheduled_strategies['random_search_x%d' % i] = search_strategy

    for strategy, search_estimator in scheduled_strategies.items():
        output_dir_strategy = os.path.join(args.defaults_dir, dataset_dir, 'live_random_search', strategy,
                                           setup_dir, str(task_id))
        os.makedirs(output_dir_strategy, exist_ok=True)
        if len(os.listdir(output_dir_strategy)) < 3:
            # we expect at least a trace.arff, predictions.arff and description.arff
            run = openml.runs.run_model_on_task(search_estimator, task)
            run.to_filesystem(output_dir_strategy, store_model=False)
            print(openmldefaults.utils.get_time(), 'Saved to: %s' % output_dir_strategy)
        else:
            print(openmldefaults.utils.get_time(), 'Results already exist: %s in %s' % (strategy, output_dir_strategy))


if __name__ == '__main__':
    run(parse_args())
