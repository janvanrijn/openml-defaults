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
    parser.add_argument('--dataset_path', type=str, default=os.path.expanduser('~') +
                        '/data/openml-defaults/surrogate__adaboost__predictive_accuracy__c8.arff')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--num_defaults', type=int, nargs="+", default=[])
    parser.add_argument('--search_iterations', type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='greedy')
    parser.add_argument('--defaults_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    return parser.parse_args()


def get_defaults_file(configuration_dir, task_idx):
    correct_holdout_task_dir = None
    for holdout_tasks_dir in os.listdir(configuration_dir):
        if not os.path.isdir(os.path.join(configuration_dir, holdout_tasks_dir)):
            continue
        try:
            holdout_tasks = json.loads(holdout_tasks_dir)
        except json.decoder.JSONDecodeError:
            continue
        if task_idx in holdout_tasks:
            correct_holdout_task_dir = holdout_tasks_dir
    if correct_holdout_task_dir is None:
        raise ValueError('Could not find holdout task dir for task: %d' % task_idx)
    experiment_dir = os.path.join(configuration_dir, correct_holdout_task_dir)
    defaults_file = os.path.join(experiment_dir, 'generated_defaults.pkl')
    print(openmldefaults.utils.get_time(), 'Defaults file: %s' % defaults_file)
    if not os.path.isfile(defaults_file):
        raise ValueError('Defaults file not generated yet: %s' % defaults_file)
    return defaults_file

def run(args):
    print(openmldefaults.utils.get_time(), args)
    meta_data = openmldefaults.utils.get_dataset_metadata(args.dataset_path)
    with open(args.dataset_path) as fp:
        column_idx_task_id = []
        for att_name, att_type in arff.load(fp)['attributes']:
            surrogate_column_prefix = meta_data['scoring'] + '_task_'
            if att_name.startswith(surrogate_column_prefix):
                column_idx_task_id.append(int(att_name[len(surrogate_column_prefix):]))
    dataset_dir = os.path.basename(args.dataset_path)

    flow = openml.flows.get_flow(meta_data['flow_id'])
    task_id = column_idx_task_id[args.task_idx]
    task = openml.tasks.get_task(task_id)
    estimator = openml.flows.flow_to_sklearn(flow, initialize_with_defaults=True)
    categoricals = task.get_dataset().get_features_by_type('nominal', [task.target_name])
    additional_params = {
        'hotencoding__categorical_features': categoricals,
        'imputation__categorical_features': categoricals,
        'imputation__fill_empty': -1,
        'hotencoding__handle_unknown': 'ignore'
    }
    # adds random state parameters
    for param in estimator.get_params():
        if param.endswith('random_state'):
            additional_params[param] = args.random_state
    if 'steps' in estimator.get_params():
        for step_name, sklearn_obj in estimator.get_params()['steps']:
            if isinstance(sklearn_obj, sklearn.preprocessing.StandardScaler):
                additional_params[step_name + '__with_mean'] = False

    estimator.set_params(**additional_params)

    config_space = getattr(openmldefaults.config_spaces, 'get_%s_default_search_space' % meta_data['classifier'])()
    param_grid = openmldefaults.search.config_space_to_dist(config_space)

    scheduled_strategies = collections.OrderedDict()

    search_scorer = openmldefaults.utils.openml_sklearn_metric_mapping(meta_data['scoring'])

    n_jobs = -1
    for n_defaults in args.num_defaults:
        setup_dir = openmldefaults.utils.get_setup_dirname(args.resized_grid_size,
                                                           n_defaults)
        configuration_dir = os.path.join(args.defaults_dir, dataset_dir,
                                         'generated_defaults', args.model_name,
                                         setup_dir)
        if not os.path.isdir(configuration_dir):
            raise ValueError('Directory does not exists: %s' % configuration_dir)
        defaults_file = get_defaults_file(configuration_dir, args.task_idx)
        with open(defaults_file, 'rb') as fp:
            generated_defaults = pickle.load(fp)['defaults']
        scheduled_strategies['%s__%d' % (args.model_name, n_defaults)] = \
            openmldefaults.search.DefaultSearchCV(estimator, generated_defaults,
                                                  scoring=search_scorer, n_jobs=n_jobs)
    for n_iterations in args.search_iterations:
        search_strategy = sklearn.model_selection.RandomizedSearchCV(estimator,
                                                                     param_grid,
                                                                     n_iterations,
                                                                     scoring=search_scorer,
                                                                     random_state=args.random_state,
                                                                     n_jobs=n_jobs)
        scheduled_strategies['random_search__%d' % n_iterations] = search_strategy

    for strategy, search_estimator in scheduled_strategies.items():
        output_dir_strategy = os.path.join(args.defaults_dir,
                                           dataset_dir,
                                           'live_random_search',
                                           str(args.resized_grid_size),
                                           strategy,
                                           str(task_id))
        os.makedirs(output_dir_strategy, exist_ok=True)
        if len(os.listdir(output_dir_strategy)) < 3:
            # we expect at least a trace.arff, predictions.arff and description.arff
            # XXX: avoid_duplicate_runs since we don't intent to upload to OpenML (and fields are to big for XSD)
            run = openml.runs.run_model_on_task(search_estimator, task, avoid_duplicate_runs=False)
            run.to_filesystem(output_dir_strategy, store_model=False)
            print(openmldefaults.utils.get_time(), 'Saved to: %s' % output_dir_strategy)
        else:
            print(openmldefaults.utils.get_time(), 'Results already exist: %s in %s' % (strategy, output_dir_strategy))


if __name__ == '__main__':
    run(parse_args())
