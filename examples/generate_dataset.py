import argparse
import ConfigSpace
import openml
import openmlcontrib
import openmldefaults
import os
import pandas as pd
import sklearn
import sklearn.ensemble


def parse_args():
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/openml_cache',
                        help='directory to store cache')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/openml_cache',
                        help='directory to store output')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--classifier', type=str, default='libsvm_svc', help='openml flow id')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--num_runs', type=int, default=500, help='max runs to obtain from openml')
    parser.add_argument('--normalize', action='store_true', help='normalizes y values per task')
    parser.add_argument('--prevent_model_cache', action='store_true', help='prevents loading old models from cache')
    parser.add_argument('--openml_server', type=str, default=None, help='the openml server location')
    parser.add_argument('--openml_apikey', type=str, default=None, help='the apikey to authenticate to OpenML')
    parser.add_argument('--num_tasks', type=int, default=None, help='limit number of tasks (for testing)')
    return parser.parse_args()


def train_surrogate_on_task(task_id, flow_id, num_runs, config_space, scoring, cache_directory):
    nominal_values_min = 10
    # obtain the data
    setup_data = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id=task_id,
                                                                       flow_id=flow_id,
                                                                       num_runs=num_runs,
                                                                       configuration_space=config_space,
                                                                       parameter_field='parameter_name',
                                                                       evaluation_measure=scoring,
                                                                       cache_directory=cache_directory)
    # assert that we have ample values for all categorical options
    for hyperparameter in config_space:
        if isinstance(hyperparameter, ConfigSpace.CategoricalHyperparameter):
            for value in hyperparameter.values:
                num_occurances = len(setup_data.loc[setup_data[hyperparameter.name] == value])
                if num_occurances < nominal_values_min:
                    raise ValueError('Nominal hyperparameter %s value %s does not have' % (hyperparameter.name, value) +
                                     ' enough values. Required %d, got: %d' % (nominal_values_min, num_occurances))

    y = setup_data['y'].values
    del setup_data['y']
    print(openmldefaults.utils.get_time(), 'Dimensions of meta-data task %d: %s' % (task_id, str(setup_data.shape)))

    # TODO: HPO
    surrogate = sklearn.pipeline.Pipeline(steps=[
        ('imputer', sklearn.preprocessing.Imputer(strategy='median')),
        ('classifier', sklearn.ensemble.RandomForestRegressor(n_estimators=64))
    ])
    surrogate.fit(pd.get_dummies(setup_data).as_matrix(), y)
    return surrogate


def run(args):
    study = openml.study.get_study(args.study_id, 'tasks')

    if args.classifier == 'random_forest':
        flow_id = 6969
        config_space = openmldefaults.config_spaces.get_random_forest_default_search_space()
    elif args.classifier == 'adaboost':
        flow_id = 6970
        config_space = openmldefaults.config_spaces.get_adaboost_default_search_space()
    elif args.classifier == 'libsvm_svc':
        flow_id = 7707
        config_space = openmldefaults.config_spaces.get_libsvm_svc_default_search_space()
    else:
        raise ValueError('classifier type not recognized')

    print(study.tasks)
    for task_id in study.tasks:
        estimator = train_surrogate_on_task(task_id, flow_id, args.num_runs,
                                            config_space, args.scoring, args.cache_directory)


if __name__ == '__main__':
    run(parse_args())
