import arff
import argparse
import ConfigSpace
import json
import numpy as np
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
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/generated_data/')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--classifier', type=str, default='libsvm_svc', help='scikit-learn flow name')
    parser.add_argument('--config_space', type=str, default='default', help='config space type')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--num_runs', type=int, default=500, help='max runs to obtain from openml')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    return parser.parse_args()


def determine_parameter_order(config_space, ignore_params):
    order = []

    def iterate_params(params):
        for param in params:
            children = config_space.get_children_of(param)
            if ignore_params is not None and param in ignore_params and len(children) > 0:
                raise ValueError('Param %s is ignored but has %d child params' % (param.name, len(children)))
            if ignore_params is None or param.name not in ignore_params:
                order.append(param.name)
                iterate_params(children)
        pass

    unconditionals = [config_space.get_hyperparameter(name) for name in config_space.get_all_unconditional_hyperparameters()]
    iterate_params(unconditionals)
    return order


def generate_configurations(config_space, param_order, current_index, max_values_per_parameter, ignore_parameters):
    def copy_recursive_configs(recursive_config_, current_name_, current_values_):
        result_ = []
        for i_ in range(len(current_values_)):
            current_value = current_values_[i_]
            cp = dict(recursive_config_)
            cp[current_name_] = current_value
            result_.append(cp)
        return result_

    if current_index >= len(param_order):
        return [{}]
    else:
        current_hyperparameter = config_space.get_hyperparameter(param_order[len(param_order) - 1 - current_index])
        if isinstance(current_hyperparameter, ConfigSpace.CategoricalHyperparameter):
            current_values = current_hyperparameter.choices
        elif isinstance(current_hyperparameter, ConfigSpace.UniformFloatHyperparameter):
            if current_hyperparameter.log:
                current_values = np.logspace(np.log(current_hyperparameter.lower),
                                             np.log(current_hyperparameter.upper),
                                             num=max_values_per_parameter,
                                             base=np.e)
            else:
                current_values = np.linspace(current_hyperparameter.lower, current_hyperparameter.upper,
                                             num=max_values_per_parameter)
        elif isinstance(current_hyperparameter, ConfigSpace.UniformIntegerHyperparameter):
            possible_values = current_hyperparameter.upper - current_hyperparameter.lower + 1
            if current_hyperparameter.log:
                current_values = np.logspace(np.log(current_hyperparameter.lower),
                                             np.log(current_hyperparameter.upper),
                                             num=min(max_values_per_parameter, possible_values),
                                             base=np.e, dtype=int)
            else:
                current_values = np.linspace(current_hyperparameter.lower, current_hyperparameter.upper,
                                             num=min(max_values_per_parameter, possible_values), dtype=int)
            current_values = [np.round(val) for val in list(current_values)]
        elif isinstance(current_hyperparameter, ConfigSpace.UnParametrizedHyperparameter) or \
                isinstance(current_hyperparameter, ConfigSpace.Constant):
            current_values = [current_hyperparameter.value]
        else:
            raise ValueError('Could not determine hyperparameter type: %s' % current_hyperparameter.name)

        recursive_configs = generate_configurations(config_space, param_order, current_index+1,
                                                    max_values_per_parameter,
                                                    ignore_parameters)
        if ignore_parameters is not None and current_hyperparameter.name in ignore_parameters:
            return recursive_configs
        current_values = [openmldefaults.config_spaces.post_process(value) for value in current_values]
        result = []

        for recursive_config in recursive_configs:
            parent_conditions = config_space.get_parent_conditions_of(current_hyperparameter.name)
            if len(parent_conditions) == 0:
                result.extend(copy_recursive_configs(recursive_config, current_hyperparameter.name, current_values))
            elif len(parent_conditions) == 1:
                condition_values = parent_conditions[0].value if isinstance(parent_conditions[0].value, list) \
                    else [parent_conditions[0].value]
                condition_values_processed = [openmldefaults.config_spaces.post_process(value)
                                              for value in condition_values]
                if recursive_config[parent_conditions[0].parent.name] in condition_values_processed:
                    result.extend(copy_recursive_configs(recursive_config, current_hyperparameter.name, current_values))
                else:
                    result.extend(copy_recursive_configs(recursive_config, current_hyperparameter.name, [np.nan]))
            else:
                raise NotImplementedError('Hyperparameter %s has multiple parent conditions' %
                                          current_hyperparameter.name)
        return result


def train_surrogate_on_task(task_id, flow_id, num_runs, config_space, ignore_parameters, scoring, cache_directory):
    nominal_values_min = 10
    # obtain the data
    setup_data = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id=task_id,
                                                                       flow_id=flow_id,
                                                                       num_runs=num_runs,
                                                                       configuration_space=config_space,
                                                                       parameter_field='parameter_name',
                                                                       evaluation_measure=scoring,
                                                                       cache_directory=cache_directory)
    if ignore_parameters is not None:
        for ignore_parameter in ignore_parameters:
            del setup_data[ignore_parameter]

    # assert that we have ample values for all categorical options
    for hyperparameter in config_space:
        if ignore_parameters is not None and hyperparameter in ignore_parameters:
            continue
        if isinstance(hyperparameter, ConfigSpace.CategoricalHyperparameter):
            for value in hyperparameter.choices:
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
        ('classifier', sklearn.ensemble.RandomForestRegressor(n_estimators=64,
                                                              random_state=42))
    ])
    surrogate.fit(pd.get_dummies(setup_data).as_matrix(), y)
    return surrogate, setup_data.columns.values


def run(args):
    study = openml.study.get_study(args.study_id, 'tasks')

    if args.classifier == 'random_forest':
        flow_id = 6969
        ignore_parameters = None
    elif args.classifier == 'adaboost':
        flow_id = 6970
        ignore_parameters = None
    elif args.classifier == 'libsvm_svc':
        flow_id = 7707
        if args.config_space == 'default':
            ignore_parameters = ['max_iter', 'tol', 'strategy', 'shrinking']
        else:
            ignore_parameters = None
    else:
        raise ValueError('classifier type not recognized')
    config_space_fn = getattr(openmldefaults.config_spaces,
                              'get_%s_%s_search_space' % (args.classifier,
                                                          args.config_space))
    config_space = config_space_fn()
    meta_data = {'flow_id': flow_id,
                 'classifier': args.classifier,
                 'config_space': args.config_space,
                 'scoring': args.scoring}

    num_params = len(config_space.get_hyperparameter_names())
    param_order = determine_parameter_order(config_space, ignore_parameters)
    configurations = generate_configurations(config_space, param_order, 0, args.resized_grid_size, ignore_parameters)

    df_orig = pd.DataFrame(configurations)
    print(openmldefaults.utils.get_time(), 'Meta-dataset dimensions: %s' % str(df_orig.shape))

    df_surrogate = df_orig.copy()
    for task_id in study.tasks:
        try:
            estimator, columns = train_surrogate_on_task(task_id, flow_id, args.num_runs, config_space,
                                                         ignore_parameters, args.scoring, args.cache_directory)
        except ValueError as e:
            print('Error at task %d: %s' % (task_id, e))
            continue
        if not np.array_equal(df_orig.columns.values, columns):
            raise ValueError('Column sets not equal: %s vs %s' % (df_orig.columns.values, columns))
        surrogate_values = estimator.predict(pd.get_dummies(df_orig).as_matrix())
        df_surrogate['%s_task_%d' % (args.scoring, task_id)] = surrogate_values

    if df_surrogate.shape[0] != len(configurations):
        raise ValueError('surrogate frame has wrong number of instances. Expected: %d Got %d' % (len(configurations),
                                                                                                 df_surrogate.shape[0]))

    if df_surrogate.shape[1] > num_params + len(study.tasks):
        raise ValueError('surrogate frame has too much of columns. Max: %d Got %d' % (num_params + len(study.tasks),
                                                                                      df_surrogate.shape[1]))
    if df_surrogate.shape[1] < num_params + len(study.tasks) / 2:
        raise ValueError('surrogate frame has too few columns. Min: %d Got %d' % (num_params + len(study.tasks) / 2,
                                                                                  df_surrogate.shape[1]))
    os.makedirs(args.output_directory, exist_ok=True)
    arff_object = openmlcontrib.meta.dataframe_to_arff(df_surrogate,
                                                       'surrogate_%s' % args.classifier,
                                                       json.dumps(meta_data))

    with open(os.path.join(args.output_directory, 'surrogate__%s__%s__%s__c%d.arff' % (args.classifier,
                                                                                       args.config_space,
                                                                                       args.scoring,
                                                                                       args.resized_grid_size)), 'w') as fp:
        arff.dump(arff_object, fp)


if __name__ == '__main__':
    run(parse_args())
