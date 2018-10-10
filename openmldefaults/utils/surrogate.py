import ConfigSpace
import numpy as np
import openmlcontrib
import openmldefaults
import pandas as pd
import sklearn
import typing


def generate_grid_configurations(config_space: ConfigSpace.ConfigurationSpace,
                                 param_order: typing.List[str],
                                 current_index: int,
                                 max_values_per_parameter: int,
                                 ignore_parameters: typing.Optional[typing.Set[str]]) \
        -> typing.List[typing.Dict[str, typing.Union[str, int, float, bool, None]]]:
    """
    Given a config space, this generates a grid of hyperparameter values.
    """
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

        recursive_configs = generate_grid_configurations(config_space, param_order, current_index+1,
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


def train_surrogate_on_task(task_id: int, flow_id: int, num_runs: int,
                            config_space: ConfigSpace.ConfigurationSpace,
                            ignore_parameters: typing.Optional[typing.List],
                            scoring: typing.List,
                            cache_directory: str) \
        -> typing.Tuple[sklearn.pipeline.Pipeline, typing.List]:
    """
    Trains a surrogate on the meta-data from a task
    """
    nominal_values_min = 10
    # obtain the data
    setup_data = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id=task_id,
                                                                       flow_id=flow_id,
                                                                       num_runs=num_runs,
                                                                       configuration_space=config_space,
                                                                       evaluation_measures=scoring,
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
