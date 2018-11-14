import ConfigSpace
import logging
import math
import numpy as np
import openmlcontrib
import openmldefaults
import pandas as pd
import sklearn
import typing


def _determine_parameter_order(config_space: ConfigSpace.ConfigurationSpace) -> typing.List[str]:
    """
    Returns the hyperparameters in an order that reflects the chain of
    dependencies (unconditional hyperparameter first, in case of a tie lexicographically sorted)
    """
    order = []

    def iterate_params(param_names: typing.List[str]):
        for param_name in param_names:
            param = config_space.get_hyperparameter(param_name)
            children_names = [c.name for c in config_space.get_children_of(param)]
            order.append(param_name)
            iterate_params(children_names)

    unconditionals = sorted(config_space.get_all_unconditional_hyperparameters())
    iterate_params(unconditionals)
    return order


def generate_grid_configurations(config_space: ConfigSpace.ConfigurationSpace,
                                 current_index: int,
                                 max_values_per_parameter: int) \
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

    param_order = _determine_parameter_order(config_space)
    assert set(param_order) == set(config_space.get_hyperparameter_names())

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

        recursive_configs = generate_grid_configurations(config_space, current_index+1,
                                                         max_values_per_parameter)
        result = []

        for recursive_config in recursive_configs:
            parent_conditions = config_space.get_parent_conditions_of(current_hyperparameter.name)
            if len(parent_conditions) == 0:
                result.extend(copy_recursive_configs(recursive_config, current_hyperparameter.name, current_values))
            elif len(parent_conditions) == 1:
                condition_values = parent_conditions[0].value if isinstance(parent_conditions[0].value, list) \
                    else [parent_conditions[0].value]
                if recursive_config[parent_conditions[0].parent.name] in condition_values:
                    result.extend(copy_recursive_configs(recursive_config, current_hyperparameter.name, current_values))
                else:
                    result.extend(copy_recursive_configs(recursive_config, current_hyperparameter.name, [np.nan]))
            else:
                raise NotImplementedError('Hyperparameter %s has multiple parent conditions' %
                                          current_hyperparameter.name)
        return result


def train_surrogate_on_task(task_id: int,
                            config_space: ConfigSpace.ConfigurationSpace,
                            setup_data: pd.DataFrame,
                            evaluation_measure: str,
                            n_estimators,
                            random_seed) \
        -> typing.Tuple[sklearn.pipeline.Pipeline, typing.List]:
    """
    Trains a surrogate on the meta-data from a task.
    """
    nominal_values_min = 10
    # obtain the data
    scaler = sklearn.preprocessing.MinMaxScaler()

    # sort columns!
    setup_data.sort_index(axis=1, inplace=True)
    # reshape because of sklearn api (does not work on vectors)
    y_reshaped = setup_data[evaluation_measure].values.reshape(-1, 1)
    setup_data[evaluation_measure] = scaler.fit_transform(y_reshaped)[:, 0]

    min_val = min(setup_data[evaluation_measure])
    assert math.isclose(min_val, 0.0), 'Not close to 0.0: %f' % min_val
    max_val = max(setup_data[evaluation_measure])
    assert math.isclose(max_val, 1.0), 'Not close to 1.0: %f' % max_val

    # assert that we have ample values for all categorical options
    for hyperparameter in config_space:
        if isinstance(hyperparameter, ConfigSpace.CategoricalHyperparameter):
            for value in hyperparameter.choices:
                num_occurances = len(setup_data.loc[setup_data[hyperparameter.name] == value])
                if num_occurances < nominal_values_min:
                    raise ValueError('Nominal hyperparameter %s value %s does not have enough values. Required '
                                     '%d, got: %d' % (hyperparameter.name, value, nominal_values_min, num_occurances))

    y = setup_data[evaluation_measure].values
    del setup_data[evaluation_measure]
    logging.info('Dimensions of meta-data task %d: %s' % (task_id, str(setup_data.shape)))

    # TODO: HPO
    nominal_pipe = sklearn.pipeline.Pipeline(steps=[
        ('imputer', sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1)),
        ('encoder', sklearn.preprocessing.OneHotEncoder())
    ])

    nominal_indicators = []
    for idx, (name, col) in enumerate(setup_data.dtypes.iteritems()):
        nominal_indicators.append(col == object)
    nominal_indicators = np.array(nominal_indicators, dtype=bool)
    col_trans = sklearn.compose.ColumnTransformer(transformers=[
        ('numeric', sklearn.impute.SimpleImputer(strategy='median'), ~nominal_indicators),
        ('nominal', nominal_pipe, nominal_indicators)
    ])

    surrogate = sklearn.pipeline.Pipeline(steps=[
        ('transformer', col_trans),
        ('classifier', sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators,
                                                              random_state=random_seed))
    ])
    surrogate.fit(setup_data.values, y)
    # the column vector is good to return, as the get_dummies function might behave in-stable
    return surrogate, setup_data.columns.values
