import arff
import ConfigSpace
import logging
import math
import numpy as np
import openmlcontrib
import openmldefaults
import pandas as pd
import sklearn
import sklearn.ensemble
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


def generate_grid_dataset(metadata_frame: pd.DataFrame,
                          configurations: typing.List[typing.Dict[str, typing.Union[str, int, float, bool, None]]],
                          task_ids: typing.List[int],
                          config_space: ConfigSpace.ConfigurationSpace,
                          scoring: str,
                          normalize: bool,
                          random_seed: int,
                          prefix_with_scoring: bool,
                          fill_nans: typing.Optional[float]) -> pd.DataFrame:
    """
    Generates a data frame where each row represents a configuration, each column
    represents an openml task and each cell represents the scoring of that
    configuration on that task.

    Parameters
    ----------
    metadata_frame: pd.Dataframe
        A dataframe with columns for all hyperparameters, a column indicating the
        task and a column indicating the scoring
    configurations: List[Dict[str, mixed]]
        A list of dicts, each dict mapping from hyperparameter name to
        hyperparameter value
    task_ids: List[int]
        The task ids to consider
    config_space: ConfigSpace.ConfigurationSpace
        Determines which hyperparameters are relevant
    scoring: str
        The optimization criterion. Should be a column of meta-data frame
    normalize: bool
        Whether to normalize the resulting frame (column-wise)
    random_seed: int
        A random seed, used for the surrogate model
    prefix_with_scoring: bool
        If set to true, the resulting frame will have column names with the
        scoring prefixed in the name (verbosity, expandability)
    fill_nans: float, optional
        Fills nans in the resulting frame with this value. Nans will only occur
        in hyperparameter values

    Returns
    -------
    pd.DataFrame
        A dataframe with all configurations set to the index, and all tasks as
        columns.
    """
    for configuration in configurations:
        if configuration.keys() != set(config_space.get_hyperparameter_names()):
            raise ValueError('Configuration does not align with ConfigSpace')

    df_orig = pd.DataFrame(configurations)
    logging.info('Meta-dataset dimensions: %s' % str(df_orig.shape))

    # copy of df_orig. Prevent copy function for correct type hints
    df_surrogate = pd.DataFrame(configurations)
    for task_id in task_ids:
        setup_frame = pd.DataFrame(metadata_frame.loc[metadata_frame['task_id'] == task_id])
        if len(setup_frame) == 0:
            raise ValueError('Did not find any configurations for task %d' % task_id)

        del setup_frame['task_id']
        try:
            estimator, columns = openmldefaults.utils.train_surrogate_on_task(task_id,
                                                                              config_space,
                                                                              setup_frame,
                                                                              scoring,
                                                                              64,
                                                                              random_seed)
        except ValueError as e:
            print('Error at task %d: %s' % (task_id, e))
            continue
        if not np.array_equal(df_orig.columns.values, columns):
            # if this goes wrong, it is due to the pd.get_dummies() fn
            raise ValueError('Column sets not equal: %s vs %s' % (df_orig.columns.values, columns))
        surrogate_values = estimator.predict(df_orig.values)
        if normalize:
            scaler = sklearn.preprocessing.MinMaxScaler()
            surrogate_values = scaler.fit_transform(surrogate_values.reshape(-1, 1))[:, 0]
        column_name = 'task_%d' % task_id
        if prefix_with_scoring:
            column_name = '%s_task_%d' % (scoring, task_id)
        df_surrogate[column_name] = surrogate_values

    if df_surrogate.shape[0] != len(configurations):
        raise ValueError('surrogate frame has wrong number of instances. Expected: %d Got %d' % (len(configurations),
                                                                                                 df_surrogate.shape[0]))
    if fill_nans:
        df_surrogate = df_surrogate.fillna(fill_nans)
    df_surrogate = df_surrogate.set_index(config_space.get_hyperparameter_names())
    return df_surrogate


def metadata_file_to_frame(metadata_file: str, config_space: ConfigSpace.ConfigurationSpace, scoring: str) -> pd.DataFrame:
    """
    Loads a meta-data set, as outputted by sklearn bot, and removes redundant
    columns and rows
    """
    with open(metadata_file, 'r') as fp:
        metadata_frame = openmlcontrib.meta.arff_to_dataframe(arff.load(fp), config_space)

    # TODO: modularize. Remove unnecessary columns
    legal_column_names = config_space.get_hyperparameter_names() + [scoring] + ['task_id']
    logging.info('Loaded Dimensions data frame: %s' % str(metadata_frame.shape))
    for column_name in metadata_frame.columns.values:
        if column_name not in legal_column_names:
            logging.info('Removing column: %s' % column_name)
            del metadata_frame[column_name]

    # TODO: modularize. Remove unnecessary rows
    to_drop_indices = []
    for row_idx, row in metadata_frame.iterrows():
        # conditionals can be nan. filter these out with notnull()
        config = {k: v for k, v in row.items() if row.isna()[k] == False}  # JvR: must have == comparison
        del config['task_id']
        del config[scoring]

        try:
            ConfigSpace.Configuration(config_space, config)
        except ValueError as e:
            logging.info('Dropping config, %s' % e)
            to_drop_indices.append(row_idx)

    metadata_frame = metadata_frame.drop(to_drop_indices)
    if metadata_frame.shape[0] == 0:
        raise ValueError()
    logging.info('New Dimensions data frame: %s' % str(metadata_frame.shape))
    return metadata_frame


def single_prediction(df: pd.DataFrame,
                      surrogate: sklearn.pipeline.Pipeline,
                      config: typing.Dict) -> float:
    df = pd.DataFrame(columns=df.columns.values)
    df = df.append(config, ignore_index=True)
    return surrogate.predict(df.values)[0]
