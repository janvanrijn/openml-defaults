import arff
import ConfigSpace
import csv
import logging
import math
import numpy as np
import openmlcontrib
import openmldefaults
import os
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


def train_surrogate_on_task(task_id: typing.Union[int, str],
                            hyperparameter_names: typing.List[str],
                            setup_data: pd.DataFrame,
                            evaluation_measure: str,
                            normalize: bool,
                            n_estimators,
                            random_seed) \
        -> typing.Tuple[sklearn.pipeline.Pipeline, typing.List]:
    """
    Trains a surrogate on the meta-data from a task.
    """
    # delete unnecessary columns
    legal_columns = set(hyperparameter_names + [evaluation_measure])
    for column in setup_data.columns.values:
        if column not in legal_columns:
            del setup_data[column]
    if set(setup_data.columns.values) != legal_columns:
        missing = legal_columns - set(setup_data.columns.values)
        over = set(setup_data.columns.values) - legal_columns
        raise ValueError('Columns for surrogate do not align with expectations. Missing: %s, over: %s' % (missing, over))

    nominal_values_min = 10
    # obtain the data
    scaler = sklearn.preprocessing.MinMaxScaler()

    # sort columns!
    setup_data.sort_index(axis=1, inplace=True)
    # reshape because of sklearn api (does not work on vectors)
    if normalize:
        y_reshaped = setup_data[evaluation_measure].values.reshape(-1, 1)
        setup_data[evaluation_measure] = scaler.fit_transform(y_reshaped)[:, 0]

        min_val = min(setup_data[evaluation_measure])
        assert math.isclose(min_val, 0.0), 'Not close to 0.0: %f' % min_val
        max_val = max(setup_data[evaluation_measure])
        assert math.isclose(max_val, 1.0), 'Not close to 1.0: %f' % max_val

    # assert that we have ample values for all categorical options
    for hyperparameter in hyperparameter_names:
        if isinstance(hyperparameter, ConfigSpace.CategoricalHyperparameter):
            for value in hyperparameter.choices:
                num_occurances = len(setup_data.loc[setup_data[hyperparameter.name] == value])
                if num_occurances < nominal_values_min:
                    raise ValueError('Nominal hyperparameter %s value %s does not have enough values. Required '
                                     '%d, got: %d' % (hyperparameter.name, value, nominal_values_min, num_occurances))

    y = setup_data[evaluation_measure].values
    del setup_data[evaluation_measure]
    logging.info('Dimensions of meta-data task %s: %s. Target %s [%f-%f]' % (task_id,
                                                                             str(setup_data.shape),
                                                                             evaluation_measure,
                                                                             min(y), max(y)))

    # TODO: HPO
    nominal_pipe = sklearn.pipeline.Pipeline(steps=[
        ('imputer', sklearn.impute.SimpleImputer(strategy='constant', fill_value='-1')),
        ('encoder', sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore'))
    ])

    nominal_indicators = []
    for idx, (name, col) in enumerate(setup_data.dtypes.iteritems()):
        nominal_indicators.append(col == object)
    nominal_indicators = np.array(nominal_indicators, dtype=bool)
    col_trans = sklearn.compose.ColumnTransformer(transformers=[
        ('numeric', sklearn.impute.SimpleImputer(strategy='constant', fill_value=-1), ~nominal_indicators),
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


def generate_dataset_using_metadata(
        metadata_frame: pd.DataFrame,
        task_ids: typing.List[typing.Union[int, str]],
        hyperparameter_names: typing.List[str],
        measure: str,
        task_id_column: str,
        scaler_type: typing.Optional[str],
        column_prefix: typing.Optional[str]) -> pd.DataFrame:
    """
    Turns a dataframe with columns representing the hyperparameter values, task
    id and evaluation measures, into a data frame where each row represents a
    configuration, each column represents an openml task and each cell represents
    the scoring of that configuration on that task.
    """
    pivoted = pd.pivot_table(data=metadata_frame,
                             index=hyperparameter_names,
                             columns=task_id_column,
                             values=measure)
    if pivoted.isnull().values.any():
        raise ValueError('meta-data not complete: pivoted frame contains nans')
    if scaler_type is not None:
        raise NotImplementedError()
    pivoted = pivoted[task_ids]
    pivoted = pivoted.add_prefix('task_')
    if column_prefix is not None:
        pivoted = pivoted.add_prefix(column_prefix)
    return pivoted


def generate_dataset_using_surrogates(
        surrogates: typing.Dict[int, sklearn.pipeline.Pipeline],
        surrogate_columns: np.array,
        task_ids: typing.List[typing.Union[int, str]],
        meta_features: pd.DataFrame,
        config_sampler: openmldefaults.symbolic.ConfigurationSampler,
        n_configurations: int,
        scaler_type: typing.Optional[str],
        column_prefix: typing.Optional[str],
        fill_nans: typing.Optional[float]) -> pd.DataFrame:
    """
    Generates a data frame where each row represents a configuration, each column
    represents an openml task and each cell represents the scoring of that
    configuration on that task.

    Parameters
    ----------
    surrogates: dict[str, RegressorMixin]
        A dictionary mapping from task id to a surrogate
    surrogate_columns: np.array
        An array of the columns expected by the surrogates
    task_ids: list
        A list of tasks to include in the resulting frame (note that each
        task must be a key in the surrogates dict, or an error will be thrown)
    meta_features: pd.DataFrame
        A dataframe with as index a task id and as columns the relevant
        meta-features
    config_sampler: ConfigurationSampler
        Used to sample configurations
    n_configurations: int
        The number of configurations to sample (should be higher than the number
        of defaults)
    scaler_type: str (optional)
        Which scalar to use for the resulting data frame (see function:
        openmldefaults.utils.get_scaler)
    column_prefix: str
        If set, the resulting frame will have column names prefixed with the
        with this value (verbosity, expandability)
    fill_nans: float, optional
        Fills nans in the resulting frame with this value. Nans will only occur
        in hyperparameter values

    Returns
    -------
    pd.DataFrame
        A dataframe with all configurations set to the index, and all tasks as
        columns.
    """
    def complete_dataframe(df_: pd.DataFrame, surrogate_column_order: np.array) -> pd.DataFrame:
        """
        Adds several columns to a dataframe, in case they were missing before.
        Also removes other columns and sets the columns in the correct order.
        """
        for column_ in surrogate_column_order:
            if column_ not in df_:
                impute_value = np.nan
                logging.info('Adding column to meta-dataset: %s. Imputing with %s' % (column_, impute_value))
                df_[column_] = impute_value
        # reorders columns to order required for surrogate (if columns were added)
        df_ = df_[surrogate_column_order]
        return df_

    configurations = config_sampler.sample_configurations(n_configurations)
    for configuration in configurations:
        illegal = set(configuration.get_dictionary(None).keys()) - set(config_sampler.get_hyperparameter_names())
        if len(illegal) > 0:
            raise ValueError('Configuration contains illegal hyper-parameters: %s' % illegal)

    df_orig = pd.DataFrame([{'configuration': c} for c in configurations])

    for task_id in task_ids:
        # TODO:
        # 1) get hyperparameter values
        # 2) get them in right order (as given by surrogate_columns)
        df_task = pd.DataFrame([c.get_dictionary(meta_features.loc[task_id]) for c in configurations])
        df_task = complete_dataframe(df_task, surrogate_columns)
        surrogate_values = surrogates[task_id].predict(df_task.values)
        if scaler_type is not None:
            logging.info('scaling predictions for task %s using %s' % (task_id, scaler_type))
            scaler = openmldefaults.utils.get_scaler(scaler_type)
            surrogate_values = scaler.fit_transform(surrogate_values.reshape(-1, 1))[:, 0]
        column_name = 'task_%s' % task_id
        if column_prefix:
            column_name = '%s_%s' % (column_prefix, column_name)
        df_orig[column_name] = surrogate_values
    if len(df_orig) != len(configurations):
        raise ValueError('surrogate frame has wrong number of instances.'
                         'Expected: %d Got %d' % (len(configurations), len(df_orig)))
    if fill_nans:
        df_orig = df_orig.fillna(fill_nans)
    df_orig = df_orig.set_index('configuration')
    return df_orig


def generate_surrogates_using_metadata(
        metadata_frame: pd.DataFrame,
        hyperparameter_names: typing.List[str],
        scoring: str,
        minimum_evals: int,
        n_estimators: int,
        random_seed: int,
        task_id_column: str) -> typing.Tuple[typing.Dict[int, sklearn.pipeline.Pipeline], np.array]:
    """
    Generates a data frame where each row represents a configuration, each
    column represents an openml task and each cell represents the scoring of
    that configuration on that task.

    Parameters
    ----------
    metadata_frame: pd.Dataframe
        A dataframe with columns for all hyperparameters, a column indicating the
        task and a column indicating the scoring
    hyperparameter_names: list[str]
        Determines which hyperparameters are relevant
    scoring: str
        The optimization criterion. Should be a column of meta-data frame
    minimum_evals: int
        Minimum number of evaluations per task (or error will be thrown)
    n_estimators: int
        The number of trees in the random forest surrogates
    random_seed: int
        A random seed, used for the surrogate model
    task_id_column: str
        The column name in metadata_frame that represents the dataset name or task id

    Returns
    -------
    surrogates: dict[int, Pipeline]
        A dictionary mapping from task id to the surrogate on that task
    columns_original: np.array
        The columns that are expected to train a surrogate on (same for all
        surrogates)
    """
    surrogates = dict()
    task_ids = metadata_frame[task_id_column].unique()
    columns_original = None
    if len(task_ids) == 0:
        raise ValueError()

    for task_id in task_ids:
        setup_frame = pd.DataFrame(metadata_frame.loc[metadata_frame[task_id_column] == task_id])
        if len(setup_frame) < minimum_evals:
            raise ValueError('Not enough evaluations in meta-frame for task %s: %d' % (task_id, minimum_evals))

        del setup_frame[task_id_column]
        estimator, columns = openmldefaults.utils.train_surrogate_on_task(task_id,
                                                                          hyperparameter_names,
                                                                          setup_frame,
                                                                          scoring,
                                                                          False,  # we will normalize predictions
                                                                          n_estimators,
                                                                          random_seed)
        if columns_original is None:
            columns_original = columns
        if not np.array_equal(columns_original, columns):
            # if this goes wrong, it is due to the pd.get_dummies() fn
            missing = set(columns_original) - set(columns)
            over = set(columns) - set(columns_original)
            raise ValueError('Column sets not equal. Missing: %s; over: %s' % (missing, over))
        surrogates[task_id] = estimator
    return surrogates, columns_original


def metadata_files_to_frame(metadata_files: typing.List[str],
                            search_space_identifier: str,
                            scoring: typing.List[str],
                            task_id_column: str,
                            skip_row_check: bool) -> pd.DataFrame:
    """
    Loads a meta-data set, as outputted by sklearn bot, and removes redundant
    columns and rows.

    Parameters:
    -----------
    metadata_files: List[str]
        List of files (str) to load. The base filename (without directory and extension) should match with the config
        space to load.

    search_space_identifier: str
        The search space identifier for load config spaces

    scoring: str
        column name that contains the evaluation measure

    task_id_column: str
        column name that contains the task id

    check_rows: bool
        If set to True, all rows will be checked whether they fall within the ConfigurationSpace. If set to False, this
        check will be skipped (saves time, but less secure)

    Returns:
    --------
    metadata_frame_total: pd.DataFrame
        Dataframe with columns being the task_id, hyperparameters and performance measure.
    """
    metadata_frame_total = None
    for metadata_file in metadata_files:
        # sanity checks
        metadata_atts = openmldefaults.utils.get_dataset_metadata(metadata_file)
        for measure in scoring:
            if measure not in metadata_atts['col_measures']:
                raise ValueError('Could not find mandatory measure %s in dataset: %s' % (measure, metadata_file))
        # open the file and extract the correct columns
        with open(metadata_file, 'r') as fp:
            classifier_name = os.path.splitext(os.path.basename(metadata_file))[0]
            config_space = openmldefaults.config_spaces.get_config_space(classifier_name, 0, search_space_identifier)
            metadata_frame_classif = openmlcontrib.meta.arff_to_dataframe(arff.load(fp), config_space)
            metadata_frame_classif['classifier'] = classifier_name
            logging.info('Loaded %s meta-data data frame. Dimensions: %s' % (classifier_name,
                                                                             str(metadata_frame_classif.shape)))
            logging.info('Columns: %s' % metadata_frame_classif.columns.values)
            for measure in scoring:
                logging.info('meta-data ranges for measure %s: [%f-%f]' % (measure,
                                                                           min(metadata_frame_classif[measure]),
                                                                           max(metadata_frame_classif[measure])))

            # TODO: modularize. Remove unnecessary columns
            legal_column_names = config_space.get_hyperparameter_names() + scoring + ['classifier', task_id_column]
            for column_name in metadata_frame_classif.columns.values:
                if column_name not in legal_column_names:
                    logging.info('Removing column: %s' % column_name)
                    del metadata_frame_classif[column_name]

            # TODO: modularize. Remove unnecessary rows
            to_drop_indices = []
            if not skip_row_check:
                for row_idx, row in metadata_frame_classif.iterrows():
                    # conditionals can be nan. filter these out with notnull()
                    config = {k: v for k, v in row.items() if row.isna()[k] == False}  # JvR: must have == comparison
                    del config[task_id_column]
                    del config['classifier']
                    for measure in scoring:
                        del config[measure]
                    try:
                        ConfigSpace.Configuration(config_space, config)
                    except ValueError as e:
                        logging.info('Dropping config, %s' % e)
                        to_drop_indices.append(row_idx)

            metadata_frame_classif = metadata_frame_classif.drop(to_drop_indices)
            if metadata_frame_classif.shape[0] == 0:
                raise ValueError()
            logging.info('New Dimensions %s meta-data frame: %s' % (classifier_name,
                                                                    str(metadata_frame_classif.shape)))
            # add prefix to column names
            metadata_frame_classif = metadata_frame_classif.rename(index=str, columns={
                param: '%s:%s' % (classifier_name, param) for param in config_space.get_hyperparameter_names()
            })

            if metadata_frame_total is None:
                metadata_frame_total = metadata_frame_classif
            else:
                # TODO: due to a bug in pandas, we need to manually cast columns back to Int64.
                # See: https://github.com/pandas-dev/pandas/issues/24768
                # also add boolean columns (will be casted to int)
                int_columns = list(metadata_frame_classif.select_dtypes(include=['Int64']).columns) + \
                              list(metadata_frame_total.select_dtypes(include=['Int64']).columns)
                bool_columns = list(metadata_frame_classif.select_dtypes(include=[bool]).columns) + \
                               list(metadata_frame_total.select_dtypes(include=[bool]).columns)
                metadata_frame_total = metadata_frame_total.append(metadata_frame_classif)
                for column in int_columns:
                    metadata_frame_total[column] = metadata_frame_total[column].astype('Int64')
                for column in bool_columns:
                    metadata_frame_total[column] = metadata_frame_total[column].astype(float)

    logging.info('Loaded %d meta-data data frames. Dimensions: %s' % (len(metadata_files),
                                                                      str(metadata_frame_total.shape)))
    logging.info('Final Columns: %s' % metadata_frame_classif.columns.values)
    return metadata_frame_total


def store_surrogate_based_results(scoring_frame: pd.DataFrame,
                                  timing_frame: typing.Optional[pd.DataFrame],
                                  task_id: typing.Union[int, str],
                                  indice_order: typing.List[int],
                                  scoring: str,
                                  usercpu_time: str,
                                  minimize_measure: bool,
                                  result_filepath_results: str):
    """
    Stores the results of the surrogated based experiment to a result file.
    """
    if not 'task_%s' % task_id in scoring_frame.columns.values:
        raise ValueError()
    if timing_frame is not None:
        if not 'task_%s' % task_id in timing_frame.columns.values:
            raise ValueError()
        if not scoring_frame.index.equals(timing_frame.index):
            raise ValueError()

    os.makedirs(os.path.dirname(result_filepath_results), exist_ok=True)
    with open(result_filepath_results, 'w') as csvfile:
        best_score = 1.0 if minimize_measure else 0.0
        total_time = 0.0

        fieldnames = ['iteration', usercpu_time, scoring]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({scoring: best_score, usercpu_time: total_time, 'iteration': 0})
        for idx in indice_order:
            current_score = scoring_frame.iloc[idx]['task_%s' % task_id]
            current_time = np.nan
            if timing_frame is not None:
                current_time = timing_frame.iloc[idx]['task_%s' % task_id]
            # Note that this is not the same as `minimize'. E.g., when generating sets of defaults while minimizing
            # runtime, we still want to select the best default based on the criterion of the original measure
            if minimize_measure:
                best_score = min(best_score, current_score)
            else:
                best_score = max(best_score, current_score)
            total_time = current_time + total_time
            writer.writerow({scoring: best_score, usercpu_time: total_time, 'iteration': idx+1})


def single_prediction(df: pd.DataFrame,
                      surrogate: sklearn.pipeline.Pipeline,
                      config: typing.Dict) -> float:
    df = pd.DataFrame(columns=df.columns.values)
    df = df.append(config, ignore_index=True)
    return surrogate.predict(df.values)[0]
