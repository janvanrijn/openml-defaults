import logging
import numpy as np
import openml
import openmlcontrib
import openmldefaults
import os
import pickle
import sklearn.model_selection
import statistics
import typing

from openmldefaults.models.defaults_generator_interface import DefaultsGenerator


AGGREGATES = {
    'median': statistics.median,
    'min': min,
    'max': max,
    'sum': sum,
}


def get_scores_live(task_id: int, defaults: typing.List[typing.Dict], search_space_identifier: str, scoring: str):
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y()
    dataset = task.get_dataset()
    nominal_indices = dataset.get_features_by_type('nominal', [task.target_name])
    numeric_indices = dataset.get_features_by_type('numeric', [task.target_name])

    res = openmldefaults.search.convert_defaults_to_multiple_param_grids(defaults,
                                                                         'classifier',
                                                                         search_space_identifier,
                                                                         numeric_indices=numeric_indices,
                                                                         nominal_indices=nominal_indices)
    classifiers, param_grids = res
    sklearn_measure, sklearn_maximize = openmldefaults.utils.openml_measure_to_sklearn(scoring)
    search_clf = openmldefaults.search.EstimatorSelectionHelper(classifiers, param_grids,
                                                                cv=3, n_jobs=1, verbose=True,
                                                                scoring=sklearn_measure,
                                                                maximize=sklearn_maximize)
    scores = sklearn.model_selection.cross_val_score(search_clf, X, y,
                                                     scoring=sklearn_measure,
                                                     cv=10, verbose=1)
    return scores


def override_parameter_in_conf(configuration: typing.Dict, override_parameter: typing.Optional[typing.Dict]):
    """
    Given a configuration dict (mapping from hyperparameter name to value), it will override the values using an
    override dict (mapping from hyperparameter name to new value)
    """
    if override_parameter is None:
        return configuration
    for key, new_value in override_parameter.items():
        if key not in configuration:
            raise ValueError()
        else:
            configuration[key] = new_value
    return configuration


def run_vanilla_surrogates_on_task(task_id: typing.Optional[int],
                                   metadata_files: typing.List[str],
                                   models: typing.List[DefaultsGenerator],
                                   use_surrogates: bool,
                                   random_seed: int,
                                   search_space_identifier: typing.Optional[str],
                                   scoring: str, minimize_measure: bool,
                                   defaults_sizes: typing.List[int],
                                   n_configurations: int,
                                   aggregate: str, a3r_r: int,
                                   normalize_base: str, normalize_a3r: str,
                                   surrogate_n_estimators: int,
                                   surrogate_minimum_evals: int,
                                   runtime_column: typing.Optional[str],
                                   consider_a3r: bool,
                                   task_limit: typing.Optional[int],
                                   evaluate_on_surrogate: bool,
                                   output_directory: str,
                                   task_id_column: str,
                                   skip_row_check: bool):
    """
    Flexible running script that performs experiments based on surrogated default generation on a single task.
    Has capabilities to use active testing, combine various search spaces and incorporate A3R.

    Parameters
    ----------
    task_id: int (Optional)
        if this variable is set, this is considered the test task. It will be removed from the meta data, the defaults
        will be generated on the other tasks and holdout experiment will be performed on this task. If set to None,
        the defaults will be generated on all tasks but no experiment will be performed.
    metadata_files: List[str]
        list of filepaths pointing to meta-data files. They should be in extended arff format, i.e., vanilla arff with
        as first line a comment with json content describing which columns are hyperparameters and which columns are
        performance measures
    models: list[DefaultsGenerator]
        A list of models to generate the defaults
    use_surrogates: bool
        The way configurations are sampled. If set to False, no new configurations will be sampled and the meta-data
        will be used. If set to True, surrogates will be trained and configurations will be sampled from the surrogated
        space.
    random_seed: int
        the random seed to perform the experiment with. Will be used for (i) numpy, (ii) config space and
        (iii) surrogate
    search_space_identifier: str (Optional)
        determines how to obtain the config space. Leave to None to obtain default version from sklearn bot. Set to
        specific name to obtain from this package
    scoring: str
        The main measure to consider. Note that this has to be a column in all dataframes loaded from metadata_files
    minimize_measure: str
        Whether to minimize this measure (e.g., loss should be minimized, whereas accuracy should be maximized)
    defaults_sizes: List[int]
        For each entry of the list, the number of defaults that needs to be generated. Note that depending on the
        search criterion, active testing can not always generate this number of defaults.
    n_configurations: int
        Number of configurations to sample
    aggregate: str
        Determines how to aggregate scores per train task
    a3r_r: str
        The r-parameter of the A3R metric (if applied)
    normalize_base: str
        A string to identify the normalization module to normalize the train frames with scoring and runtime results
    normalize_a3r: str
        A string to identify the normalization module to normalize the A3R frame (if applicable)
    surrogate_n_estimators: int
        The number of estimators hyperparameter of the surrogate
    surrogate_minimum_evals: int
        Hyperparameter identifying the minumum number of data points for a specific task required to build a surrogate
    runtime_column: str (optional)
        The name of the column in the dataframe loaded based on metadata_files where runtime values are stored. If set,
        also the `minimize runtime' baseline will be calculated
    consider_a3r: bool
        Whether to calculate results based on A3R
    task_limit: int (Optional)
        If set, only this number of tasks will be considered in the train set (speed)
    evaluate_on_surrogate: bool
        Whether to perform a surrogated experiment or a live experiment
    output_directory: str
        Where to place the results (will create a directory structure)
    task_id_column: str
        The name of the column in the dataframe loaded based on metadata_files where task ids are stored
    skip_row_check: bool
        Usually the config space library checks for every configuration whether it falls within a given config space.
        This can be disabled (speed)
    """
    np.random.seed(random_seed)
    if a3r_r % 2 == 0 and normalize_base == 'StandardScaler':
        raise ValueError('Incompatible experiment parameters.')
    if consider_a3r and runtime_column is None:
        raise ValueError('Can only consider a3r when runtime is also considered.')
    
    logging.info('Starting Default Search Experiment on Task %s' % task_id)
    a3r = 'a3r'
    measures = [scoring]

    if runtime_column:
        measures = [scoring, runtime_column]

    classifier_names = [os.path.splitext(os.path.basename(file))[0] for file in metadata_files]
    classifier_identifier = '__'.join(sorted(classifier_names))
    config_space = openmldefaults.config_spaces.get_config_spaces(classifier_names,
                                                                  random_seed,
                                                                  search_space_identifier)
    configuration_sampler = openmldefaults.symbolic.VanillaConfigurationSpaceSampler(config_space)
    configurations = configuration_sampler.sample_configurations(n_configurations)
    metadata_frame = openmldefaults.utils.metadata_files_to_frame(metadata_files,
                                                                  search_space_identifier,
                                                                  measures,
                                                                  task_id_column,
                                                                  skip_row_check)
    logging.info(metadata_frame.columns.values)

    # this ensures that we only take tasks on which a surrogate was trained
    # (note that not all tasks do have meta-data, due to problems on OpenML)
    tasks_all = list(metadata_frame[task_id_column].unique())
    # obtain meta-features
    meta_features = openmlcontrib.meta.get_tasks_qualities_as_dataframe(tasks_all, False, -1.0, True, False)
    if set(tasks_all) != set(meta_features.index.values):
        missing = set(tasks_all) - set(meta_features.index.values)
        if task_id in missing:
            raise ValueError('Missing meta-data for test task %s' % missing)
        logging.warning('Could not obtain meta-features for tasks %s, dropping. ' % missing)
        tasks_all = list(meta_features.index.values)

    tasks_tr = list(tasks_all)
    # remove train task from list
    if task_id is not None:
        tasks_tr.remove(task_id)
        if task_limit:
            tasks_tr = tasks_tr[:task_limit]
        tasks_te = [task_id]
    else:
        tasks_te = []

    config_frame_tr = dict()
    config_frame_te = dict()
    measures_normalize = [(scoring, normalize_base)]
    
    if runtime_column:
        measures_normalize.append((runtime_column, normalize_base))
    for measure, normalize in measures_normalize:
        logging.info('Generating surrogated frames for measure: %s. Columns: %s' % (measure, metadata_frame.columns.values))
        if use_surrogates:
            surrogates, columns = openmldefaults.utils.generate_surrogates_using_metadata(
                metadata_frame=metadata_frame,
                hyperparameter_names=configuration_sampler.get_hyperparameter_names(),
                scoring=measure,
                minimum_evals=surrogate_minimum_evals,
                n_estimators=surrogate_n_estimators,
                random_seed=random_seed,
                task_id_column=task_id_column
            )
            frame_tr = openmldefaults.utils.generate_dataset_using_surrogates(
                surrogates=surrogates,
                surrogate_columns=columns,
                task_ids=tasks_tr,
                meta_features=meta_features,
                configurations=configurations,
                n_configurations=n_configurations,
                scaler_type=normalize,
                column_prefix=None,
                fill_nans=None
            )
        else:
            surrogates = None
            columns = None
            frame_tr = openmldefaults.utils.generate_dataset_using_metadata(
                metadata_frame=metadata_frame,
                task_ids=tasks_tr,
                hyperparameter_names=configuration_sampler.get_hyperparameter_names(),
                measure=measure,
                task_id_column=task_id_column,
                scaler_type=normalize,
                column_prefix=None,
            )
        config_frame_tr[measure] = frame_tr
        if task_id:
            # NEVER! Normalize the test frame
            if use_surrogates:
                frame_te = openmldefaults.utils.generate_dataset_using_surrogates(
                    surrogates=surrogates,
                    surrogate_columns=columns,
                    task_ids=tasks_te,
                    meta_features=meta_features,
                    configurations=configurations,
                    n_configurations=n_configurations,
                    scaler_type=None,
                    column_prefix=None,
                    fill_nans=None
                )
            else:
                frame_te = openmldefaults.utils.generate_dataset_using_metadata(
                    metadata_frame=metadata_frame,
                    task_ids=tasks_te,
                    hyperparameter_names=configuration_sampler.get_hyperparameter_names(),
                    measure=measure,
                    task_id_column=task_id_column,
                    scaler_type=None,
                    column_prefix=None,
                )
            config_frame_te[measure] = frame_te
            logging.info('Ranges test task %s for measure %s [%f-%f]:' % (task_id,
                                                                          measure,
                                                                          min(frame_te['task_%s' % task_id]),
                                                                          max(frame_te['task_%s' % task_id])))
    if consider_a3r:
        # adds A3R frame
        config_frame_tr[a3r] = openmldefaults.utils.create_a3r_frame(config_frame_tr[scoring],
                                                                     config_frame_tr[runtime_column],
                                                                     a3r_r)

        config_frame_tr[a3r] = openmldefaults.utils.normalize_df_columnwise(config_frame_tr[a3r], normalize_a3r)

    # whether to optimize scoring is parameterized, same for a3r (which follows from scoring). runtime always min
    for model in models:
        for measure, minimize in [(scoring, minimize_measure), (runtime_column, True), (a3r, minimize_measure)]:
            if measure not in config_frame_tr:
                continue
            for n_defaults in defaults_sizes:
                logging.info('Started measure %s, minimize: %d, n_defaults %d' % (measure, minimize, n_defaults))
                strategy = '%s_%s_%s' % (model.name, 'min' if minimize else 'max', measure)
                result_directory = os.path.join(output_directory, classifier_identifier, str(task_id), strategy,
                                                str(n_defaults), str(n_configurations), str(random_seed), aggregate,
                                                str(a3r_r), str(normalize_base), str(normalize_a3r))
                os.makedirs(result_directory, exist_ok=True)
                result_filepath_defaults = os.path.join(result_directory, 'defaults_%d_%d.pkl' % (n_defaults, minimize))

                if os.path.isfile(result_filepath_defaults):
                    logging.info('will load defaults from: %s' % result_filepath_defaults)
                    with open(result_filepath_defaults, 'rb') as fp:
                        result_indices, result_defaults, meta_data = pickle.load(fp)
                else:
                    result_indices, meta_data = model.generate_defaults_discretized(
                        config_frame_tr[measure], n_defaults, minimize, AGGREGATES[aggregate], config_space, False)
                    # note that a result without defaults is wrong, although having less defaults than requested
                    # is fine
                    if len(result_indices) == 0:
                        raise ValueError('No defaults selected')
                    if len(result_indices) > n_defaults:
                        # slice to have the exact number of requested defaults (or less)
                        result_indices = result_indices[0: n_defaults]
                    task_meta_features = meta_features.loc[task_id].to_dict()
                    result_defaults = [
                        config_frame_tr[scoring].index[idx].get_dictionary(task_meta_features) for idx in result_indices
                    ]

                    with open(result_filepath_defaults, 'wb') as fp:
                        pickle.dump([result_indices, result_defaults, meta_data], fp, protocol=0)
                    logging.info('defaults generated, saved to: %s' % result_filepath_defaults)

                if not task_id:
                    logging.warning('No test task id specified. Will not perform experiment.')
                else:
                    if evaluate_on_surrogate:
                        result_filepath_experiment = os.path.join(result_directory, 'surrogated_%d_%d.csv' % (n_defaults,
                                                                                                              minimize))
                        if not os.path.exists(result_filepath_experiment):
                            openmldefaults.utils.store_surrogate_based_results(config_frame_te[scoring],
                                                                               config_frame_te[runtime_column] if runtime_column else None,
                                                                               task_id,
                                                                               result_indices,
                                                                               scoring,
                                                                               runtime_column,
                                                                               minimize_measure,
                                                                               result_filepath_experiment)
                            logging.info('surrogated results generated, saved to: %s' % result_filepath_experiment)
                        else:
                            logging.info('surrogated results already exists, see: %s' % result_filepath_experiment)
                    else:  # run on live
                        result_filepath_experiment = os.path.join(result_directory, 'live_%d_%d.pkl' % (n_defaults,
                                                                                                        minimize))
                        if not os.path.exists(result_filepath_experiment):
                            scores = get_scores_live(task_id, result_defaults, search_space_identifier, scoring)
                            with open(result_filepath_experiment, 'wb') as fp:
                                pickle.dump(scores, fp, protocol=0)
                            logging.info('live results generated, saved to: %s' % result_filepath_experiment)
                        else:
                            logging.info('live results already exists, see: %s' % result_filepath_experiment)
