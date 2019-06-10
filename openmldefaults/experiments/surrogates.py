import logging
import numpy as np
import openml
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


def run_random_search_surrogates(metadata_files: typing.List[str], random_seed: int,
                                 search_space_identifier: str, scoring: str,
                                 minimize_measure: bool, n_defaults: int,
                                 surrogate_n_estimators: int,
                                 surrogate_minimum_evals: int,
                                 consider_runtime: bool,
                                 run_on_surrogate: bool,
                                 output_directory: str,
                                 task_id_column: str,
                                 skip_row_check: bool,
                                 override_parameters: typing.Dict[str, typing.Any]):
    np.random.seed(random_seed)
    logging.info('Starting Random Search Experiment')
    usercpu_time = 'usercpu_time_millis'
    measures = [scoring]
    if consider_runtime:
        measures = [scoring, usercpu_time]

    classifier_names = [os.path.splitext(os.path.basename(file))[0] for file in metadata_files]
    classifier_identifier = '__'.join(sorted(classifier_names))
    strategy_name = 'random_search_%s_%s' % ('min' if minimize_measure else 'max', scoring)
    config_space = openmldefaults.config_spaces.get_config_spaces(classifier_names,
                                                                  random_seed,
                                                                  search_space_identifier)
    configurations = [override_parameter_in_conf(c.get_dictionary(), override_parameters)
                      for c in config_space.sample_configuration(n_defaults)]
    logging.info(configurations)
    metadata_frame = openmldefaults.utils.metadata_files_to_frame(metadata_files,
                                                                  search_space_identifier,
                                                                  measures,
                                                                  task_id_column,
                                                                  skip_row_check=skip_row_check)
    task_ids = list(metadata_frame[task_id_column].unique())

    config_frame = dict()
    for measure in measures:
        logging.info('Generating surrogated frames for measure: %s. Columns: %s' % (measure, metadata_frame.columns.values))
        surrogates, columns = openmldefaults.utils.generate_surrogates_using_metadata(metadata_frame,
                                                                                      config_space,
                                                                                      measure,
                                                                                      surrogate_minimum_evals,
                                                                                      surrogate_n_estimators,
                                                                                      random_seed,
                                                                                      task_id_column)
        config_frame[measure] = openmldefaults.utils.generate_dataset_using_surrogates(
            surrogates, columns, task_ids, config_space, configurations, None, None, -1)

    for task_id in task_ids:
        result_directory = os.path.join(output_directory, classifier_identifier,
                                        str(task_id), strategy_name,
                                        str(n_defaults), str(random_seed))
        if run_on_surrogate:
            result_filepath_surrogated = os.path.join(result_directory, 'surrogated_%d_%d.csv' % (n_defaults, minimize_measure))
            if not os.path.exists(result_filepath_surrogated):
                openmldefaults.utils.store_surrogate_based_results(config_frame[scoring],
                                                                   config_frame[usercpu_time] if consider_runtime else None,
                                                                   task_id,
                                                                   list(range(n_defaults)),
                                                                   scoring,
                                                                   usercpu_time,
                                                                   minimize_measure,
                                                                   result_filepath_surrogated)
                logging.info('surrogated random search results generated, saved to: %s' % result_filepath_surrogated)
            else:
                logging.info('surrogated random search results already exists, see: %s' % result_filepath_surrogated)
        else:
            result_filepath_live = os.path.join(result_directory, 'live_%d_%d.csv' % (n_defaults, minimize_measure))
            if not os.path.exists(result_filepath_live):
                configs = [openmldefaults.utils.selected_row_to_config_dict(config_frame[scoring], idx, config_space) for idx in range(len(config_frame[scoring]))]
                scores = get_scores_live(task_id, configs, search_space_identifier, scoring)
                with open(result_filepath_live, 'wb') as fp:
                    pickle.dump(scores, fp, protocol=0)
                logging.info('live results generated, saved to: %s' % result_filepath_live)
            else:
                logging.info('live results already exists, see: %s' % result_filepath_live)


def run_vanilla_surrogates_on_task(task_id: typing.Optional[int],
                                   metadata_files: typing.List[str],
                                   models: typing.List[DefaultsGenerator],
                                   use_surrogates: bool,
                                   random_seed: int,
                                   search_space_identifier: typing.Optional[str],
                                   scoring: str, minimize_measure: bool,
                                   n_defaults: typing.Optional[int], aggregate: str, a3r_r: int,
                                   normalize_base: str, normalize_a3r: str,
                                   surrogate_n_estimators: int,
                                   surrogate_minimum_evals: int,
                                   runtime_column: typing.Optional[str],
                                   consider_a3r: bool,
                                   task_limit: typing.Optional[int],
                                   run_on_surrogate: bool,
                                   output_directory: str,
                                   task_id_column: str,
                                   skip_row_check: bool,
                                   override_parameters: typing.Dict[str, typing.Any]):
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
    n_defaults: int (optional)
        The number of defaults that needs to be generated. Note that depending on the search criterion, active testing
        can not always generate this number of defaults. If set to None, the number of defaults in the meta-data is
        used.
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
    run_on_surrogate: bool
        Whether to perform a surrogated experiment or a live experiment
    output_directory: str
        Where to place the results (will create a directory structure)
    task_id_column: str
        The name of the column in the dataframe loaded based on metadata_files where task ids are stored
    skip_row_check: bool
        Usually the config space library checks for every configuration whether it falls within a given config space.
        This can be disabled (speed)
    override_parameters: dict
        TODO jan
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
    metadata_frame = openmldefaults.utils.metadata_files_to_frame(metadata_files,
                                                                  search_space_identifier,
                                                                  measures,
                                                                  task_id_column,
                                                                  skip_row_check)
    logging.info(metadata_frame.columns.values)

    # this ensures that we only take tasks on which a surrogate was trained
    # (note that not all tasks do have meta-data, due to problems on OpenML)
    tasks_tr = list(metadata_frame[task_id_column].unique())
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

    configurations_sampled = [override_parameter_in_conf(c.get_dictionary(), override_parameters)
                              for c in config_space.sample_configuration(n_defaults)]
    logging.info('Sampled Configurations: %s' % configurations_sampled)
    if runtime_column:
        measures_normalize.append((runtime_column, normalize_base))
    for measure, normalize in measures_normalize:
        logging.info('Generating surrogated frames for measure: %s. Columns: %s' % (measure, metadata_frame.columns.values))
        if use_surrogates:
            surrogates, columns = openmldefaults.utils.generate_surrogates_using_metadata(metadata_frame,
                                                                                          config_space,
                                                                                          measure,
                                                                                          surrogate_minimum_evals,
                                                                                          surrogate_n_estimators,
                                                                                          random_seed,
                                                                                          task_id_column)
            frame_tr = openmldefaults.utils.generate_dataset_using_surrogates(
                surrogates, columns, tasks_tr, config_space, configurations_sampled, normalize, None, None)
        else:
            surrogates = None
            columns = None
            frame_tr = openmldefaults.utils.generate_dataset_using_metadata(
                metadata_frame, tasks_tr, config_space, measure, task_id_column, normalize, None)
        config_frame_tr[measure] = frame_tr
        if task_id:
            # NEVER! Normalize the test frame
            if use_surrogates:
                frame_te = openmldefaults.utils.generate_dataset_using_surrogates(
                    surrogates, columns, tasks_te, config_space, configurations_sampled, None, None, None)
            else:
                frame_te = openmldefaults.utils.generate_dataset_using_metadata(
                    metadata_frame, tasks_te, config_space, measure, task_id_column, None, None)
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

            logging.info('Started measure %s, minimize: %d' % (measure, minimize))
            strategy = '%s_%s_%s' % (model.name, 'min' if minimize else 'max', measure)
            result_directory = os.path.join(output_directory, classifier_identifier, str(task_id), strategy,
                                            str(n_defaults), str(random_seed), aggregate, str(a3r_r), str(normalize_base),
                                            str(normalize_a3r))
            os.makedirs(result_directory, exist_ok=True)
            result_filepath_defaults = os.path.join(result_directory, 'defaults_%d_%d.pkl' % (n_defaults, minimize))
            exp_type = 'surrogated' if run_on_surrogate else 'live'
            result_filepath_experiment = os.path.join(result_directory, '%s_%d_%d.csv' % (exp_type, n_defaults, minimize))

            if os.path.isfile(result_filepath_defaults):
                with open(result_filepath_defaults, 'rb') as fp:
                    result_defaults = pickle.load(fp)
                logging.info('defaults loaded, loaded from: %s' % result_filepath_defaults)
            else:
                result_defaults = model.generate_defaults_discretized(config_frame_tr[measure],
                                                                      n_defaults,
                                                                      minimize,
                                                                      AGGREGATES[aggregate],
                                                                      config_space,
                                                                      False)
                if len(result_defaults['defaults']) == 0:
                    raise ValueError('No defaults selected')
                with open(result_filepath_defaults, 'wb') as fp:
                    pickle.dump(result_defaults, fp, protocol=0)
                logging.info('defaults generated, saved to: %s' % result_filepath_defaults)

            if not task_id:
                logging.warning('No test task id specified. Will not perform experiment.')
            else:
                if run_on_surrogate:
                    if not os.path.exists(result_filepath_experiment):
                        openmldefaults.utils.store_surrogate_based_results(config_frame_te[scoring],
                                                                           config_frame_te[runtime_column] if runtime_column else None,
                                                                           task_id,
                                                                           result_defaults['indices'],
                                                                           scoring,
                                                                           runtime_column,
                                                                           minimize_measure,
                                                                           result_filepath_experiment)
                        logging.info('surrogated results generated, saved to: %s' % result_filepath_experiment)
                    else:
                        logging.info('surrogated results already exists, see: %s' % result_filepath_experiment)
                else:  # run on live
                    if not os.path.exists(result_filepath_experiment):
                        scores = get_scores_live(task_id, result_defaults['defaults'], search_space_identifier, scoring)
                        with open(result_filepath_experiment, 'wb') as fp:
                            pickle.dump(scores, fp, protocol=0)
                        logging.info('live results generated, saved to: %s' % result_filepath_experiment)
                    else:
                        logging.info('live results already exists, see: %s' % result_filepath_experiment)
