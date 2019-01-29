import joblib
import logging
import numpy as np
import openml
import openmldefaults
import os
import pickle
import sklearn.model_selection
import statistics
import typing


AGGREGATES = {
    'median': statistics.median,
    'min': min,
    'max': max,
    'sum': sum,
}


def run_random_search_surrogates(metadata_files: typing.List[str], random_seed: int,
                                 search_space_identifier: str, scoring: str,
                                 minimize_measure: bool, n_defaults: int,
                                 surrogate_n_estimators: int,
                                 surrogate_minimum_evals: int,
                                 consider_runtime: bool,
                                 output_directory: str):
    np.random.seed(random_seed)
    logging.info('Starting Random Search Experiment')
    usercpu_time = 'usercpu_time_millis'
    measures = [scoring]
    if consider_runtime:
        measures = [scoring, usercpu_time]
    # joblib speed ups
    memory = joblib.Memory(location=os.path.join(output_directory, '.cache'), verbose=0)
    metadata_files_to_frame = memory.cache(openmldefaults.utils.metadata_files_to_frame)
    generate_surrogates_using_metadata = memory.cache(openmldefaults.utils.generate_surrogates_using_metadata)

    classifier_names = [os.path.splitext(os.path.basename(file))[0] for file in metadata_files]
    classifier_identifier = '__'.join(sorted(classifier_names))
    strategy_name = 'random_search_%s_%s' % ('min' if minimize_measure else 'max', scoring)
    config_space = openmldefaults.config_spaces.get_config_spaces(classifier_names,
                                                                  random_seed,
                                                                  search_space_identifier)
    configurations = [c.get_dictionary() for c in config_space.sample_configuration(n_defaults)]
    metadata_frame = metadata_files_to_frame(metadata_files, search_space_identifier, measures)
    task_ids = list(metadata_frame['task_id'].unique())

    config_frame = dict()
    for measure in measures:
        logging.info('Generating surrogated frames for measure: %s. Columns: %s' % (measure, metadata_frame.columns.values))
        surrogates, columns = generate_surrogates_using_metadata(metadata_frame,
                                                                 config_space,
                                                                 measure,
                                                                 surrogate_minimum_evals,
                                                                 surrogate_n_estimators,
                                                                 random_seed)
        config_frame[measure] = openmldefaults.utils.generate_dataset_using_surrogates(
            surrogates, columns, task_ids, config_space, configurations, None, None, -1)

    for task_id in task_ids:
        result_directory = os.path.join(output_directory, classifier_identifier,
                                        str(int(task_id)), strategy_name,
                                        str(n_defaults), str(random_seed))
        result_filepath_results = os.path.join(result_directory, 'results_%d_%d.csv' % (n_defaults, minimize_measure))
        openmldefaults.utils.store_surrogate_based_results(config_frame[scoring],
                                                           config_frame[usercpu_time] if consider_runtime else None,
                                                           task_id,
                                                           list(range(n_defaults)),
                                                           scoring,
                                                           usercpu_time,
                                                           minimize_measure,
                                                           result_filepath_results)


def run_vanilla_surrogates_on_task(task_id: int, metadata_files: typing.List[str],
                                   random_seed: int, search_space_identifier: str,
                                   n_configurations: int,
                                   scoring: str, minimize_measure: bool,
                                   n_defaults: int, aggregate: str, a3r_r: int,
                                   normalize_base: str, normalize_a3r: str,
                                   surrogate_n_estimators: int,
                                   surrogate_minimum_evals: int,
                                   consider_runtime: bool,
                                   consider_a3r: bool,
                                   task_limit: int,
                                   run_on_surrogate: bool,
                                   output_directory: str):
    np.random.seed(random_seed)
    if a3r_r % 2 == 0 and normalize_base == 'StandardScaler':
        raise ValueError('Incompatible experiment parameters.')
    if consider_a3r and not consider_runtime:
        raise ValueError('Can only consider a3r when runtime is also considered.')
    
    logging.info('Starting Default Search Experiment on Task %d' % task_id)
    model = openmldefaults.models.GreedyDefaults()
    usercpu_time = 'usercpu_time_millis'
    a3r = 'a3r'
    measures = [scoring]
    if consider_runtime:
        measures = [scoring, usercpu_time]

    # joblib speed ups
    memory = joblib.Memory(location=os.path.join(output_directory, '.cache'), verbose=0)
    metadata_files_to_frame = memory.cache(openmldefaults.utils.metadata_files_to_frame)
    generate_surrogates_using_metadata = memory.cache(openmldefaults.utils.generate_surrogates_using_metadata)
    generate_defaults_discretized = memory.cache(model.generate_defaults_discretized)

    classifier_names = [os.path.splitext(os.path.basename(file))[0] for file in metadata_files]
    classifier_identifier = '__'.join(sorted(classifier_names))
    config_space = openmldefaults.config_spaces.get_config_spaces(classifier_names,
                                                                  random_seed,
                                                                  search_space_identifier)
    configurations = [c.get_dictionary() for c in config_space.sample_configuration(n_configurations)]
    metadata_frame = metadata_files_to_frame(metadata_files, search_space_identifier, measures)

    # this ensures that we only take tasks on which a surrogate was trained
    # (note that not all tasks do have meta-data, due to problems on OpenML)
    tasks_tr = list(metadata_frame['task_id'].unique())
    # remove train task from list
    tasks_tr.remove(task_id)
    if task_limit:
        tasks_tr = tasks_tr[:task_limit]
    tasks_te = [task_id]

    config_frame_tr = dict()
    config_frame_te = dict()
    measures_normalize = [(scoring, normalize_base)]
    if consider_runtime:
        measures_normalize.append((usercpu_time, normalize_base))
    for measure, normalize in measures_normalize:
        logging.info('Generating surrogated frames for measure: %s. Columns: %s' % (measure, metadata_frame.columns.values))
        surrogates, columns = generate_surrogates_using_metadata(metadata_frame,
                                                                 config_space,
                                                                 measure,
                                                                 surrogate_minimum_evals,
                                                                 surrogate_n_estimators,
                                                                 random_seed)
        frame_tr = openmldefaults.utils.generate_dataset_using_surrogates(
            surrogates, columns, tasks_tr, config_space, configurations, normalize, None, None)
        config_frame_tr[measure] = frame_tr
        # NEVER! Normalize the test frame
        frame_te = openmldefaults.utils.generate_dataset_using_surrogates(
            surrogates, columns, tasks_te, config_space, configurations, None, None, None)
        config_frame_te[measure] = frame_te
        logging.info('Ranges test task %d for measure %s [%f-%f]:' % (task_id,
                                                                      measure,
                                                                      min(frame_te['task_%d' % task_id]),
                                                                      max(frame_te['task_%d' % task_id])))
    if consider_a3r:
        # adds A3R frame
        config_frame_tr[a3r] = openmldefaults.utils.create_a3r_frame(config_frame_tr[scoring],
                                                                     config_frame_tr[usercpu_time],
                                                                     a3r_r)

        config_frame_tr[a3r] = openmldefaults.utils.normalize_df_columnwise(config_frame_tr[a3r], normalize_a3r)

    # whether to optimize scoring is parameterized, same for a3r (which follows from scoring). runtime always min
    for measure, minimize in [(scoring, minimize_measure), (usercpu_time, True), (a3r, minimize_measure)]:
        if measure not in config_frame_tr:
            continue

        logging.info('Started measure %s, minimize: %d' % (measure, minimize))
        strategy = '%s_%s' % ('min' if minimize else 'max', measure)
        result_directory = os.path.join(output_directory, classifier_identifier, str(task_id), strategy,
                                        str(n_defaults), str(random_seed), aggregate, str(a3r_r), str(normalize_base),
                                        str(normalize_a3r))
        os.makedirs(result_directory, exist_ok=True)
        result_filepath_defaults = os.path.join(result_directory, 'defaults_%d_%d.pkl' % (n_defaults, minimize))
        result_filepath_surrogated = os.path.join(result_directory, 'surrogated_%d_%d.csv' % (n_defaults, minimize))
        result_filepath_live = os.path.join(result_directory, 'live_%d_%d.csv' % (n_defaults, minimize))

        if os.path.isfile(result_filepath_defaults):
            with open(result_filepath_defaults, 'rb') as fp:
                result_defaults = pickle.load(fp)
            logging.info('defaults loaded, loaded from: %s' % result_filepath_defaults)
        else:
            result_defaults = generate_defaults_discretized(config_frame_tr[measure],
                                                            n_defaults,
                                                            minimize,
                                                            AGGREGATES[aggregate],
                                                            config_space,
                                                            False)
            with open(result_filepath_defaults, 'wb') as fp:
                pickle.dump(result_defaults, fp, protocol=0)
            logging.info('defaults generated, saved to: %s' % result_filepath_defaults)

        if run_on_surrogate:
            if not os.path.exists(result_filepath_surrogated):
                openmldefaults.utils.store_surrogate_based_results(config_frame_te[scoring],
                                                                   config_frame_te[usercpu_time] if consider_runtime else None,
                                                                   task_id,
                                                                   result_defaults['indices'],
                                                                   scoring,
                                                                   usercpu_time,
                                                                   minimize_measure,
                                                                   result_filepath_surrogated)
                logging.info('surrogated results generated, saved to: %s' % result_filepath_surrogated)
            else:
                logging.info('surrogated results already exists, see: %s' % result_filepath_surrogated)
        else:  # run on live
            if not os.path.exists(result_filepath_live):
                task = openml.tasks.get_task(task_id)
                X, y = task.get_X_and_y()
                dataset = task.get_dataset()
                nominal_indices = dataset.get_features_by_type('nominal', [task.target_name])
                numeric_indices = dataset.get_features_by_type('numeric', [task.target_name])

                res = openmldefaults.search.convert_defaults_to_multiple_param_grids(result_defaults['defaults'],
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
                with open(result_filepath_live, 'wb') as fp:
                    pickle.dump(scores, fp, protocol=0)
                logging.info('live results generated, saved to: %s' % result_filepath_live)
            else:
                logging.info('live results already exists, see: %s' % result_filepath_live)
