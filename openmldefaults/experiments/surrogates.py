import joblib
import logging
import openmldefaults
import os
import pickle
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
                                 output_directory: str):
    logging.info('Starting Random Search Experiment')
    usercpu_time = 'usercpu_time_millis'
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
    metadata_frame = metadata_files_to_frame(metadata_files, search_space_identifier, [scoring, usercpu_time])
    task_ids = list(metadata_frame['task_id'].unique())

    config_frame = dict()
    for measure in [scoring, usercpu_time]:
        logging.info('Generating surrogated frames for measure: %s. Columns: %s' % (measure, metadata_frame.columns.values))
        surrogates = generate_surrogates_using_metadata(metadata_frame,
                                                        configurations,
                                                        config_space,
                                                        measure,
                                                        surrogate_minimum_evals,
                                                        surrogate_n_estimators,
                                                        random_seed)
        config_frame[measure] = openmldefaults.utils.generate_dataset_using_surrogates(
            surrogates, task_ids, config_space, configurations, None, None, -1)

    for task_id in task_ids:
        result_directory = os.path.join(output_directory, classifier_identifier, str(int(task_id)), strategy_name, str(random_seed))
        result_filepath_results = os.path.join(result_directory, 'results_%d_%d.csv' % (n_defaults, minimize_measure))
        openmldefaults.utils.store_surrogate_based_results(config_frame[scoring],
                                                           config_frame[usercpu_time],
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
                                   surrogate_minimum_evals: int, task_limit: int,
                                   output_directory: str):
    if a3r_r % 2 == 0 and normalize_base == 'StandardScaler':
        raise ValueError('Incompatible experiment parameters.')
    
    logging.info('Starting Default Search Experiment on Task %d' % task_id)
    model = openmldefaults.models.GreedyDefaults()
    usercpu_time = 'usercpu_time_millis'
    a3r = 'a3r'

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
    metadata_frame = metadata_files_to_frame(metadata_files, search_space_identifier, [scoring, usercpu_time])

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
    for measure, normalize in [(scoring, normalize_base), (usercpu_time, normalize_base)]:
        logging.info('Generating surrogated frames for measure: %s. Columns: %s' % (measure, metadata_frame.columns.values))
        surrogates = generate_surrogates_using_metadata(metadata_frame,
                                                        configurations,
                                                        config_space,
                                                        measure,
                                                        surrogate_minimum_evals,
                                                        surrogate_n_estimators,
                                                        random_seed)
        frame_tr = openmldefaults.utils.generate_dataset_using_surrogates(
            surrogates, tasks_tr, config_space, configurations, normalize, None, -1)
        config_frame_tr[measure] = frame_tr
        frame_te = openmldefaults.utils.generate_dataset_using_surrogates(
            surrogates, tasks_te, config_space, configurations, normalize, None, -1)
        config_frame_te[measure] = frame_te
    # adds A3R frame
    config_frame_tr[a3r] = openmldefaults.utils.create_a3r_frame(config_frame_tr[scoring],
                                                                 config_frame_tr[usercpu_time],
                                                                 a3r_r)

    config_frame_tr[a3r] = openmldefaults.utils.normalize_df_columnwise(config_frame_tr[a3r], normalize_a3r)

    # whether to optimize scoring is parameterized, same for a3r (which follows from scoring). runtime always min
    for measure, minimize in [(scoring, minimize_measure), (usercpu_time, True), (a3r, minimize_measure)]:
        logging.info('Started measure %s, minimize: %d' % (measure, minimize))
        strategy = '%s_%s' % ('min' if minimize else 'max', measure)
        result_directory = os.path.join(output_directory, classifier_identifier, str(task_id), strategy,
                                        str(random_seed), aggregate, str(a3r_r), str(normalize_base),
                                        str(normalize_a3r))
        os.makedirs(result_directory, exist_ok=True)
        result_filepath_defaults = os.path.join(result_directory, 'defaults_%d_%d.pkl' % (n_defaults, minimize))
        result_filepath_results = os.path.join(result_directory, 'results_%d_%d.csv' % (n_defaults, minimize))

        result = generate_defaults_discretized(config_frame_tr[measure], n_defaults, minimize, AGGREGATES[aggregate], False)
        with open(result_filepath_defaults, 'wb') as fp:
            pickle.dump(result, fp, protocol=0)
        logging.info('defaults generated, saved to: %s' % result_filepath_defaults)
        openmldefaults.utils.store_surrogate_based_results(config_frame_te[scoring],
                                                           config_frame_te[usercpu_time],
                                                           task_id,
                                                           result['indices'],
                                                           scoring,
                                                           usercpu_time,
                                                           minimize,
                                                           result_filepath_results)
        logging.info('results generated, saved to: %s' % result_filepath_results)
