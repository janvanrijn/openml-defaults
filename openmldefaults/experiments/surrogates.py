import csv
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
    
    logging.info('Starting on Task %d' % task_id)
    model = openmldefaults.models.GreedyDefaults()
    usercpu_time = 'usercpu_time_millis'
    a3r = 'a3r'

    # joblib speed ups
    memory = joblib.Memory(location=os.path.join(output_directory, '.cache'), verbose=0)
    metadata_files_to_frame = memory.cache(openmldefaults.utils.metadata_files_to_frame)
    generate_surrogates_using_metadata = memory.cache(openmldefaults.utils.generate_surrogates_using_metadata)
    generate_defaults_discretized = memory.cache(model.generate_defaults_discretized)

    classifier_names = [os.path.splitext(os.path.basename(file))[0] for file in metadata_files]
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
        classifier_identifier = '__'.join(sorted(classifier_names))
        result_directory = os.path.join(output_directory, classifier_identifier, str(task_id), strategy, aggregate,
                                        str(a3r_r), str(normalize_base), str(normalize_a3r))
        os.makedirs(result_directory, exist_ok=True)
        result_filepath_defaults = os.path.join(result_directory, 'defaults_%d_%d.pkl' % (n_defaults, minimize))
        result_filepath_results = os.path.join(result_directory, 'results_%d_%d.csv' % (n_defaults, minimize))

        result = generate_defaults_discretized(config_frame_tr[measure], n_defaults, minimize, AGGREGATES[aggregate], False)
        with open(result_filepath_defaults, 'wb') as fp:
            pickle.dump(result, fp, protocol=0)
        logging.info('defaults generated, saved to: %s' % result_filepath_defaults)

        with open(result_filepath_results, 'w') as csvfile:
            best_score = 1.0 if minimize_measure else 0.0
            total_time = 0.0

            fieldnames = [usercpu_time, scoring]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({scoring: best_score, usercpu_time: total_time})
            for idx, default in zip(result['indices'], result['defaults']):
                current_score = config_frame_te[scoring].iloc[idx]['task_%d' % task_id]
                current_time = config_frame_te[usercpu_time].iloc[idx]['task_%d' % task_id]
                # Note that this is not the same as `minimize'. E.g., when generating sets of defaults while minimizing
                # runtime, we still want to select the best default based on the criterion of the original measure
                if minimize_measure:
                    best_score = min(best_score, current_score)
                else:
                    best_score = max(best_score, current_score)
                total_time = current_time + total_time
                writer.writerow({scoring: best_score, usercpu_time: total_time})
        logging.info('results generated, saved to: %s' % result_filepath_results)
