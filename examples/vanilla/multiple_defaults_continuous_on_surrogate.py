import argparse
import csv
import joblib
import logging
import pickle
import openml
import openmldefaults
import os
import pandas as pd


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    metadata_file = os.path.expanduser('~/data/openml-defaults/svc.arff')
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/vanilla_defaults/')
    parser.add_argument('--strategy', type=str, default='max_accuracy')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--metadata_file', type=str, default=metadata_file)
    parser.add_argument('--classifier_name', type=str, default='svc', help='scikit-learn flow name')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--search_space', type=str, default='small')
    parser.add_argument('--minimize', action='store_true')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--n_defaults', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task_limit', type=int, default=None, help='For speed')
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    memory = joblib.Memory(location=os.path.join(args.output_directory, '.cache'), verbose=0)
    usercpu_time = 'usercpu_time_millis'
    a3r = 'a3r'

    study = openml.study.get_study(args.study_id, 'tasks')
    task_id = study.tasks[args.task_idx]

    config_space = openmldefaults.config_spaces.get_config_space(args.classifier_name,
                                                                 args.random_seed,
                                                                 args.search_space)

    metadata_atts = openmldefaults.utils.get_dataset_metadata(args.metadata_file)
    if args.scoring not in metadata_atts['measure']:
        raise ValueError('Could not find measure: %s' % args.scoring)
    if usercpu_time not in metadata_atts['measure']:
        raise ValueError('Could not find measure: %s' % usercpu_time)
    measures = [args.scoring, usercpu_time]
    metadata_file_to_frame = memory.cache(openmldefaults.utils.metadata_file_to_frame)
    metadata_frame = metadata_file_to_frame(args.metadata_file, config_space, measures)

    tasks_tr = list(metadata_frame['task_id'].unique())
    tasks_tr.remove(task_id)
    if args.task_limit:
        tasks_tr = tasks_tr[:args.task_limit]
    tasks_te = [task_id]

    configurations = openmldefaults.utils.generate_grid_configurations(config_space, 0, args.resized_grid_size)
    config_frame_tr = dict()
    config_frame_te = dict()
    for measure, normalize in [(args.scoring, True), (usercpu_time, True)]:
        logging.info('Generating frames for measure: %s' % measure)
        frame_tr = openmldefaults.utils.generate_grid_dataset(metadata_frame,
                                                              configurations,
                                                              tasks_tr,
                                                              config_space,
                                                              measure,
                                                              normalize,
                                                              args.random_seed,
                                                              False,
                                                              -1)
        config_frame_tr[measure] = frame_tr
        frame_te = openmldefaults.utils.generate_grid_dataset(metadata_frame,
                                                              configurations,
                                                              tasks_te,
                                                              config_space,
                                                              measure,
                                                              False,
                                                              args.random_seed,
                                                              False,
                                                              -1)
        config_frame_te[measure] = frame_te
    # adds A3R and normalizes it
    config_frame_tr[a3r] = openmldefaults.utils.create_a3r_frame(config_frame_tr[args.scoring],
                                                                 config_frame_tr[usercpu_time])
    config_frame_tr[a3r] = openmldefaults.utils.normalize_df_columnwise(config_frame_tr[a3r])

    # whether to optimize scoring is parameterized, same for a3r (which follows from scoring). runtime always min
    for measure, minimize in [(args.scoring, args.minimize), (usercpu_time, True), (a3r, args.minimize)]:
        logging.info('Started measure %s, minimize: %d' % (measure, minimize))
        strategy = '%s_%s' % ('min' if minimize else 'max', measure)
        # config_hash = openmldefaults.utils.hash_df(config_frame_tr[measure])
        result_directory = os.path.join(args.output_directory, str(task_id), strategy)
        os.makedirs(result_directory, exist_ok=True)
        result_filepath_defaults = os.path.join(result_directory, 'defaults_%d_%d.pkl' % (args.n_defaults, minimize))
        result_filepath_results = os.path.join(result_directory, 'results_%d_%d.csv' % (args.n_defaults, minimize))

        model = openmldefaults.models.GreedyDefaults()
        result = model.generate_defaults(config_frame_tr[measure], args.n_defaults, minimize)
        with open(result_filepath_defaults, 'wb') as fp:
            pickle.dump(result, fp, protocol=0)
        logging.info('defaults generated, saved to: %s' % result_filepath_defaults)

        with open(result_filepath_results, 'w') as csvfile:
            best_score = 1.0 if args.minimize else 0.0
            total_time = 0.0

            fieldnames = [usercpu_time, args.scoring]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({args.scoring: best_score, usercpu_time: total_time})
            for idx, default in zip(result['indices'], result['defaults']):
                current_score = config_frame_te[args.scoring].iloc[idx]['task_%d' % task_id]
                current_time = config_frame_te[usercpu_time].iloc[idx]['task_%d' % task_id]
                if args.minimize:
                    best_score = min(best_score, current_score)
                else:
                    best_score = max(best_score, current_score)
                total_time = current_time + total_time
                writer.writerow({args.scoring: best_score, usercpu_time: total_time})
        logging.info('results generated, saved to: %s' % result_filepath_results)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
