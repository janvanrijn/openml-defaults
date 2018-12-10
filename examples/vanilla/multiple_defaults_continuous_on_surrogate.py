import argparse
import logging
import pickle
import openml
import openmldefaults
import os
import pandas as pd
import sklearnbot


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    metadata_file = '/home/janvanrijn/experiments/sklearn-bot/results/results__500__svc__predictive_accuracy.arff'
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/vanilla_defaults/')
    parser.add_argument('--strategy', type=str, default='max_accuracy')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--metadata_file', type=str, default=metadata_file)
    parser.add_argument('--classifier_name', type=str, default='svc', help='scikit-learn flow name')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--n_defaults', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task_limit', type=int, default=4, help='For speed')
    return parser.parse_args()


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    minimize = False

    study = openml.study.get_study(args.study_id, 'tasks')
    task_id = study.tasks[args.task_idx]

    config_space = sklearnbot.config_spaces.get_config_space(args.classifier_name, args.random_seed)

    metadata_atts = openmldefaults.utils.get_dataset_metadata(args.metadata_file)
    if args.scoring not in metadata_atts['measure']:
        raise ValueError('Could not find measure: %s' % args.scoring)

    metadata_frame = openmldefaults.utils.metadata_file_to_frame(args.metadata_file, config_space, args.scoring)
    tasks_tr = list(metadata_frame['task_id'].unique())
    tasks_tr.remove(task_id)
    if args.task_limit:
        tasks_tr = tasks_tr[:args.task_limit]
    tasks_te = [task_id]

    configurations = openmldefaults.utils.generate_grid_configurations(config_space, 0, args.resized_grid_size)
    config_frame_tr = openmldefaults.utils.generate_grid_dataset(metadata_frame,
                                                                 configurations,
                                                                 tasks_tr,
                                                                 config_space,
                                                                 args.scoring,
                                                                 True,
                                                                 args.random_seed,
                                                                 False,
                                                                 -1)
    config_frame_te = openmldefaults.utils.generate_grid_dataset(metadata_frame,
                                                                 configurations,
                                                                 tasks_te,
                                                                 config_space,
                                                                 args.scoring,
                                                                 True,
                                                                 args.random_seed,
                                                                 False,
                                                                 -1)
    config_hash = openmldefaults.utils.hash_df(config_frame_tr)
    result_directory = os.path.join(args.output_directory, str(task_id), args.strategy)
    os.makedirs(result_directory, exist_ok=True)
    result_filepath_defaults = os.path.join(result_directory, 'defaults_%s_%d_%d.pkl' % (config_hash,
                                                                                         args.n_defaults,
                                                                                         minimize))
    result_filepath_results = os.path.join(result_directory, 'results_%s_%d_%d.pkl' % (config_hash,
                                                                                       args.n_defaults,
                                                                                       minimize))
    if os.path.isfile(result_filepath_defaults):
        with open(result_filepath_defaults, 'rb') as fp:
            result = pickle.load(fp)
        logging.info('defaults loaded from cache')
    else:
        model = openmldefaults.models.GreedyDefaults()
        result = model.generate_defaults(config_frame_tr, args.n_defaults, minimize)
        with open(result_filepath_defaults, 'wb') as fp:
            pickle.dump(result, fp, protocol=0)
        logging.info('defaults generated')

    holdout_score = sum(openmldefaults.utils.selected_set_index(config_frame_te, result['indices'], minimize))
    result['holdout_score'] = holdout_score
    with open(result_filepath_results, 'wb') as fp:
        pickle.dump(result, fp, protocol=0)
    logging.info('Saved result to %s' % result_filepath_results)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
