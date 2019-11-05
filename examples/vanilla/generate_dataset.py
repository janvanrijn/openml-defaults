import sys
sys.path.append('/home/flo/Documents/projects/sklearn-bot')
sys.path.append('/home/flo/Documents/projects/openml-python-contrib')
sys.path.append('/home/flo/Documents/projects/openml-defaults')

import arff
import argparse
import json
import logging
import openmlcontrib
import openmldefaults
import os
import sklearnbot


def parse_args():
    # metadata_file = '/home/janvanrijn/experiments/sklearn-bot/results/results__500__svc__predictive_accuracy.arff'
    metadata_file = '/home/flo/Documents/projects/openml-defaults/data/small/classif_svm.arff'
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/generated_data/')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--metadata_file', type=str, default=[metadata_file])
    parser.add_argument('--classifier_name', type=str, default='svc', help='scikit-learn flow name')
    parser.add_argument('--task_id_column', type=str, default='task_id')
    parser.add_argument('--scoring', type=str, default='perf.mmce')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--skip_row_check', action='store_true')
    return parser.parse_args()


def run(args):
    args = parse_args()
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # config_space = sklearnbot.config_spaces.get_config_space(args.classifier_name, args.random_seed)

    # meta_data = openmldefaults.utils.get_dataset_metadata(args.metadata_file)
    # if args.scoring not in meta_data['col_measures']:
    #     raise ValueError('Could not find measure: %s' % args.scoring)
    metadata_frame = openmldefaults.utils.metadata_files_to_frame(args.metadata_file,
                                                                  config_space,
                                                                  [args.scoring],
                                                                  task_id_column=args.task_id_column,
                                                                  skip_row_check=args.skip_row_check)

    df_surrogate = openmldefaults.utils.generate_grid_dataset(metadata_frame,
                                                              config_space,
                                                              args.resized_grid_size,
                                                              args.scoring,
                                                              args.random_seed)
    # if df_surrogate.shape[1] < num_params + len(study.tasks) / 2:
    #    raise ValueError('surrogate frame has too few columns. Min: %d Got %d' % (num_params + len(study.tasks) / 2,
    #                                                                              df_surrogate.shape[1]))
    os.makedirs(args.output_directory, exist_ok=True)
    df_surrogate.reset_index(inplace=True)
    arff_object = openmlcontrib.meta.dataframe_to_arff(df_surrogate,
                                                       'surrogate_%s' % args.classifier_name,
                                                       json.dumps(meta_data))
    filename = os.path.join(args.output_directory, 'surrogate__%s__%s__c%d.arff' % (args.classifier_name,
                                                                                   args.scoring,
                                                                                   args.resized_grid_size))
    with open(filename, 'w') as fp:
        arff.dump(arff_object, fp)
    logging.info('Saved to: %s' % filename)


if __name__ == '__main__':
    run(parse_args())
