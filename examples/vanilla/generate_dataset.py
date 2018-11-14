import arff
import argparse
import json
import logging
import numpy as np
import openml
import openmlcontrib
import openmldefaults
import os
import pandas as pd
import sklearnbot


def parse_args():
    metadata_file = '/home/janvanrijn/experiments/sklearn-bot/results/results__500__svc__predictive_accuracy.arff'
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/generated_data/')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--metadata_file', type=str, default=metadata_file)
    parser.add_argument('--classifier_name', type=str, default='svc', help='scikit-learn flow name')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=42)
    return parser.parse_args()


def run(args):
    study = openml.study.get_study(args.study_id, 'tasks')
    config_space = sklearnbot.config_spaces.get_config_space(args.classifier_name, args.random_seed)

    num_params = len(config_space.get_hyperparameter_names())
    configurations = openmldefaults.utils.generate_grid_configurations(config_space, 0, args.resized_grid_size)

    df_orig = pd.DataFrame(configurations)
    logging.info('Meta-dataset dimensions: %s' % str(df_orig.shape))

    meta_data = openmldefaults.utils.get_dataset_metadata(args.metadata_file)
    if args.scoring not in meta_data['measure']:
        raise ValueError('Could not find measure: %s' % args.scoring)
    metadata_frame = openmldefaults.utils.metadata_file_to_frame(args.metadata_file, config_space, args.scoring)

    # copy of df_orig. Prevent copy function for correct type hints
    df_surrogate = pd.DataFrame(configurations)
    for task_id in study.tasks:
        setup_frame = pd.DataFrame(metadata_frame.loc[metadata_frame['task_id'] == task_id])
        del setup_frame['task_id']
        try:
            estimator, columns = openmldefaults.utils.train_surrogate_on_task(task_id,
                                                                              config_space,
                                                                              setup_frame,
                                                                              args.scoring,
                                                                              64,
                                                                              args.random_seed)
        except ValueError as e:
            print('Error at task %d: %s' % (task_id, e))
            continue
        if not np.array_equal(df_orig.columns.values, columns):
            # if this goes wrong, it is due to the pd.get_dummies() fn
            raise ValueError('Column sets not equal: %s vs %s' % (df_orig.columns.values, columns))
        surrogate_values = estimator.predict(df_orig.values)
        df_surrogate['%s_task_%d' % (args.scoring, task_id)] = surrogate_values

    if df_surrogate.shape[0] != len(configurations):
        raise ValueError('surrogate frame has wrong number of instances. Expected: %d Got %d' % (len(configurations),
                                                                                                 df_surrogate.shape[0]))

    if df_surrogate.shape[1] > num_params + len(study.tasks):
        raise ValueError('surrogate frame has too much of columns. Max: %d Got %d' % (num_params + len(study.tasks),
                                                                                      df_surrogate.shape[1]))
    # if df_surrogate.shape[1] < num_params + len(study.tasks) / 2:
    #    raise ValueError('surrogate frame has too few columns. Min: %d Got %d' % (num_params + len(study.tasks) / 2,
    #                                                                              df_surrogate.shape[1]))
    os.makedirs(args.output_directory, exist_ok=True)
    arff_object = openmlcontrib.meta.dataframe_to_arff(df_surrogate,
                                                       'surrogate_%s' % args.classifier_name,
                                                       json.dumps(meta_data))

    with open(os.path.join(args.output_directory, 'surrogate__%s__%s__c%d.arff' % (args.classifier_name,
                                                                                   args.scoring,
                                                                                   args.resized_grid_size)), 'w') as fp:
        arff.dump(arff_object, fp)


if __name__ == '__main__':
    run(parse_args())
