import argparse
import openml
import openmldefaults
import os
import pandas as pd


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/'
                                                          'surrogate__adaboost__predictive_accuracy__c8.arff')
    parser.add_argument('--input_dir', type=str,
                        default=os.path.expanduser('~') + '/experiments/openml-defaults')
    parser.add_argument('--experiment_prefix', type=str, default='20180826')
    parser.add_argument('--dir_structure', type=str, nargs='+', default=['configuration_specification', 'strategy_name', 'task_id'])
    return parser.parse_args()


def construct_path(base_directory, recursed_directories):
    directory_path = base_directory
    for item in recursed_directories:
        directory_path = os.path.join(directory_path, item)
        if not os.path.isdir(directory_path):
            raise ValueError('Directory does not exist: %s' % directory_path)
    return directory_path


def load_result(base_directory, recursed_directories, required_dir_structure, scoring):
    current_dir = construct_path(base_directory, recursed_directories)
    run = openml.runs.OpenMLRun.from_filesystem(current_dir, expect_model=False)
    sklearn_metric, kwargs = openmldefaults.utils.openml_sklearn_metric_fn_mapping(scoring)
    evaluation_scores = run.get_metric_fn(sklearn_metric, kwargs)
    evaluation_avg = sum(evaluation_scores) / len(evaluation_scores)
    row = {key: value for key, value in zip(required_dir_structure, recursed_directories)}
    row['evaluation'] = evaluation_avg
    return pd.DataFrame([row])


def recurse(base_directory, recursed_directories, required_dir_structure, scoring):
    if len(recursed_directories) == len(required_dir_structure):
        current_dir = construct_path(base_directory, recursed_directories)
        # assumption: if description exists, the run exists
        if os.path.isfile(os.path.join(current_dir, 'description.xml')):
            return load_result(base_directory,
                               recursed_directories,
                               required_dir_structure,
                               scoring)
        else:
            # description does not exists
            return None

    result = None
    current_dir = construct_path(base_directory, recursed_directories)
    for sub_directory in os.listdir(current_dir):
        if os.path.isfile(sub_directory):
            continue
        subresult = recurse(base_directory,
                            recursed_directories + [sub_directory],
                            required_dir_structure,
                            scoring)
        if result is None:
            result = subresult
        elif subresult is not None:
            result = result.append(subresult)
    return result


def run(input_dir, experiment_prefix, dataset_path, dir_structure):
    if not os.path.isdir(input_dir):
        raise ValueError('Input directory does not exists: %s' % input_dir)
    dataset_name = os.path.basename(dataset_path)
    strategies_dir = os.path.join(input_dir, experiment_prefix, dataset_name, 'live_random_search')
    if not os.path.isdir(strategies_dir):
        raise ValueError('Could not find strategies directory: %s' % strategies_dir)
    meta_data = openmldefaults.utils.get_dataset_metadata(dataset_path)
    results = recurse(strategies_dir, [], dir_structure, meta_data['scoring'])
    if results is None:
        raise ValueError('Did not obtain any results from directory: %s' % strategies_dir)
    result_path = os.path.join(strategies_dir, 'results.csv')
    results.to_csv(path_or_buf=result_path, sep=',')
    print('Saved to %s' %result_path)
    pass


if __name__ == '__main__':
    args_ = parse_args()
    run(args_.input_dir, args_.experiment_prefix, args_.dataset_path, args_.dir_structure)
