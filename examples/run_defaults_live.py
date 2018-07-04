import argparse
import json
import openmldefaults
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/svm-ongrid.arff')
    parser.add_argument('--config_space', type=str, default='libsvm_svc')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--num_defaults', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='cpp_bruteforce')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/habanero_experiments/openml-defaults')
    return parser.parse_args()


def run(args):
    solver_dir = args.model_name
    dataset_dir = os.path.basename(args.dataset_path)
    setup_dir = openmldefaults.utils.get_setup_dirname(args.resized_grid_size, args.num_defaults)
    configuration_dir = os.path.join(args.output_dir, dataset_dir, solver_dir, setup_dir)
    if not os.path.isdir(configuration_dir):
        raise ValueError('Directory does not exists: %s' % configuration_dir)
    correct_holdout_task_dir = None
    for holdout_tasks_dir in os.listdir(configuration_dir):
        if not os.path.isdir(os.path.join(configuration_dir, holdout_tasks_dir)):
            continue
        try:
            holdout_tasks = json.loads(holdout_tasks_dir)
        except json.decoder.JSONDecodeError:
            continue
        if args.task_idx in holdout_tasks:
            correct_holdout_task_dir = holdout_tasks_dir
    if correct_holdout_task_dir is None:
        raise ValueError('Could not find holdout task dir for task: %d' % args.task_idx)
    experiment_dir = os.path.join(configuration_dir, correct_holdout_task_dir)
    experiment_file = os.path.join(experiment_dir, 'generated_defaults.pkl')
    if not os.path.isfile(experiment_file):
        raise ValueError('Defaults file not generated yet: %s' %experiment_file)
    print(openmldefaults.utils.get_time(), 'Experiments file: %s' % experiment_file)

    with open(experiment_file, 'rb') as fp:
        generated_defaults = pickle.load(fp)['defaults']

    config_space = getattr(openmldefaults.config_spaces, 'get_%s_default_search_space' % args.config_space)()
    param_grid = openmldefaults.search.config_space_to_dist(config_space)
    print(param_grid)

    pass


if __name__ == '__main__':
    run(parse_args())
