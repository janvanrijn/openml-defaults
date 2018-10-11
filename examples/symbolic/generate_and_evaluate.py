import argparse
import ConfigSpace
import copy
import logging
import numpy as np
import openml
import openmlcontrib
import openmldefaults
import os
import pandas as pd
import pickle
import sklearn
import typing


def parse_args():
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--cache_directory', type=str, default=os.path.expanduser('~') + '/experiments/openml_cache',
                        help='directory to store cache')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/symbolic_defaults/')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--classifier', type=str, default='libsvm_svc', help='scikit-learn flow name')
    parser.add_argument('--config_space', type=str, default='micro', help='config space type')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--num_runs', type=int, default=200, help='max runs to obtain from openml')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    return parser.parse_args()


def inverse_transform_fn(param_value: float, meta_feature_value: float) -> float:
    # raising is ok
    if meta_feature_value == 0.0:
        raise ZeroDivisionError()
    result = param_value / meta_feature_value
    if np.isinf(result):
        raise OverflowError()
    return result


def power_transform_fn(param_value: float, meta_feature_value: float) -> float:
    return param_value ** meta_feature_value


def multiply_transform_fn(param_value: float, meta_feature_value: float) -> float:
    return param_value * meta_feature_value


def select_best_configuration_across_tasks(config_frame: pd.DataFrame,
                                           surrogates: typing.Dict[int, sklearn.pipeline.Pipeline],
                                           surrogate_train_cols: np.array,
                                           symbolic_hyperparameter: typing.Optional[str],
                                           symbolic_fn: typing.Optional[typing.Callable],
                                           symbolic_alpha_value: typing.Optional[float],
                                           symbolic_mf_values: typing.Optional[typing.Dict[int, float]]) \
        -> typing.Tuple[pd.Series, np.array]:
    num_configs = config_frame.shape[0]
    num_tasks = len(surrogates)
    results = np.zeros((num_tasks, num_configs), dtype=np.float)
    for idx, (task_id, task_surrogate) in enumerate(surrogates.items()):
        transformed_frame = pd.DataFrame(config_frame)
        if symbolic_hyperparameter is not None:
            current_value = symbolic_fn(symbolic_alpha_value, symbolic_mf_values[task_id])
            transformed_frame[symbolic_hyperparameter] = current_value
        transformed_frame.sort_index(axis=1, inplace=True)
        if not np.array_equal(transformed_frame.columns.values, surrogate_train_cols):
            raise ValueError('Column set not equal: %s vs %s' % (transformed_frame.columns.values,
                                                                 surrogate_train_cols))
        results[idx] = task_surrogate.predict(pd.get_dummies(transformed_frame).as_matrix())
    average_measure_per_configuration = np.average(results, axis=0)
    best_idx = np.argmax(average_measure_per_configuration)
    best_config = config_frame.iloc[best_idx]
    best_results = results[:, best_idx]
    return best_config.to_dict(), best_results


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    study = openml.study.get_study(args.study_id, 'tasks')
    study.tasks = study.tasks
    if 34536 in study.tasks:
        study.tasks.remove(34536)

    config_space_fn = getattr(openmldefaults.config_spaces,
                              'get_%s_%s_search_space' % (args.classifier,
                                                          args.config_space))
    config_space = config_space_fn()
    configurations = openmldefaults.utils.generate_grid_configurations(config_space, 0, args.resized_grid_size)

    config_frame_orig = pd.DataFrame(configurations)
    config_frame_orig.sort_index(axis=1, inplace=True)
    quality_frame = openmlcontrib.meta.get_tasks_qualities_as_dataframe(study.tasks, False, -1, True)

    surrogates = dict()
    for idx, task_id in enumerate(study.tasks):
        logging.info('Training surrogate on Task %d (%d/%d)' % (task_id, idx + 1, len(study.tasks)))
        estimator, columns = openmldefaults.utils.train_surrogate_on_task(
            task_id, config_space.meta['flow_id'], args.num_runs, config_space, args.scoring, args.cache_directory)
        if not np.array_equal(config_frame_orig.columns.values, columns):
            # if this goes wrong, it is due to the pd.get_dummies() fn
            raise ValueError('Column set not equal: %s vs %s' % (config_frame_orig.columns.values, columns))
        surrogates[task_id] = estimator

    # performance untransformed
    best_config_vanilla, best_results_vanilla = select_best_configuration_across_tasks(
        config_frame_orig, surrogates, config_frame_orig.columns.values, None, None, None, None)
    best_avg_vanilla = np.average(best_results_vanilla)
    logging.info('Baseline: %s [%s] %s' % (best_config_vanilla, best_results_vanilla, best_avg_vanilla))

    transform_fns = {
        'inverse_transform_fn': inverse_transform_fn,
        'power_transform_fn': power_transform_fn,
        'multiply_transform_fn': multiply_transform_fn
    }

    results = list()
    for hyperparameter in config_space.get_hyperparameters():
        if not isinstance(hyperparameter, ConfigSpace.hyperparameters.NumericalHyperparameter):
            continue
        logging.info('Started with hyperparameter %s' % hyperparameter.name)
        config_space_prime = openmldefaults.config_spaces.remove_hyperparameter(config_space, hyperparameter.name)
        configurations = openmldefaults.utils.generate_grid_configurations(config_space_prime, 0, args.resized_grid_size)
        config_frame_prime = pd.DataFrame(configurations)
        for transform_name, transform_fn in transform_fns.items():
            logging.info('- Transformer fn %s' % transform_name)
            for alpha_value in np.geomspace(0.01, 2, 10):
                logging.info('--- Alpha value %f' % alpha_value)
                for meta_feature in quality_frame.columns.values:
                    try:
                        best_config_current, best_results_current = select_best_configuration_across_tasks(
                            config_frame_prime,
                            surrogates,
                            config_frame_orig.columns.values,  # note to take the original frame
                            hyperparameter.name,
                            transform_fn,
                            alpha_value,
                            quality_frame[meta_feature].to_dict(),
                        )
                        best_avg_current = np.average(best_results_current)
                        if best_avg_current > best_avg_vanilla:
                            current_result = {
                                'config': best_config_current,
                                'results': best_results_current,
                                'avg': best_avg_current,
                                'hyperparameter': hyperparameter.name,
                                'transform_fn': transform_name,
                                'alpha_value': alpha_value,
                                'meta_feature': meta_feature
                            }
                            results.append(current_result)
                            logging.info('Found improvement over base-line: %s' % current_result)
                    except ZeroDivisionError:
                        logging.warning('Zero division error with (fn=%s, alpha=%s, meta_f=%s). '
                                        'skipping. ' % (transform_name, alpha_value, meta_feature))
                        pass
                    except OverflowError:
                        logging.warning('Overflow error with (fn=%s, alpha=%s, meta_f=%s). '
                                        'skipping. ' % (transform_name, alpha_value, meta_feature))
                        pass
                    except ValueError:
                        # keep a close eye on this one. Question: why do the others not catch this one?
                        logging.warning('Overflow error with (fn=%s, alpha=%s, meta_f=%s). '
                                        'skipping. ' % (transform_name, alpha_value, meta_feature))
                        pass
    total = {
        'baseline_configuration': best_config_vanilla,
        'baseline_avg_performance': best_avg_vanilla,
        'outperforming': results
    }
    os.makedirs(args.output_directory, exist_ok=True)
    with open(os.path.join(args.output_directory, 'results.pkl'), 'wb') as fp:
        pickle.dump(obj=total, file=fp, protocol=0)


if __name__ == '__main__':
    run(parse_args())
