import argparse
import ConfigSpace
import json
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
    parser.add_argument('--task_idx', type=int, default=None)
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


# def sigmoid_transform_fn(param_value: float, meta_feature_value: float) -> float:
#     return 1 / (1 + np.e ** (-1 * meta_feature_value))


def log_transform_fn(param_value: float, meta_feature_value: float) -> float:
    return param_value * np.log(meta_feature_value)


def root_transform_fn(param_value: float, meta_feature_value: float) -> float:
    return param_value * np.sqrt(meta_feature_value)


def single_prediction(df: pd.DataFrame,
                      surrogate: sklearn.pipeline.Pipeline,
                      config: typing.Dict) -> float:
    # TODO: might break with categoricals
    df = pd.DataFrame(columns=df.columns.values)
    df = df.append(config, ignore_index=True) # TODO: ignore true ?
    return surrogate.predict(pd.get_dummies(df).values)[0]


def select_best_configuration_across_tasks(config_frame: pd.DataFrame,
                                           surrogates: typing.Dict[int, sklearn.pipeline.Pipeline],
                                           surrogate_train_cols: np.array,
                                           symbolic_hyperparameter: typing.Optional[str],
                                           symbolic_fn: typing.Optional[typing.Callable],
                                           symbolic_alpha_value: typing.Optional[float],
                                           symbolic_mf_values: typing.Optional[typing.Dict[int, float]]) \
        -> typing.Tuple[typing.Dict, np.array]:
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
        results[idx] = task_surrogate.predict(pd.get_dummies(transformed_frame).values)
    average_measure_per_configuration = np.average(results, axis=0)
    best_idx = np.argmax(average_measure_per_configuration)
    best_config = config_frame.iloc[best_idx]
    best_results = results[:, best_idx]
    return best_config.to_dict(), best_results


def run_on_tasks(config_frame_orig: pd.DataFrame,
                 surrogates: typing.Dict[int, sklearn.pipeline.Pipeline],
                 quality_frame: pd.DataFrame,
                 config_space: ConfigSpace.ConfigurationSpace,
                 hold_out_task: typing.Optional[int],
                 resized_grid_size: int,
                 output_file: str):
    hold_out_surrogate = None
    if hold_out_task is not None:
        hold_out_surrogate = surrogates[hold_out_task]
        surrogates = dict(surrogates)
        del surrogates[hold_out_task]

    # performance untransformed
    baseline_configuration, baseline_results_per_task = select_best_configuration_across_tasks(
        config_frame_orig, surrogates, config_frame_orig.columns.values, None, None, None, None)
    baseline_avg_performance = np.average(baseline_results_per_task)
    baseline_holdout = None
    baseline_random_search = None
    if hold_out_task is not None:
        baseline_holdout = single_prediction(config_frame_orig,
                                             hold_out_surrogate,
                                             baseline_configuration)
        baseline_random_search = [single_prediction(config_frame_orig,
                                                    hold_out_surrogate,
                                                    config_space.sample_configuration(1).get_dictionary()) for i in range(50)]
    logging.info('Baseline: %s [%s] %s. Holdout task: %s' % (baseline_configuration,
                                                             baseline_results_per_task,
                                                             baseline_avg_performance,
                                                             baseline_holdout))

    transform_fns = {
        'inverse_transform_fn': inverse_transform_fn,
        'power_transform_fn': power_transform_fn,
        'multiply_transform_fn': multiply_transform_fn,
        'log_transform_fn': log_transform_fn,
        'root_transform_fn': root_transform_fn
    }

    symbolic_defaults = list()
    for hyperparameter in config_space.get_hyperparameters():
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant):
            logging.info('Skipping Constant Hyperparameter: %s' % hyperparameter.name)
            continue
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter):
            logging.info('Skipping Unparameterized Hyperparameter: %s' % hyperparameter.name)
            continue
        if not isinstance(hyperparameter, ConfigSpace.hyperparameters.NumericalHyperparameter):
            logging.info('Skipping Non-Numerical Hyperparameter: %s' % hyperparameter.name)
            continue
        logging.info('Started with hyperparameter %s' % hyperparameter.name)
        config_space_prime = openmldefaults.config_spaces.remove_hyperparameter(config_space, hyperparameter.name)
        configurations = openmldefaults.utils.generate_grid_configurations(config_space_prime, 0,
                                                                           resized_grid_size)
        config_frame_prime = pd.DataFrame(configurations)
        for transform_name, transform_fn in transform_fns.items():
            logging.info('- Transformer fn %s' % transform_name)
            geom_space = np.geomspace(0.01, 2, 10)
            geom_space = np.append(geom_space, [1])
            for alpha_value in geom_space:
                logging.info('--- Alpha value %f' % alpha_value)
                for meta_feature in quality_frame.columns.values:
                    try:
                        symbolic_config, symbolic_results_per_task = select_best_configuration_across_tasks(
                            config_frame_prime,
                            surrogates,
                            config_frame_orig.columns.values,  # note to take the original frame
                            hyperparameter.name,
                            transform_fn,
                            alpha_value,
                            quality_frame[meta_feature].to_dict(),
                        )
                        symbolic_average_performance = np.average(symbolic_results_per_task)
                        if symbolic_average_performance > baseline_avg_performance:
                            symbolic_holdout_score = None
                            if hold_out_surrogate is not None:
                                symbolic_value = transform_fn(alpha_value, quality_frame[meta_feature][hold_out_task])
                                symbolic_config[hyperparameter.name] = symbolic_value
                                symbolic_holdout_score = single_prediction(config_frame_orig,
                                                                           hold_out_surrogate,
                                                                           symbolic_config)
                            current_result = {
                                'configuration': symbolic_config,
                                'results_per_task': symbolic_results_per_task,
                                'avg_performance': symbolic_average_performance,
                                'holdout_score': symbolic_holdout_score,
                                'trasnform_hyperparameter': hyperparameter.name,
                                'transform_fn': transform_name,
                                'transform_alpha_value': alpha_value,
                                'transform_meta_feature': meta_feature,
                            }
                            symbolic_defaults.append(current_result)
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
        'baseline_configuration': baseline_configuration,
        'baseline_avg_performance': baseline_avg_performance,
        'baseline_random_search': baseline_random_search,
        'baseline_results_per_task': baseline_results_per_task,
        'baseline_holdout_score': baseline_holdout,
        'symbolic_defaults': symbolic_defaults
    }
    with open(output_file, 'wb') as fp:
        pickle.dump(obj=total, file=fp, protocol=0)


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
        setup_frame = openmlcontrib.meta.get_task_flow_results_as_dataframe(task_id=task_id,
                                                                            flow_id=config_space.meta['flow_id'],
                                                                            num_runs=args.num_runs,
                                                                            raise_few_runs=False,
                                                                            configuration_space=config_space,
                                                                            evaluation_measures=[args.scoring],
                                                                            cache_directory=args.cache_directory)

        estimator, columns = openmldefaults.utils.train_surrogate_on_task(
            task_id, config_space, setup_frame, args.scoring)
        if not np.array_equal(config_frame_orig.columns.values, columns):
            # if this goes wrong, it is due to the pd.get_dummies() fn
            raise ValueError('Column set not equal: %s vs %s' % (config_frame_orig.columns.values, columns))
        surrogates[task_id] = estimator

    os.makedirs(args.output_directory, exist_ok=True)
    output_file = os.path.join(args.output_directory, args.classifier, 'results_all.pkl')
    if args.task_idx is None:
        logging.info('Evaluating on train tasks')
        run_on_tasks(config_frame_orig=config_frame_orig,
                     surrogates=surrogates,
                     quality_frame=quality_frame,
                     config_space=config_space,
                     resized_grid_size=args.resized_grid_size,
                     hold_out_task=None,
                     output_file=output_file)
    else:
        task_id = study.tasks[args.task_idx]
        logging.info('Evaluating on holdout task %d (%d/%d)' %
                     (task_id, args.task_idx + 1, len(study.tasks)))
        output_file = os.path.join(args.output_directory, args.classifier, 'results_%d.pkl' % task_id)
        run_on_tasks(config_frame_orig=config_frame_orig,
                     surrogates=surrogates,
                     quality_frame=quality_frame,
                     config_space=config_space,
                     resized_grid_size=args.resized_grid_size,
                     hold_out_task=task_id,
                     output_file=output_file)


if __name__ == '__main__':
    run(parse_args())
