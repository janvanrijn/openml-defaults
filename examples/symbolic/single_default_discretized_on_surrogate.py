import arff
import argparse
import ConfigSpace
import logging
import numpy as np
import openmlcontrib
import openmldefaults
import os
import pandas as pd
import pickle
import sklearn
import typing


def parse_args():
    metadata_svc = os.path.expanduser('~/projects/sklearn-bot/data/svc.arff')
    metadata_qualities = os.path.expanduser('~/projects/openml-python-contrib/data/metafeatures_openml100.arff')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/symbolic_defaults/')
    parser.add_argument('--task_idx', type=int, default=None)
    parser.add_argument('--metadata_performance_file', type=str, default=metadata_svc)
    parser.add_argument('--metadata_qualities_file', type=str, default=metadata_qualities)
    parser.add_argument('--search_qualities', type=str, nargs='+')
    parser.add_argument('--search_hyperparameters', type=str, nargs='+')
    parser.add_argument('--search_transform_fns', type=str, nargs='+')
    parser.add_argument('--classifier_name', type=str, default='svc', help='scikit-learn flow name')
    parser.add_argument('--search_space_identifier', type=str, default=None)
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--n_estimators', type=int, default=64)
    parser.add_argument('--task_id_column', default='task_id', type=str)
    parser.add_argument('--skip_row_check', action='store_true')
    return parser.parse_args()


def select_best_configuration_across_tasks(config_frame: pd.DataFrame,
                                           surrogates: typing.Dict[int, sklearn.pipeline.Pipeline],
                                           surrogate_train_cols: np.array,
                                           symbolic_hyperparameter: typing.Optional[str],
                                           symbolic_fn: typing.Optional[typing.Callable],
                                           symbolic_alpha_value: typing.Optional[float],
                                           symbolic_mf_values: typing.Optional[typing.Dict[int, float]]) \
        -> typing.Tuple[typing.Dict, np.array]:
    # TODO: normalize predictions
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
        results[idx] = task_surrogate.predict(transformed_frame.values)
    average_measure_per_configuration = np.average(results, axis=0)
    best_idx = np.argmax(average_measure_per_configuration)
    best_config = config_frame.iloc[best_idx]
    best_results = results[:, best_idx]
    return best_config.to_dict(), best_results


def run_on_tasks(config_frame_orig: pd.DataFrame,
                 surrogates: typing.Dict[int, sklearn.pipeline.Pipeline],
                 quality_frame: pd.DataFrame,
                 config_space: ConfigSpace.ConfigurationSpace,
                 search_hyperparameters: typing.List[str],
                 search_transform_fns: typing.List[str],
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
        baseline_holdout = openmldefaults.utils.single_prediction(config_frame_orig,
                                                                  hold_out_surrogate,
                                                                  baseline_configuration)
        baseline_random_search = [
            openmldefaults.utils.single_prediction(config_frame_orig,
                                                   hold_out_surrogate,
                                                   config_space.sample_configuration(1).get_dictionary())
            for i in range(50)
        ]
    logging.info('Baseline: %s [%s] %s. Holdout task: %s' % (baseline_configuration,
                                                             baseline_results_per_task,
                                                             baseline_avg_performance,
                                                             baseline_holdout))

    transform_fns = openmldefaults.symbolic.all_transform_fns()
    symbolic_defaults = list()
    search_hyperparameters = search_hyperparameters if search_hyperparameters is not None \
        else [hp.name for hp in config_space.get_hyperparameters()]
    for idx_hp, hyperparameter_name in enumerate(search_hyperparameters):
        hyperparameter = config_space.get_hyperparameter(hyperparameter_name)
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.Constant):
            logging.warning('Skipping Constant Hyperparameter: %s' % hyperparameter.name)
            continue
        if isinstance(hyperparameter, ConfigSpace.hyperparameters.UnParametrizedHyperparameter):
            logging.warning('Skipping Unparameterized Hyperparameter: %s' % hyperparameter.name)
            continue
        if not isinstance(hyperparameter, ConfigSpace.hyperparameters.NumericalHyperparameter):
            logging.warning('Skipping Non-Numerical Hyperparameter: %s' % hyperparameter.name)
            continue
        logging.info('Started with hyperparameter %s (%d/%d)' % (hyperparameter.name,
                                                                 idx_hp + 1,
                                                                 len(search_hyperparameters)))
        config_space_prime = openmldefaults.utils.remove_hyperparameter(config_space, hyperparameter.name)
        configurations = openmldefaults.utils.generate_grid_configurations(config_space_prime, 0,
                                                                           resized_grid_size)
        config_frame_prime = pd.DataFrame(configurations)
        for idx_trnfm_fn, transform_name in enumerate(search_transform_fns):
            logging.info('- Transformer fn %s (%d/%d)' % (transform_name, idx_trnfm_fn + 1, len(transform_fns)))
            geom_space = np.geomspace(0.01, 2, 10)
            geom_space = np.append(geom_space, [1])
            for idx_av, alpha_value in enumerate(geom_space):
                logging.info('--- Alpha value %f (%d/%d)' % (alpha_value, idx_av + 1, len(geom_space)))
                for meta_feature in quality_frame.columns.values:
                    try:
                        transform_fn = openmldefaults.symbolic.all_transform_fns()[transform_name]
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
                                symbolic_holdout_score = openmldefaults.utils.single_prediction(config_frame_orig,
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
    logging.info('Saved result file to: %s' % output_file)


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    config_space = openmldefaults.config_spaces.get_config_spaces([args.classifier_name],
                                                                  args.random_seed,
                                                                  args.search_space_identifier)
    configurations = openmldefaults.utils.generate_grid_configurations(config_space, 0, args.resized_grid_size)

    config_frame_orig = pd.DataFrame(configurations)
    config_frame_orig.sort_index(axis=1, inplace=True)

    with open(args.metadata_qualities_file, 'r') as fp:
        metadata_quality_frame = openmlcontrib.meta.arff_to_dataframe(arff.load(fp), None)
        metadata_quality_frame = metadata_quality_frame.set_index(['task_id'])
    if args.search_qualities is not None:
        if not isinstance(args.search_qualities, list):
            raise ValueError()
        metadata_quality_frame = metadata_quality_frame[args.search_qualities]

    metadata_atts = openmldefaults.utils.get_dataset_metadata(args.metadata_performance_file)
    if args.scoring not in metadata_atts['col_measures']:
        raise ValueError('Could not find measure: %s' % args.scoring)
    metadata_performance_frame = openmldefaults.utils.metadata_files_to_frame([args.metadata_performance_file],
                                                                              args.search_space_identifier,
                                                                              [args.scoring],
                                                                              task_id_column=args.task_id_column,
                                                                              skip_row_check=args.skip_row_check)
    all_task_ids = metadata_quality_frame.index.unique()
    all_task_ids_perf = set(metadata_performance_frame[args.task_id_column].unique())
    if all_task_ids_perf != set(all_task_ids):
        missing_quality = all_task_ids_perf - set(all_task_ids)
        missing_perform = set(all_task_ids) - all_task_ids_perf
        logging.warning('Task ids performance frame and quality frame do not align. Missing performance frame: %s, '
                        'Missing quality frame: %s' % (missing_perform, missing_quality))
        if len(missing_perform) > 0:
            raise ValueError('Missing values in Performance Frame, please resolve.')

    surrogates = dict()
    for idx, task_id in enumerate(all_task_ids):
        logging.info('Training surrogate on Task %d (%d/%d)' % (task_id, idx + 1, len(all_task_ids)))

        setup_frame = pd.DataFrame(metadata_performance_frame.loc[metadata_performance_frame['task_id'] == task_id])
        del setup_frame['task_id']
        logging.info('obtained meta-data from arff file. Dimensions: %s' % str(setup_frame.shape))

        if len(getattr(setup_frame, args.scoring).unique()) == 1:
            logging.warning('Not enough unique performance measures for task %d. Skipping' % task_id)
            continue
        if setup_frame.shape[0] == 0:
            logging.warning('No results for task %d. Skipping' % task_id)
            continue

        estimator, columns = openmldefaults.utils.train_surrogate_on_task(
            task_id,
            config_space,
            setup_frame,
            args.scoring,
            normalize=False,
            n_estimators=args.n_estimators,
            random_seed=args.random_seed)
        if not np.array_equal(config_frame_orig.columns.values, columns):
            # if this goes wrong, it is due to the pd.get_dummies() fn
            raise ValueError('Column set not equal: %s vs %s' % (config_frame_orig.columns.values, columns))
        surrogates[task_id] = estimator

    os.makedirs(os.path.join(args.output_directory, args.classifier_name), exist_ok=True)
    output_file = os.path.join(args.output_directory, args.classifier_name, 'results_all.pkl')
    if args.task_idx is None:
        logging.info('Evaluating on train tasks (%d)' % len(all_task_ids))
        run_on_tasks(config_frame_orig=config_frame_orig,
                     surrogates=surrogates,
                     quality_frame=metadata_quality_frame,
                     config_space=config_space,
                     search_hyperparameters=args.search_hyperparameters,
                     search_transform_fns=args.search_transform_fns,
                     resized_grid_size=args.resized_grid_size,
                     hold_out_task=None,
                     output_file=output_file)
    else:
        task_id = all_task_ids[args.task_idx]
        logging.info('Evaluating on holdout task %d (%d/%d)' %
                     (task_id, args.task_idx + 1, len(all_task_ids)))
        output_file = os.path.join(args.output_directory, args.classifier_name, 'results_%d.pkl' % task_id)
        run_on_tasks(config_frame_orig=config_frame_orig,
                     surrogates=surrogates,
                     quality_frame=metadata_quality_frame,
                     config_space=config_space,
                     search_hyperparameters=args.search_hyperparameters,
                     search_transform_fns=args.search_transform_fns,
                     resized_grid_size=args.resized_grid_size,
                     hold_out_task=task_id,
                     output_file=output_file)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
