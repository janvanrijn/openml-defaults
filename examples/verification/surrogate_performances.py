%load_ext autoreload
%autoreload
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
import pdb
import sys
import logging
from sklearn.metrics import explained_variance_score, r2_score, mean_squared_error
from scipy.stats import spearmanr

if __name__ == '__main__':
    datasets = {
        'svc' : {
            'name'     : 'svc',
            'metadata' : os.path.expanduser('~/Documents/projects/sklearn-bot/data/svc.arff'),
            'scoring'  : 'predictive_accuracy',
            'tid_col'  : 'task_id'
        },
        'classif_svm' : {
            'name'     : 'classif_svm',
            'metadata' : os.path.expanduser('~/Documents/projects/openml-defaults/data/classif_svm.arff'),
            'scoring'  : 'predictive_accuracy',
            'tid_col'  : 'task_id'
        },
        'classif_xgboost' : {
            'name'     : "classif_xgboost",
            'metadata' : os.path.expanduser('~/Documents/projects/openml-defaults/data/classif_xgboost.arff'),
            'scoring'  : 'predictive_accuracy',
            'tid_col'  : 'task_id'
        }
    }

    algorithm = datasets['classif_svm']
    config_space = openmldefaults.config_spaces.get_config_spaces([algorithm['name']],
                                                                  42,
                                                                  None)
    configurations = openmldefaults.utils.generate_grid_configurations(config_space, 0, 20)
    config_frame_orig = pd.DataFrame(configurations)
    config_frame_orig.sort_index(axis=1, inplace=True)
    # --- Metadata ---
    metadata_performance_frame = openmldefaults.utils.metadata_files_to_frame([algorithm['metadata']],
                                                                              None,
                                                                              [algorithm['scoring']],
                                                                              task_id_column=algorithm['tid_col'],
                                                                              skip_row_check=True)
    all_task_ids = set(metadata_performance_frame[algorithm['tid_col']].unique())
    result = pd.DataFrame()
    for idx, task_id in enumerate(all_task_ids):
        logging.info('Training surrogate on Task %d (%d/%d)' % (task_id, idx + 1, len(all_task_ids)))

        setup_frame = pd.DataFrame(metadata_performance_frame.loc[metadata_performance_frame[algorithm['tid_col']] == task_id])
        del setup_frame['task_id']
        logging.info('obtained meta-data from arff file. Dimensions: %s' % str(setup_frame.shape))
        if len(getattr(setup_frame, 'predictive_accuracy').unique()) == 1:
            logging.warning('Not enough unique performance measures for task %d. Skipping' % task_id)
            continue
        if setup_frame.shape[0] == 0:
            logging.warning('No results for task %d. Skipping' % task_id)
            continue
        y_true = setup_frame.predictive_accuracy
        estimator, columns = openmldefaults.utils.train_surrogate_on_task(  task_id,
                                                                            config_space.get_hyperparameter_names(),
                                                                            setup_frame.sample(frac = 0.5),
                                                                            algorithm['scoring'],
                                                                            normalize=False,
                                                                            n_estimators=20,
                                                                            random_seed=42)
        if not np.array_equal(config_frame_orig.columns.values, columns):
            # if this goes wrong, it is due to the pd.get_dummies() fn
            raise ValueError('Column set not equal: %s vs %s' % (config_frame_orig.columns.values, columns))

        y_pred = np.array(estimator.predict(setup_frame))

        result = result.append(pd.DataFrame(data = {
            'algo'    : algorithm['name'],
            'task_id' : task_id,
            'mse'     : mean_squared_error(y_true, y_pred),
            'exp_var' : explained_variance_score(y_true, y_pred),
            'r2'      : r2_score(y_true, y_pred),
            'spearm'  : spearmanr(y_true, y_pred)[0]
        }, index = [0]), ignore_index=True)


