import ConfigSpace
import logging
import numpy as np
import openmldefaults
import pandas as pd
import unittest


class TestStrategies(unittest.TestCase):

    def setUp(self):
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        self.cs = ConfigSpace.ConfigurationSpace('sklearn.ensemble.GradientBoostingClassifier', 0)
        self.cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(name='algorithm', choices=['a1', 'a2', 'a3']))

    @staticmethod
    def get_simple_dataset():
        data = [['a0', 1, 1, 1, 3], ['a1', 2, 2, 2, 2], ['a2', 3, 3, 3, 1]]
        columns = ['algorithm', 't0', 't1', 't2', 't3']
        df = pd.DataFrame(data=data, columns=columns).set_index('algorithm')
        expected_results = {
            'average_rank': {
                True: [0, 1, 2],
                False: [2, 1, 0],
            },
            'greedy': {
                True: [0, 2],
                False: [2, 0],
            },
            'active_testing': {
                True: [0, 2, 1],
                False: [2, 0, 1],
            },
        }
        return df, expected_results

    def test_models_on_simple_dataset(self):
        models = [
            openmldefaults.models.AverageRankDefaults(),
            openmldefaults.models.GreedyDefaults(),
            openmldefaults.models.ActiveTestingDefaults(),
        ]

        df, expected_results = TestStrategies.get_simple_dataset()

        for model in models:
            for minimize in {False, True}:
                logging.info('Testing strategy %s with minimize=%s' % (model.name, minimize))
                indices, _ = model.generate_defaults_discretized(df, 3, minimize, np.sum, self.cs, False)
                self.assertListEqual(indices, expected_results[model.name][minimize])

    def test_average_rank_on_text_classification(self):
        ground_truth = pd.read_csv('../data/text_classification_ar.csv')
        del ground_truth['rank']
        del ground_truth['workflow']

        classifier_names = ['text_classification']
        random_seed = 0
        search_space_identifier = 'ferreira'
        metadata_files = ['../data/text_classification.arff']
        measures = ['predictive_accuracy', 'runtime']
        task_id_column = 'dataset'
        skip_row_check = True

        config_space = openmldefaults.config_spaces.get_config_spaces(classifier_names,
                                                                      random_seed,
                                                                      search_space_identifier)
        metadata_frame = openmldefaults.utils.metadata_files_to_frame(metadata_files,
                                                                      search_space_identifier,
                                                                      measures,
                                                                      task_id_column,
                                                                      skip_row_check)
        for hyperparameter in ground_truth.columns.values:
            rename_dict = {hyperparameter: 'text_classification:' + hyperparameter}
            ground_truth = ground_truth.rename(columns=rename_dict)
        ground_truth['classifier'] = 'text_classification'
        tasks_tr = list(metadata_frame[task_id_column].unique())
        frame_tr = openmldefaults.utils.generate_dataset_using_metadata(
            metadata_frame=metadata_frame,
            task_ids=tasks_tr,
            hyperparameter_names=config_space.get_hyperparameter_names(),
            measure=measures[0],
            task_id_column=task_id_column,
            scaler_type=None,
            column_prefix=None,
        )

        model = openmldefaults.models.AverageRankDefaults()

        # accuracy defaults
        indices, _ = model.generate_defaults_discretized(frame_tr, 384, False, sum, config_space, True)
        for idx, row in ground_truth.iterrows():
            config = {name: value for name, value in zip(frame_tr.index.names, frame_tr.index[indices[idx]])}
            self.assertDictEqual(row.to_dict(), config)
