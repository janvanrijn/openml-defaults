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
                False: [2, 1, 0]
            },
            'greedy': {
                True: [0, 2],
                False: [2, 0]
            }
        }
        return df, expected_results

    def test_models_on_simple_dataset(self):
        models = [openmldefaults.models.AverageRankDefaults(), openmldefaults.models.GreedyDefaults()]

        df, expected_results = TestStrategies.get_simple_dataset()

        for model in models:
            for minimize in {False, True}:
                logging.info('Testing strategy %s with minimize=%s' % (model.name, minimize))
                res = model.generate_defaults_discretized(df, 3, minimize, np.sum, self.cs, False)
                self.assertListEqual(list(res['indices']), expected_results[model.name][minimize])

