import openmldefaults
import sklearn.datasets
import sklearn.tree
import unittest


class TestCvFunctions(unittest.TestCase):

    def test_get_cv_indices(self):
        X, y = sklearn.datasets.load_iris(True)
        estimator = sklearn.tree.DecisionTreeClassifier()

        defaults = [
            {'max_depth': 2, 'min_samples_leaf': 1},
            {'max_depth': 4, 'min_samples_leaf': 1},
            {'max_depth': 6, 'min_samples_leaf': 1},
            {'max_depth': 8, 'min_samples_leaf': 1},
            {'max_depth': 10, 'min_samples_leaf': 1},
        ]

        search = openmldefaults.search.DefaultSearchCV(estimator, defaults)
        search.fit(X, y)

        self.assertListEqual(list(search.cv_results_['params']), defaults)
