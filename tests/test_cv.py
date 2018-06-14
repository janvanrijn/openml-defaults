import openmldefaults
import unittest


class TestCvFunctions(unittest.TestCase):

    def test_get_cv_indices(self):
        cases = [
            (23, 5),
            (10, 1),
            (100, 10),
            (13, 2),
            (13, 3),
            (13, 27)
        ]

        for num_tasks, num_iterations in cases:
            all_holdouts = []
            for i in range(num_iterations):
                current_holdout = openmldefaults.utils.get_cv_indices(num_tasks, num_iterations, i)
                all_holdouts.extend(current_holdout)
            assert sorted(all_holdouts) == list(range(num_tasks))

