import openmldefaults
import pandas as pd
import unittest


class TestParetoFunctions(unittest.TestCase):

    def test_simple_cull(self):
        frame = pd.DataFrame(data=[[0.0, 1.0, 1.0, 1.0],
                                   [1.0, 1.0, 2.0, 3.0],
                                   [2.0, 3.0, 2.0, 1.0],
                                   [3.0, 4.0, 1.0, 1.0]],
                             columns=['idx', 'task 1', 'task 2', 'task 3'],
                             dtype=float).set_index('idx')

        pareto_points, dominated_points = openmldefaults.utils.simple_cull(frame, openmldefaults.utils.dominates)
        reconstructed = pareto_points.append(dominated_points).sort_index().astype(float)
        pd.testing.assert_frame_equal(frame, reconstructed)

    def test_simple_cull_multi_level_index(self):
        frame = pd.DataFrame(data=[[0.0, 0.0, 1.0, 1.0, 1.0],
                                   [0.0, 1.0, 1.0, 2.0, 3.0],
                                   [1.0, 0.0, 3.0, 2.0, 1.0],
                                   [1.0, 1.0, 4.0, 1.0, 1.0]],
                             columns=['idx0', 'idx1', 'task 1', 'task 2', 'task 3'],
                             dtype=float).set_index(['idx0', 'idx1'])

        pareto_points, dominated_points = openmldefaults.utils.simple_cull(frame, openmldefaults.utils.dominates)
        reconstructed = pareto_points.append(dominated_points).sort_index().astype(float)
        pd.testing.assert_frame_equal(frame, reconstructed)
