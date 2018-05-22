import openmldefaults
import pandas as pd
import unittest


class TestMiscFunctions(unittest.TestCase):

    def test_simple_cull(self):
        def dominates(dominater, dominated):
            return sum([dominater[x] >= dominated[x] for x in range(len(dominater))]) == len(dominater)

        frame = pd.DataFrame(data=[[1, 1, 1], [1, 2, 3], [3, 2, 1], [4, 1, 1]],
                             columns=['task 1', 'task 2', 'task 3'],
                             dtype=int)

        pareto_points, dominated_points = openmldefaults.utils.simple_cull(frame, dominates)
        reconstructed = pareto_points.append(dominated_points).sort_index().astype(int)
        pd.testing.assert_frame_equal(frame, reconstructed)

    def test_simple_cull_multi_level_index(self):
        def dominates(dominater, dominated):
            return sum([dominater[x] >= dominated[x] for x in range(len(dominater))]) == len(dominater)

        frame = pd.DataFrame(data=[[0, 0, 1, 1, 1], [0, 1, 1, 2, 3], [1, 0, 3, 2, 1], [1, 1, 4, 1, 1]],
                             columns=['idx0', 'idx1', 'task 1', 'task 2', 'task 3'],
                             dtype=int).set_index(['idx0', 'idx1'])

        pareto_points, dominated_points = openmldefaults.utils.simple_cull(frame, dominates)
        reconstructed = pareto_points.append(dominated_points).sort_index().astype(int)
        pd.testing.assert_frame_equal(frame, reconstructed)