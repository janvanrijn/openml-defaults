import openmldefaults
import pandas as pd
import unittest


class TestMiscFunctions(unittest.TestCase):

    def test_get_posterior(self):
        def dominates(dominater, dominated):
            return sum([dominater[x] >= dominated[x] for x in range(len(dominater))]) == len(dominater)

        frame = pd.DataFrame(data=[[1, 1, 1], [1, 2, 3], [3, 2, 1], [4, 1, 1]], dtype=int)
        print(frame)
        pareto_points, dominated_points = openmldefaults.utils.simple_cull(frame, dominates)
        reconstructed = pareto_points.append(dominated_points).sort_index().astype(int)
        pd.testing.assert_frame_equal(frame, reconstructed)
