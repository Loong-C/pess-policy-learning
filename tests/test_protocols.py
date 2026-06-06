import unittest

import numpy as np

from algs.pess import compute_variance_terms
from experiments.reproduce import FULL_CONTEXTUAL_T_VALUES
from utils.compute import apply_floor
from utils.dgp import MultiLinear, MultiQuad, PublishedMultiLinear, PublishedMultiQuad
from utils.thompson import LinTS


class ProtocolTests(unittest.TestCase):
    def test_published_contextual_sample_size_grid(self):
        self.assertEqual(FULL_CONTEXTUAL_T_VALUES, [500, 1000, 2000, 5000])

    def test_paper_and_published_quad_formulas_are_distinct(self):
        xs = np.array([[2.0, 0.0], [0.0, 0.0]])
        paper = MultiQuad(2, 3, sigma=0).mean(xs)
        published = PublishedMultiQuad(2, 3, sigma=0).mean(xs)
        np.testing.assert_allclose(paper[:, 0], [-0.5, 1.5])
        np.testing.assert_allclose(published[:, 0], [-2.0, 2.0])

    def test_paper_and_published_linear_formulas_are_distinct(self):
        xs = np.array([[1.0, -1.0]])
        paper = MultiLinear(2, 3, sigma=0).mean(xs)
        published = PublishedMultiLinear(2, 3, sigma=0).mean(xs)
        np.testing.assert_allclose(paper[0], [3.0, 2.5, 2.0])
        np.testing.assert_allclose(published[0], [0.0, 1.0, 2.0])

    def test_displayed_algorithm_variance_terms(self):
        propensities = np.array([[0.25, 0.75], [0.5, 0.5]])
        observed = np.array([0, 1])
        policy = np.array([0, 0])
        result = compute_variance_terms(observed, policy, propensities)
        self.assertAlmostEqual(result["Vs"], 2.0)
        self.assertAlmostEqual(result["Vp"], np.sqrt(6.0) / 2.0)

    def test_probability_floor_projection(self):
        result = apply_floor(np.array([0.99, 0.01]), np.array([0.1, 0.1]))
        self.assertAlmostEqual(float(result.sum()), 1.0)
        self.assertTrue(np.all(result >= 0.1))

    def test_published_floor_uses_current_time(self):
        agent = LinTS(2, 1)
        agent.set_floor(if_floor=True, floor_start=0.5, floor_decay=0.5)
        np.testing.assert_allclose(agent._floor_vector(100), [0.05, 0.05])


if __name__ == "__main__":
    unittest.main()
