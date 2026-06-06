import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from algs.pess import compute_variance_terms
from experiments.analyze_full_reproduction import (
    _cluster_mean_differences,
    _paired_tost,
    add_equivalence_statistics,
)
from experiments.reproduce import (
    FULL_CONTEXTUAL_T_VALUES,
    PAPER_SPEC_REAL_DATASETS,
    PUBLISHED_REAL_DATASETS,
    TREE_LINEAR_BETAS,
    TREE_PESS_BETAS,
    _contextual_noise_sigma,
    _real_grid,
    _run_chunked,
    _stable_seed,
    _cached_eval_data,
    _ts_eval_seed,
    _ts_rep_seed,
    eval_linear_pevi,
    eval_linear_pevi_grid,
    fit_linear_pevi,
)
from experiments.write_reproduction_report_zh import (
    _overall_verdict,
    _primary_rows,
)
from utils.compute import apply_floor
from utils.dgp import (
    MultiLinear,
    MultiQuad,
    PublishedMultiLinear,
    PublishedMultiQuad,
    generate_bandit_data,
)
from utils.experiment import run_experiment
from utils.thompson import LinTS


class ProtocolTests(unittest.TestCase):
    def test_published_contextual_sample_size_grid(self):
        self.assertEqual(FULL_CONTEXTUAL_T_VALUES, [500, 1000, 2000, 5000])

    def test_tree_beta_grids_match_figure_5(self):
        self.assertEqual(
            TREE_PESS_BETAS,
            [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 5, 10],
        )
        self.assertEqual(TREE_LINEAR_BETAS, [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 5, 10])

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

    def test_contextual_noise_matches_each_reproduction_target(self):
        self.assertEqual(_contextual_noise_sigma("published"), 1.0)
        self.assertEqual(_contextual_noise_sigma("paper-spec"), 0.1)

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

    def test_published_ridge_covariance_has_no_artificial_jitter(self):
        agent = LinTS(1, 1, ridge_mode="cv")
        xs = np.array([[-1.0], [0.0], [1.0], [2.0]])
        ys = np.array([0.0, 0.2, 0.8, 1.1])
        agent.update_TS(xs, np.zeros(len(xs), dtype=int), ys)
        design = np.column_stack([np.ones(len(xs)), xs])
        # Refit through RidgeCV to recover the selected alpha directly.
        from sklearn.linear_model import RidgeCV

        model = RidgeCV(
            alphas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            fit_intercept=True,
        ).fit(xs, ys)
        gram = design.T @ design + model.alpha_ * np.eye(2)
        expected = np.var(ys - model.predict(xs)) * np.linalg.inv(gram)
        np.testing.assert_allclose(agent.V[0], expected)

    def test_fixed_beta_and_cv_use_the_same_ts_stream(self):
        expected_eval = _stable_seed(
            7, "published", "ts_eval", 2, 1000, 10, "0.5", False
        )
        expected_rep = _stable_seed(
            7, "published", "ts", 2, 1000, 10, "0.5", False, 3
        )
        self.assertEqual(
            _ts_eval_seed(7, "published", 2, 1000, 10, "0.5"),
            expected_eval,
        )
        self.assertEqual(
            _ts_rep_seed(7, "published", 2, 1000, 10, "0.5", 3),
            expected_rep,
        )

    def test_collector_preserves_deterministic_initial_actions(self):
        xs = np.linspace(-1, 1, 200).reshape(100, 2)
        ys = np.zeros((100, 10))
        logged = run_experiment(
            xs,
            ys,
            batch_sizes=[100],
            num_mc=10,
            ridge_mode="cv",
        )
        np.testing.assert_array_equal(logged["ws"], np.arange(100) % 10)
        np.testing.assert_allclose(logged["ps"], 0.1)

    def test_equivalence_statistics_distinguish_match_and_difference(self):
        comparison = pd.DataFrame(
            {
                "ours_mean": [1.0, 1.2],
                "paper_mean": [1.0, 1.0],
                "ours_se": [0.005, 0.005],
                "paper_se_used": [0.005, 0.005],
            }
        )
        result = add_equivalence_statistics(comparison, margin=0.05, alpha=0.05)
        self.assertTrue(bool(result.loc[0, "equivalent"]))
        self.assertTrue(bool(result.loc[1, "different"]))

    def test_paired_tost_uses_equivalence_confidence_interval(self):
        result = _paired_tost(pd.Series([0.005, -0.004, 0.002, -0.001]), 0.02, 0.05)
        self.assertTrue(result["equivalent"])
        self.assertLess(result["ci90_low"], result["mean_diff"])
        self.assertGreater(result["ci90_high"], result["mean_diff"])

    def test_real_comparison_clusters_repeated_dataset_panels(self):
        comparison = pd.DataFrame(
            {
                "dataset": ["a", "a", "b", "b"],
                "diff": [0.01, 0.03, -0.02, 0.02],
            }
        )
        clustered = _cluster_mean_differences(comparison, "dataset")
        self.assertEqual(len(clustered), 2)
        self.assertAlmostEqual(clustered.loc["a"], 0.02)
        self.assertAlmostEqual(clustered.loc["b"], 0.0)

    def test_linear_pevi_grid_matches_single_beta_evaluation(self):
        model = (
            np.array([1.0, 0.5, 0.5, -0.25]),
            np.eye(4),
        )
        eval_data = {
            "xs": np.array([[-1.0, 0.0], [0.5, 0.0], [2.0, 0.0]]),
            "ys": np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.75]]),
        }
        grid = eval_linear_pevi_grid(model, eval_data, [0.1, 1.0])
        for beta in [0.1, 1.0]:
            _, _, expected = eval_linear_pevi(model, eval_data, beta)
            self.assertAlmostEqual(grid[beta], expected)

    def test_published_linear_pevi_uses_all_covariates(self):
        xs = np.array(
            [[-1.0, 0.5], [0.0, -0.5], [1.0, 1.5], [2.0, -1.5]]
        )
        arms = np.array([0, 1, 0, 1])
        outcomes = np.array([0.1, 0.2, 0.8, 1.0])
        theta, inv_gram, feature_count = fit_linear_pevi(
            xs,
            outcomes,
            arms,
            arm_count=2,
            feature_count=2,
        )
        self.assertEqual(feature_count, 2)
        self.assertEqual(theta.shape, (6,))
        self.assertEqual(inv_gram.shape, (6, 6))

    def test_worker_eval_cache_reuses_matching_key(self):
        calls = []

        def factory():
            calls.append(1)
            return object()

        first = _cached_eval_data(("unit-test", 1), factory)
        second = _cached_eval_data(("unit-test", 1), factory)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)

    def test_chunk_runner_fails_when_a_task_does_not_finish(self):
        def fail(_task):
            raise ValueError("expected test failure")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "1 task chunks are missing"):
                _run_chunked(
                    [{"task_id": "broken"}],
                    fail,
                    root / "chunks",
                    root / "combined.csv",
                    jobs=1,
                )
            self.assertTrue((root / "chunks" / "errors.jsonl").exists())

    def test_report_requires_every_primary_figure(self):
        summary = pd.DataFrame(
            [
                {
                    "figure": figure,
                    "margin": margin,
                    "all_cells_equivalent": True,
                    "paired_mean_equivalent": True,
                    "clustered_mean_equivalent": True,
                }
                for figure, margin in {
                    4: 0.0025,
                    5: 0.05,
                    6: 0.05,
                    7: 0.05,
                    8: 0.05,
                    9: 0.05,
                    10: 0.02,
                }.items()
            ]
        )
        primary = _primary_rows(summary)
        verdict = _overall_verdict(primary)
        self.assertEqual(len(primary), 7)
        self.assertTrue(verdict.endswith("\u3002"))
        primary.loc[primary["figure"] == 6, "all_cells_equivalent"] = False
        aggregate_only = _overall_verdict(primary)
        self.assertIn(
            "\u603b\u4f53\u5e73\u5747\u5c42\u9762\u590d\u73b0",
            aggregate_only,
        )

    def test_published_real_grid_matches_released_script(self):
        betas, settings, datasets, batches, _, _ = _real_grid(
            "full", "published"
        )
        self.assertEqual(betas, [0.1, 0.5, 1, 2, 5, 10])
        self.assertEqual(settings[0], ("pure", None, None))
        self.assertEqual(batches, [10, 100])
        self.assertEqual(datasets, PUBLISHED_REAL_DATASETS)
        self.assertEqual(len(datasets), 32)
        self.assertNotIn("skin-segmentation", datasets)
        self.assertEqual(len(PAPER_SPEC_REAL_DATASETS), 33)

    def test_published_real_transform_caps_rows_and_scales_signal(self):
        xs = np.arange(200, dtype=float).reshape(100, 2)
        labels = np.tile([0, 1], 50)
        data, _ = generate_bandit_data(
            xs,
            labels,
            signal_strength=0.5,
            max_rows=20,
            seed=4,
        )
        self.assertEqual(data["T"] + data["T_test"], 20)
        self.assertEqual(float(data["muxs"].max()), 0.5)
        self.assertEqual(float(data["muxs_test"].max()), 0.5)


if __name__ == "__main__":
    unittest.main()
