# -*- coding: utf-8 -*-
"""Testes sinteticos de independencia e invariantes do seletor por pessoa."""
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wavelet_participant_selection import ParticipantMeanANOVA


class ParticipantMeanANOVATest(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(np.arange(8), 3)
        self.y = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 3)
        rng = np.random.default_rng(31)
        self.X = rng.normal(size=(24, 6))
        self.X[:, 0] += self.y * 0.9

    def test_ranking_matches_one_mean_per_person(self):
        selector = ParticipantMeanANOVA(k=3).fit(self.X, self.y, self.groups)
        means = np.vstack([self.X[self.groups == p].mean(axis=0) for p in range(8)])
        expected, _ = f_classif(means, self.y[::3])
        assert_allclose(selector.scores_, expected)
        self.assertEqual(selector.n_participants_, 8)
        assert_array_equal(selector.participant_class_counts_, [4, 4])

    def test_repeating_all_phrases_of_one_person_does_not_reweight_ranking(self):
        one_person = np.flatnonzero(self.groups == 0)
        repeated = np.r_[np.arange(24), np.tile(one_person, 8)]
        first = ParticipantMeanANOVA(k=3).fit(self.X, self.y, self.groups)
        second = ParticipantMeanANOVA(k=3).fit(self.X[repeated], self.y[repeated], self.groups[repeated])
        assert_allclose(first.scores_, second.scores_, rtol=1e-12, atol=1e-12)
        assert_array_equal(first.get_support(), second.get_support())

    def test_transform_cannot_refit_on_held_out_people(self):
        selector = ParticipantMeanANOVA(k=2).fit(self.X, self.y, self.groups)
        frozen_scores = selector.scores_.copy()
        frozen_indices = selector.get_support(indices=True)
        held_out = np.full((5, 6), 1e9)
        held_out[:, frozen_indices] = -1e9
        result = selector.transform(held_out)
        assert_array_equal(result, held_out[:, frozen_indices])
        assert_array_equal(selector.scores_, frozen_scores)
        assert_array_equal(selector.get_support(indices=True), frozen_indices)
        self.assertEqual(selector.n_participants_, 8)

    def test_transform_preserves_floating_dtype_of_model_inputs(self):
        selector = ParticipantMeanANOVA(k=2).fit(self.X, self.y, self.groups)
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                data = self.X.astype(dtype)
                actual = selector.transform(data)
                self.assertEqual(actual.dtype, np.dtype(dtype))
                assert_array_equal(actual, data[:, selector.get_support(indices=True)])

    def test_invalid_identifiers_labels_and_lengths_are_rejected(self):
        missing = self.groups.astype(object)
        missing[0] = pd.NA
        empty = self.groups.astype(object)
        empty[0] = " "
        conflicting = self.y.copy()
        conflicting[0] = 1
        cases = [
            (self.X, self.y, None),
            (self.X, self.y, self.groups[:-1]),
            (self.X, self.y, self.groups[:, None]),
            (self.X, self.y, missing),
            (self.X, self.y, empty),
            (self.X, conflicting, self.groups),
            (self.X, np.where(np.arange(24) == 0, np.nan, self.y), self.groups),
            (self.X, np.full(24, 1), self.groups),
            (self.X, self.y[:, None], self.groups),
            (self.X, self.y, np.repeat([0, 1], 12)),
        ]
        for X, y, groups in cases:
            with self.subTest(groups=repr(groups)):
                with self.assertRaises(ValueError):
                    ParticipantMeanANOVA(k=2).fit(X, y, groups)

    def test_exact_ties_use_original_column_order_and_constants_rank_last(self):
        X = np.column_stack([self.X[:, 0], self.X[:, 0], np.full(24, 7.0)])
        selector = ParticipantMeanANOVA(k=1).fit(X, self.y, self.groups)
        assert_array_equal(selector.get_support(indices=True), [0])
        self.assertEqual(selector.scores_[2], -np.inf)
        # F infinito por separacao perfeita e valido e nao deve ser descartado.
        perfect = np.column_stack([self.X[:, 0], self.y.astype(float), np.full(24, 2.0)])
        selector.fit(perfect, self.y, self.groups)
        assert_array_equal(selector.get_support(indices=True), [1])
        self.assertEqual(selector.scores_[1], np.inf)

    def test_clone_unfitted_and_k_limited_to_available_features(self):
        fitted = ParticipantMeanANOVA(k=128).fit(self.X, self.y, self.groups)
        self.assertEqual(fitted.effective_k_, 6)
        assert_array_equal(fitted.transform(self.X), self.X)
        fresh = clone(fitted)
        self.assertEqual(fresh.k, 128)
        with self.assertRaises(NotFittedError):
            fresh.transform(self.X)
        for bad_k in (0, -1, 1.5, True, "all"):
            with self.subTest(k=bad_k):
                with self.assertRaises(ValueError):
                    ParticipantMeanANOVA(k=bad_k).fit(self.X, self.y, self.groups)

    def test_pipeline_passes_training_groups_after_imputation_and_variance(self):
        X = np.column_stack([self.X, np.full(24, 1.0)])
        X[0, 1] = np.nan
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold()),
            ("selector", ParticipantMeanANOVA(k=2)),
            ("scale", StandardScaler()),
            ("model", SVC(kernel="linear", class_weight="balanced")),
        ])
        pipeline.fit(X, self.y, selector__groups=self.groups)
        self.assertEqual(pipeline.named_steps["selector"].n_features_in_, 6)
        self.assertEqual(pipeline.decision_function(X[:4]).shape, (4,))

    def test_names_and_number_of_columns_must_match_training(self):
        frame = pd.DataFrame(self.X, columns=[f"feat{i}" for i in range(6)])
        selector = ParticipantMeanANOVA(k=2).fit(frame, self.y, self.groups)
        expected = frame.columns[selector.get_support()].to_numpy()
        assert_array_equal(selector.get_feature_names_out(), expected)
        with self.assertRaises(ValueError):
            selector.transform(frame.iloc[:, ::-1])
        with self.assertRaises(ValueError):
            selector.transform(self.X[:, :-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
