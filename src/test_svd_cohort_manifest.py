# -*- coding: utf-8 -*-
"""Invariantes sinteticos da coorte antes de qualquer avaliacao SVD."""
import unittest

import numpy as np
import pandas as pd

from svd_cohort_manifest import canonical_sessions, match_participants


class SVDCohortTest(unittest.TestCase):
    def test_matching_maximizes_cardinality_before_distance(self):
        people = pd.DataFrame({"participant_id": ["1", "2", "3", "4"], "y": [1, 1, 0, 0],
                               "age": [20., 23., 22., 17.], "sex": ["m"] * 4, "year": [2000] * 4})
        # A greedy closest choice for person1 would consume person3 and leave
        # person2 unmatched. Two pairs require 1->4 and 2->3 instead.
        pairs = match_participants(people)
        self.assertEqual(set(zip(pairs.positive_id, pairs.negative_id)), {("1", "4"), ("2", "3")})

    def test_matching_respects_sex_and_calipers_and_minimizes_valid_cost(self):
        people = pd.DataFrame({"participant_id": ["1", "2", "3", "4", "5", "6"], "y": [1, 1, 0, 0, 0, 0],
                               "age": [20., 22., 19., 23., 20., 20.], "sex": ["m", "m", "m", "m", "w", "m"],
                               "year": [2000, 2000, 2000, 2000, 2000, 2002]})
        pairs = match_participants(people)
        self.assertEqual(set(zip(pairs.positive_id, pairs.negative_id)), {("1", "3"), ("2", "4")})
        self.assertAlmostEqual(pairs.matching_cost.sum(), 2/3)
        self.assertTrue(pairs.age_difference.le(3).all())
        self.assertTrue(pairs.year_difference.le(1).all())
        self.assertFalse(pairs.positive_id.duplicated().any())
        self.assertFalse(pairs.negative_id.duplicated().any())
        pd.testing.assert_frame_equal(pairs, match_participants(people.iloc[::-1]))

    def test_earliest_eligible_session_and_numeric_tie(self):
        rows = pd.DataFrame({"participant_id": ["1"] * 4, "session_id": ["99", "10", "2", "1"],
                             "session_date": ["1990-01-01", "2000-01-01", "2000-01-01", "1999-01-01"],
                             "birth_date": ["1980-01-01"] * 4, "sex": ["w"] * 4,
                             "task": ["phrase", "a_n", "a_n", "a_h"]})
        chosen, removed = canonical_sessions(rows)
        self.assertEqual(chosen.session_id.tolist(), ["2"])
        self.assertEqual(removed.session_id.tolist(), ["10"])
        self.assertEqual(chosen.sex.tolist(), ["w"])
        self.assertEqual(chosen.year.tolist(), [2000])
        self.assertTrue(np.isfinite(chosen.age).all())

    def test_missing_date_or_sex_is_not_silently_excluded(self):
        base = pd.DataFrame({"participant_id": ["1"], "session_id": ["2"], "session_date": ["2000-01-01"],
                             "birth_date": ["1980-01-01"], "sex": ["m"], "task": ["a_n"]})
        for column, value in (("session_date", "bad-date"), ("birth_date", ""), ("sex", "unknown")):
            with self.subTest(column=column):
                invalid = base.copy()
                invalid[column] = value
                with self.assertRaises(ValueError):
                    canonical_sessions(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
