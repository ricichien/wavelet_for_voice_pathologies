# -*- coding: utf-8 -*-
"""Verifica o auditor28 usando somente amostras artificiais pequenas."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("audit28_tests", HERE/"verificar_camadas_ssl28.py")
AUDIT = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AUDIT)
EXP = AUDIT.EXP28


class Audit28Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(928)
        cls.meta = pd.DataFrame([{"sample_id": f"FAKE{i:03d}_{j}", "participant_id": f"FAKE{i:03d}",
                                  "task": str(j).zfill(3), "y": int(i % 4 != 0)}
                                 for i in range(40) for j in range(2+int(i % 5 != 0))])
        cls.matrices = {key: rng.normal(size=(len(cls.meta), 8)).astype(np.float32)
                        for key in ("wpt", "ssl24", "ssl05", "ssl12", "ssl16")}
        cls.names = {key: np.array([f"{key}_{j}" for j in range(8)]) for key in cls.matrices}
        processors = {key: (modality, 3 if key.startswith("SSL256") else 2)
                      for key, (modality, _) in EXP.PREPROCESSORS.items()}
        cls.patch = patch.object(EXP, "PREPROCESSORS", processors); cls.patch.start()
        cls.grid = (.01, 1.)
        cls.train, cls.test = next(AUDIT.independent_splits(cls.meta, 5, 20261105))
        cls.frames, cls.arrays = EXP.evaluate_fold(cls.matrices, cls.meta, cls.names, cls.train, cls.test,
                                                  20261105, 1, c_grid=cls.grid)

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()

    def test_one_fold_end_to_end_all_nine_arms(self):
        totals = AUDIT.blank_totals()
        AUDIT.audit_checkpoint(self.matrices, self.meta, self.names, self.train, self.test, 20261105, 1,
                               self.frames, self.arrays, totals, self.grid)
        self.assertEqual(totals["outer_folds"], 1)
        self.assertEqual(totals["classifiers"], 9)
        self.assertEqual(totals["preprocessors"], 36)
        self.assertEqual(totals["test_margins"], 9*len(self.test))
        self.assertEqual(totals["sampled_train_margins"], 27)
        self.assertLess(totals["max_margin_error"], 1e-7)

    def test_saved_checkpoint_roundtrip_passes_independent_audit(self):
        with TemporaryDirectory() as folder:
            p = Path(folder)/"checkpoint"
            EXP.save_checkpoint(p, self.frames, self.arrays, "synthetic-signature")
            frames, arrays = EXP.load_checkpoint(p, "synthetic-signature")
            AUDIT.audit_checkpoint(self.matrices, self.meta, self.names, self.train, self.test, 20261105, 1,
                                   frames, arrays, AUDIT.blank_totals(), self.grid)

    def test_corrupt_test_margin_is_rejected(self):
        frames = {name: frame.copy() for name, frame in self.frames.items()}
        frames["predicoes_teste_gravacao.csv"].loc[0, "score"] += .25
        with self.assertRaises((AssertionError, ValueError)):
            AUDIT.audit_checkpoint(self.matrices, self.meta, self.names, self.train, self.test, 20261105, 1,
                                   frames, self.arrays, AUDIT.blank_totals(), self.grid)

    def test_corrupt_coefficients_are_rejected(self):
        arrays = {k: v.copy() for k, v in self.arrays.items()}
        arrays[f"{EXP.ARMS[0]}__coef"] += .2
        with self.assertRaises((AssertionError, ValueError)):
            AUDIT.audit_checkpoint(self.matrices, self.meta, self.names, self.train, self.test, 20261105, 1,
                                   self.frames, arrays, AUDIT.blank_totals(), self.grid)

    def test_statistics_from_external_rows_are_rejected(self):
        arrays = {k: v.copy() for k, v in self.arrays.items()}
        arrays["outer__WPT128__scaler_mean"] += .1
        with self.assertRaises(AssertionError):
            AUDIT.verify_preprocessor(self.matrices["wpt"], self.meta.y.to_numpy(), self.train,
                                       arrays, "outer__WPT128__", 2)

    def test_support_original_rows_must_belong_to_training(self):
        arrays = {k: v.copy() for k, v in self.arrays.items()}
        arrays[f"{EXP.ARMS[0]}__support_original_rows"][0] = self.test[0]
        with self.assertRaises((AssertionError, ValueError)):
            AUDIT.audit_checkpoint(self.matrices, self.meta, self.names, self.train, self.test, 20261105, 1,
                                   self.frames, arrays, AUDIT.blank_totals(), self.grid)

    def test_winner_uses_C_then_predefined_layer_order(self):
        d = pd.DataFrame([{"candidate_id": f"{layer}-{c}", "representation": f"SSL128_L{layer:02d}",
                           "layer": layer, "C": c, "inner_BA": .6+1e-13*int(layer == 5)}
                          for layer in (5, 16, 12, 24) for c in (1., .01)])
        winner = AUDIT.independent_winner(d, d.representation.unique())
        self.assertEqual(winner.C, .01)
        self.assertEqual(winner.layer, 24)

    def test_person_split_retains_all_recordings(self):
        for train, test in AUDIT.independent_splits(self.meta, 5, 123):
            self.assertFalse(set(self.meta.iloc[train].participant_id) & set(self.meta.iloc[test].participant_id))
            self.assertEqual(set(train) | set(test), set(range(len(self.meta))))

    def test_bootstrap_keeps_repeated_seeds_as_person_cluster(self):
        people = pd.DataFrame({"participant_id": [f"S{i}" for i in range(8)], "y": [0]*4+[1]*4})
        a = people.assign(seed=1, arm="A", score=[-1, -1, 1, 1, 1, 1, 1, -1])
        b = people.assign(seed=1, arm="B", score=[-1, 1, 1, 1, 1, 1, -1, -1])
        raw = pd.concat([a, b], ignore_index=True); raw["predicao"] = (raw.score >= 0).astype(int)
        once = AUDIT.paired_interval(raw, [1], {"A": 1, "B": -1})
        repeated = pd.concat([raw.assign(seed=s) for s in range(1, 6)], ignore_index=True)
        many = AUDIT.paired_interval(repeated, list(range(1, 6)), {"A": 1, "B": -1})
        np.testing.assert_array_equal(once, many)
        self.assertAlmostEqual(once[0], .25)

    def test_incomplete_run_cannot_create_a_report(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            report = root/"report.txt"
            with self.assertRaises(AssertionError):
                AUDIT.audit(root, root, root, report)
            self.assertFalse(report.exists())
            self.assertFalse(report.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
