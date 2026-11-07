# -*- coding: utf-8 -*-
"""Verificacoes sinteticas de isolamento de teste e bootstrap agrupado."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("experiment15_test", Path(__file__).with_name("15_avaliar_wpt_compacta_participante.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Experiment15Integrity(unittest.TestCase):
    def test_outer_test_cannot_change_search_or_selected_features(self):
        rng = np.random.default_rng(261105)
        ids = np.repeat([f"S{i:03d}" for i in range(24)], 3)
        y = np.repeat([0, 1] * 12, 3)
        meta = pd.DataFrame({"participant_id": ids, "sample_id": [f"X{i}" for i in range(len(ids))], "y": y})
        x = rng.normal(size=(len(meta), 384)).astype(np.float32)
        x[:, :4] += y[:, None] * 0.7
        names = np.asarray([f"WPT__sym5__L5__{i}" for i in range(192)]
                           + [f"WPT__sym5__L6__{i}" for i in range(192)])
        masks = {"L5": np.arange(192), "L6": np.arange(192, 384), "L5_L6": np.arange(384)}
        train, test = next(MODULE.EXP13.row_splits(meta, 5, 20261105))
        grid = (("L5_L6", 32, 0.1), ("L5", 64, 1.0))
        original = MODULE.evaluate_fold(x, meta, names, masks, 20261105, 1, train, test, lambda _: None, grid)
        changed = x.copy()
        changed[test] = rng.normal(1e5, 1e4, size=changed[test].shape)
        altered = MODULE.evaluate_fold(changed, meta, names, masks, 20261105, 1, train, test, lambda _: None, grid)
        for artifact in ("candidatos.csv", "oof_interno.csv", "atributos.csv"):
            pd.testing.assert_frame_equal(original[artifact], altered[artifact])
        self.assertFalse(np.array_equal(original["predicoes.csv"].score, altered["predicoes.csv"].score))

    def test_cluster_bootstrap_does_not_invent_people_from_repetitions(self):
        one = pd.DataFrame({"participant_id": ["P0", "P1", "P2", "P3"] * 2,
                            "y": [0, 0, 1, 1] * 2, "seed": [101] * 8,
                            "arm": ["E"] * 4 + ["D"] * 4,
                            "predicao": [0, 0, 1, 0, 0, 1, 0, 0]})
        result = MODULE.bootstrap_pairs(one, [101], "E", "D")
        self.assertEqual(result["delta_BA"], 0.5)
        copied = one.copy()
        copied["seed"] = 102
        duplicated = MODULE.bootstrap_pairs(pd.concat([one, copied]), [101, 102], "E", "D")
        self.assertEqual(result, duplicated)

    def test_checkpoint_predictions_reject_label_or_identity_corruption(self):
        source = pd.DataFrame({"participant_id": ["P0", "P1"], "y": [0, 1],
                               "score": [-0.5, 0.8], "predicao": [0, 1], "seed": [101, 101], "fold": [1, 1]})
        frame = pd.concat([source.assign(arm=arm) for arm in MODULE.ARMS], ignore_index=True)
        expected = pd.Series({"P0": 0, "P1": 1})
        MODULE.validate_predictions(frame, {"P0", "P1"}, expected, 101, 1)
        for column, value in (("y", 1), ("participant_id", "OUTRO"), ("predicao", 1)):
            corrupted = frame.copy()
            corrupted.loc[0, column] = value
            with self.assertRaises(ValueError):
                MODULE.validate_predictions(corrupted, {"P0", "P1"}, expected, 101, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
