# -*- coding: utf-8 -*-
"""Checks sinteticos do auditor23; nenhum audio ou classificador."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

source = Path(__file__).with_name("verificar_extracao23.py")
spec = importlib.util.spec_from_file_location("audit23_tested", source)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class MetadataComparisonTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"sample_id": ["A004", "B004"],
                                   "speech_regions": [1, 2], "window_count": [13, 27],
                                   "active_duration_s": [1.25, 2.5],
                                   "recording_duration_s": [2., 5.],
                                   "vad_fallback": [0, 1], "speech_ratio": [.625, .5]})

    def test_one_ulp_only_in_ratio_is_accepted_and_reported(self):
        changed = self.frame.copy()
        changed.loc[0, "speech_ratio"] = np.nextafter(.625, 1.)
        result = audit.compare_metadata18(self.frame, changed)
        self.assertEqual(result["speech_ratio_different"], 1)
        self.assertEqual(result["speech_ratio_max_abs_error"], np.spacing(.625))
        self.assertEqual(result["speech_ratio_rtol"], 0)

    def test_larger_ratio_change_is_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, "speech_ratio"] += 1e-12
        with self.assertRaises(AssertionError):
            audit.compare_metadata18(self.frame, changed)

    def test_discrete_and_duration_fields_stay_exact(self):
        for key, value in (("window_count", 14), ("vad_fallback", 1),
                           ("sample_id", "Z004"),
                           ("active_duration_s", np.nextafter(1.25, 2.))):
            changed = self.frame.copy()
            changed.loc[0, key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                audit.compare_metadata18(self.frame, changed)

    def test_nan_ratio_is_rejected(self):
        changed = self.frame.copy()
        changed.loc[0, "speech_ratio"] = np.nan
        with self.assertRaises(AssertionError):
            audit.compare_metadata18(self.frame, changed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
