# -*- coding: utf-8 -*-
"""Testes numericos/cache da extracao18 sem ler audio ou treinar classificadores."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("exp18_test", Path(__file__).with_name("18_extrair_wpt_16khz_avfad.py"))
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


class WPTExtraction18(unittest.TestCase):
    def test_vectorized_descriptors_match_legacy_formulas(self):
        rng = np.random.default_rng(20261105)
        windows = rng.normal(size=(3, 19200)).astype(np.float32)
        windows[0] = 0
        windows[1, 20] = 100
        _, all_base = E.LEGACY.build_feature_names()
        keep = np.asarray([n.startswith("WPT__sym5__") for n in all_base])
        expected = np.vstack([E.LEGACY.wavelet_features_one_window(w)[keep] for w in windows])
        actual = E.descriptors_batch(windows)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(actual.dtype, np.float32)

    def test_batch_size_cannot_change_summary_or_feature_order(self):
        rng = np.random.default_rng(18)
        windows = rng.normal(size=(7, 6400)).astype(np.float32)
        one = E.summarize_windows(windows, batch_size=1)
        many = E.summarize_windows(windows, batch_size=4)
        np.testing.assert_array_equal(one, many)
        names, base = E.feature_names()
        self.assertEqual(len(names), 1920)
        self.assertEqual(len(base), 480)
        for i, summary in enumerate(("mean", "std", "q10", "q90")):
            self.assertTrue(all(n.endswith("__" + summary) for n in names[i * 480:(i + 1) * 480]))

    def test_vad_is_called_at_48k_before_regions_are_resampled(self):
        rng = np.random.default_rng(1818)
        regions = [rng.normal(size=17000).astype(np.float32), rng.normal(size=36000).astype(np.float32)]
        signal = np.r_[regions[0], np.zeros(1000), regions[1]].astype(np.float32)
        with patch.object(E.LEGACY, "detect_speech_regions", return_value=(regions, False)) as vad:
            vector, diag = E.extract_signal48(signal, 16000, 2)
        vad.assert_called_once()
        self.assertEqual(vad.call_args.args[1], 48000)
        self.assertEqual(diag["speech_regions"], 2)
        self.assertEqual(diag["target_active_samples"], sum((len(r) + 2) // 3 for r in regions))
        self.assertEqual(diag["sample_rate_hz"], 16000)
        self.assertEqual(vector.shape, (1920,))
        self.assertTrue(np.isfinite(vector).all())

    def test_cache_rejects_corruption_and_signature_changes(self):
        rows = pd.DataFrame({"sample_id": ["A004"], "participant_id": ["A"], "task": ["004"]})
        blob = np.zeros(1920, dtype=np.float32).tobytes()
        with TemporaryDirectory(prefix="avfad18_test_") as tmp:
            path = Path(tmp) / "new_cache.sqlite"
            conn = E.init_cache(path, "fixed_signature")
            with conn:
                conn.execute("INSERT INTO features VALUES (?,?,?,?,?,?)", ("A004", "A", "004", blob,
                             hashlib.sha256(blob).hexdigest(), "{}"))
            self.assertEqual(set(E.validate_cache(conn, rows)), {"A004"})
            with conn:
                conn.execute("UPDATE features SET vector=?", (b"corrupted",))
            with self.assertRaisesRegex(ValueError, "corrompido"):
                E.validate_cache(conn, rows)
            conn.close()
            with self.assertRaisesRegex(ValueError, "Assinatura"):
                E.init_cache(path, "other_signature")

    def test_export_embeds_ids_and_resume_never_rereads_completed_audio(self):
        rows = pd.DataFrame({"sample_id": ["A004", "B004", "C004"], "participant_id": ["A", "B", "C"],
                             "task": ["004"] * 3, "y": [0, 1, 1], "row_index": [0, 1, 2]})
        names, base = E.feature_names()
        ident = {"mode": "sintetico_temporario", "sample_ids": rows.sample_id.tolist()}
        controls = [{"sample_id": sid, "equivalent": True} for sid in rows.sample_id]
        def fake_control(root, meta, reference, out, batch_size):
            E.atomic_csv(out / "CONTROLE_REFERENCIA_48KHZ.csv", pd.DataFrame(controls))
            return controls
        diag = {"window_count": 2, "vad_fallback": 0, "sample_rate_hz": 16000}
        with TemporaryDirectory(prefix="avfad18_export_test_") as tmp:
            out = Path(tmp) / "output"
            with patch.object(E, "reference_check", side_effect=fake_control), \
                 patch.object(E, "read_signal48", return_value=(np.ones(100), {})), \
                 patch.object(E, "extract_signal48", return_value=(np.arange(1920, dtype=np.float32), diag.copy())):
                E.run(Path(tmp), out, rows, np.zeros((3, 1920)), names, base, ident, False, 2)
            with np.load(out / "features_wpt_sym5_16khz.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["sample_id"], rows.sample_id)
                np.testing.assert_array_equal(z["participant_id"], rows.participant_id)
                np.testing.assert_array_equal(z["task"], rows.task)
                self.assertEqual(z["wavelet"].shape, (3, 1920))
            before = {name: E.digest_file(out / name) for name in E.EXPORT_FILES}
            with patch.object(E, "read_signal48", side_effect=AssertionError("Audio foi relido")):
                E.run(Path(tmp), out, rows, np.zeros((3, 1920)), names, base, ident, False, 2)
            self.assertEqual(before, {name: E.digest_file(out / name) for name in E.EXPORT_FILES})
            (out / "ARTEFATOS_COMPLETOS.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Marcador"):
                E.run(Path(tmp), out, rows, np.zeros((3, 1920)), names, base, ident, False, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
