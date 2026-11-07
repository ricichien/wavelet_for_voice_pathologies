# -*- coding: utf-8 -*-
"""Testes29 sintéticos: não abrem corpus, importam Torch ou usam GPU/rede."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp29_tests", HERE / "29_extrair_vogal_wavelet_ssl_avfad.py")
E = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(E)


class Vowel29Tests(unittest.TestCase):
    def meta(self):
        return pd.DataFrame([{"sample_id": sid, "participant_id": sid[:-3], "task": "002", "y": i % 2,
                              "label": "Pathological" if i % 2 else "Normal", "age": 54, "sex": "F", "visit_place": "HSA",
                              "zip_name": "synthetic.zip", "audio_path_in_zip": sid[:-3]+"/"+sid+".wav",
                              "expected_sample_rate_hz": 48000, "expected_channels": 1, "expected_frames": 9600,
                              "expected_bits_per_sample": 16} for i, sid in enumerate(E.PILOT_IDS)])

    def arrays(self):
        return {key: (np.arange(width, dtype=np.float32)+i)/100 for i, (key, width) in enumerate(E.ARRAY_WIDTHS.items())}

    def diag(self, full=False):
        return {"window_count": 3, "hidden_frames": 5, "chunk_count": 1, "speech_regions": 1, "vad_fallback": 0,
                "active_duration_s": .2, "active_rms_before": .2, "rms_global_gain": .5, "active_rms_after": .1,
                "ssl24_control_done": True, "ssl24_control_exact": True, "ssl24_control_max_abs_error": 0., "ssl24_control_frames": 5,
                "pilot_reference_checked": full, "pilot_reference_exact": True if full else None,
                "pilot_reference_max_abs_error": 0. if full else None}

    def quality(self):
        quality = E.EXP23.audio_quality(np.zeros((9600, 1), np.float32), 48000)
        quality.update(decoded_format="WAV", decoded_subtype="PCM_16", decoded_bits_per_sample=16)
        return quality

    def ident(self, mode="piloto_3"):
        return {"mode": mode, "sample_ids": list(E.PILOT_IDS), "configuration": E.configuration("synthetic", False),
                "code": {str(Path(E.__file__).resolve()): E.sha(E.__file__)}, "model_files": {}, "environment": {"synthetic": True}}

    def cache(self, out, signature="synthetic", full=False):
        con = E.EXP27.init_cache(out / "checkpoint_vogal_wavelet_ssl.sqlite", signature)
        for row in self.meta().itertuples(index=False):
            E.store(con, row, self.arrays(), {"diagnostics": self.diag(full), "quality": self.quality()})
        return con

    def test_people_headers_original_validation_and_case_variants_are_checked(self):
        meta = self.meta()
        people = meta[["participant_id", "y", "label", "age", "sex", "visit_place"]].copy()
        headers = pd.DataFrame([{"participant_id": row.participant_id, "label": row.label, "age": row.age,
                                 "sex": row.sex, "visit_place": row.visit_place, "numeric_code": 1220 if row.y else 0,
                                 "audio_found": True, "audio_path": row.audio_path_in_zip.upper(),
                                 "sample_rate_hz": 48000, "channels": 1, "bits_per_sample": 16, "frames": 9600}
                                for row in meta.itertuples(index=False)])
        E.validate_people(people, headers, headers.copy(), real=False)
        for field, value in (("numeric_code", 0), ("frames", 0), ("audio_path", "../../X002.wav")):
            changed = headers.copy(); changed.loc[1, field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): E.validate_people(people, changed, changed, real=False)
        changed = headers.copy(); changed.loc[0, "frames"] += 1
        with self.assertRaises(ValueError): E.validate_people(people, changed, headers, real=False)

    def test_reader_matches18_exactly_for_synthetic_pcm16_wav(self):
        signal = (.15*np.sin(2*np.pi*177*np.arange(9600)/48000)+.015).astype(np.float32)
        stream = io.BytesIO(); E.EXP23.sf.write(stream, signal, 48000, format="WAV", subtype="PCM_16")
        row = next(self.meta().itertuples(index=False))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / row.zip_name, "w") as z: z.writestr(row.audio_path_in_zip, stream.getvalue())
            opened29, opened18 = {}, {}
            try:
                actual, quality = E.read_audio(root, row, opened29)
                expected, _ = E.EXP18.read_signal48(root, row, opened18)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(quality["decoded_subtype"], "PCM_16")
                self.assertEqual(quality["source_samples"], 9600)
                changed = SimpleNamespace(**{**row._asdict(), "expected_frames": 9601})
                with self.assertRaises(ValueError): E.read_audio(root, changed, opened29)
            finally:
                for z in [*opened29.values(), *opened18.values()]: z.close()

    def test_WPT_RMS_uses_one_global_gain_and_retains_same_windows(self):
        t = np.arange(48000, dtype=np.float64)/48000
        signal = (.05*np.sin(2*np.pi*173*t)+.01*np.sin(2*np.pi*346*t)).astype(np.float32)
        with patch.object(E.EXP23.LEGACY, "detect_speech_regions", side_effect=lambda x, sr: ([x], False)) as vad:
            before, diag1 = E.extract_wpt(signal)
            after, diag2 = E.extract_wpt(signal*3)
        self.assertEqual(vad.call_count, 2)
        self.assertEqual(diag1["window_count"], diag2["window_count"])
        self.assertAlmostEqual(diag1["active_rms_after"], .1, places=7)
        self.assertAlmostEqual(diag1["active_rms_before"]*diag1["rms_global_gain"], .1, places=12)
        normalized1, _ = E.EXP23.normalize_regions([signal])
        normalized2, _ = E.EXP23.normalize_regions([signal*3])
        np.testing.assert_allclose(normalized1[0], normalized2[0], rtol=2e-6, atol=2e-8)
        # EPS e quantização float32 podem alterar ZCR em subbandas quase nulas;
        # não afirmar invariância exata de todos os descritores nesse caso.
        energy = np.arange(480) % 5 == 0
        np.testing.assert_allclose(before["wavelet_rms"].reshape(4,480)[:, energy],
                                   after["wavelet_rms"].reshape(4,480)[:, energy], rtol=2e-5, atol=2e-5)
        self.assertFalse(np.array_equal(before["wavelet_raw"], after["wavelet_raw"]))

    def test_control24_and_pilot_compare_all_six_vectors(self):
        a = np.ones(2048, np.float32)
        self.assertTrue(E.check_control(a, a.copy(), 5, 5)["ssl24_control_exact"])
        with self.assertRaises(ValueError): E.check_control(a, a, 5, 6)
        with self.assertRaises(ValueError): E.check_control(a, a+.01, 5, 5)
        before = self.arrays(); after = {key: value.copy() for key, value in before.items()}
        self.assertTrue(E.compare_pilot(before, after)["pilot_reference_exact"])
        for key in E.ARRAY_WIDTHS:
            changed = {k: value.copy() for k, value in before.items()}; changed[key][0] += .01
            with self.subTest(key=key), self.assertRaises(ValueError): E.compare_pilot(changed, before)

    def test_cache_hashes_vectors_and_nested_diagnostics_quality(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp); con = self.cache(out)
            found = E.validate_cache(con, self.meta())
            np.testing.assert_array_equal(found[E.PILOT_IDS[0]][0]["embeddings_layer16"], self.arrays()["embeddings_layer16"])
            con.execute("UPDATE features SET diagnostics='{}' WHERE sample_id=?", (E.PILOT_IDS[0],)); con.commit()
            with self.assertRaises(ValueError): E.validate_cache(con, self.meta())
            con.close()
            with self.assertRaises(ValueError): E.EXP27.init_cache(out / "checkpoint_vogal_wavelet_ssl.sqlite", "different")
        with TemporaryDirectory() as tmp:
            out = Path(tmp); con = self.cache(out)
            con.execute("UPDATE features SET vectors=? WHERE sample_id=?", (b"broken", E.PILOT_IDS[0])); con.commit()
            with self.assertRaises(ValueError): E.validate_cache(con, self.meta())
            con.close()

    def test_export_preserves_order_shapes_names_and_full_marker(self):
        meta, ident = self.meta(), self.ident()
        with TemporaryDirectory() as tmp:
            out = Path(tmp); con = self.cache(out); found = E.validate_cache(con, meta); con.close()
            E.EXP20.atomic_text(out / "assinatura_execucao.json", E.EXP20.json_text(ident))
            E.EXP20.atomic_text(out / "PROTOCOLO.txt", E.protocol(ident))
            E.export(out, meta, found, ident, 1., {})
            with np.load(out / E.EXPORTS[0], allow_pickle=False) as z:
                np.testing.assert_array_equal(z["sample_id"], E.PILOT_IDS)
                np.testing.assert_array_equal(z["hidden_layer_indices"], [24, 5, 12, 16])
                for key, width in E.ARRAY_WIDTHS.items():
                    self.assertEqual(z[key].shape, (3, width)); self.assertEqual(z[key].dtype, np.float32)
                for key, value in E.names().items(): np.testing.assert_array_equal(z[key], value)
            self.assertEqual(set(E.verify_exports(out)), set(E.EXPORTS))
            pilot, _ = E.load_pilot(out)
            self.assertEqual(set(pilot), set(E.PILOT_IDS))
            with self.assertRaises(ValueError): E.export(out, meta, {E.PILOT_IDS[0]: found[E.PILOT_IDS[0]]}, ident, 1, {})

    def test_guard_rejects_before_any_output_encoder_or_WAV(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "new"
            with patch.object(E, "previous_complete", side_effect=ValueError("pending")), patch.object(E.EXP20, "load_encoder", side_effect=AssertionError("no encoder")):
                with self.assertRaises(ValueError): E.run(Path(tmp), out, self.meta(), self.ident(), Path("unused"), False, "synthetic")
            self.assertFalse(out.exists())

    def test_end_to_end_simulated_pilot_full_resume_without_real_audio_or_torch(self):
        meta = self.meta(); base = self.arrays()
        def fake_wpt(signal):
            return {key: value.copy() for key, value in base.items() if key.startswith("wavelet_")}, self.diag()
        infer27 = (np.stack([base[f"embeddings_layer{x:02d}"] for x in E.EXP27.LAYERS]), base["embeddings_layer24"], 5)
        patches = [patch.object(E, "previous_complete", return_value=None),
                   patch.object(E.EXP20, "load_encoder", return_value=(None, None, None, {"gpu": "synthetic"})),
                   patch.object(E, "read_audio", return_value=(np.zeros(9600, np.float32), self.quality())),
                   patch.object(E, "extract_wpt", side_effect=fake_wpt),
                   patch.object(E.EXP20, "prepare_signal", return_value=([np.zeros(3200, np.float32)],
                       {"speech_regions": 1, "vad_fallback": 0, "active_duration_s": .2, "chunk_count": 1, "target_active_samples": 3200})),
                   patch.object(E.EXP27, "infer_vectors", return_value=infer27),
                   patch.object(E.EXP20, "infer_vector", return_value=(base["embeddings_layer24"], 5))]
        from contextlib import ExitStack
        with TemporaryDirectory() as tmp, ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            root = Path(tmp); pilot_out = root / "pilot"; full_out = root / "full"
            E.run(root, pilot_out, meta, self.ident(), Path("unused"), False, "synthetic")
            pilot, _ = E.load_pilot(pilot_out)
            E.run(root, full_out, meta, self.ident("completo"), Path("unused"), False, "synthetic", pilot)
            full_meta = pd.read_csv(full_out / E.EXPORTS[1])
            self.assertTrue(full_meta.pilot_reference_checked.eq(True).all())
            self.assertTrue(full_meta.pilot_reference_exact.eq(True).all())
            before = {name: (E.sha(full_out/name), (full_out/name).stat().st_mtime_ns) for name in E.EXPORTS}
            with patch.object(E.EXP20, "load_encoder", side_effect=AssertionError("no reload")), patch.object(E, "read_audio", side_effect=AssertionError("no WAV")):
                E.run(root, full_out, meta, self.ident("completo"), Path("unused"), False, "synthetic", pilot)
            self.assertEqual(before, {name: (E.sha(full_out/name), (full_out/name).stat().st_mtime_ns) for name in E.EXPORTS})
            (full_out / E.EXPORTS[0]).write_bytes(b"tampered")
            with self.assertRaises(ValueError): E.run(root, full_out, meta, self.ident("completo"), Path("unused"), False, "synthetic", pilot)


if __name__ == "__main__": unittest.main(verbosity=2)
