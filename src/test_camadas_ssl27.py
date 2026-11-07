# -*- coding: utf-8 -*-
"""Testes27 sintéticos: não importam Torch, não leem WAV nem usam GPU/rede."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("exp27_test", Path(__file__).with_name("27_extrair_camadas_ssl_avfad.py"))
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


class ArrayTensor:
    def __init__(self, array): self.array = np.asarray(array)
    @property
    def shape(self): return self.array.shape
    def __getitem__(self, key): return ArrayTensor(self.array[key])
    def cpu(self): return self
    def numpy(self): return self.array
    def to(self, device): return self


class Extraction27Tests(unittest.TestCase):
    def metadata(self):
        return pd.DataFrame({"sample_id": ["B004", "A004"], "participant_id": ["B", "A"],
                             "task": ["004", "004"], "y": [1, 0], "matrix_row": [0, 1]})

    def diagnostics(self):
        return {"hidden_frames": 9, "control24_allclose": True, "control24_exact": True,
                "control24_max_abs_error": 0., "control24_mean_abs_error": 0.,
                "control24_reference_frames": 9, "chunk_count": 2, "vad_fallback": 0,
                "target_active_samples": 4096, "speech_regions": 1, "original_sample_rate_hz": 48000,
                "channels_original": 1, "active_duration_s": .256, "recording_duration_s": .256,
                "speech_ratio": 1.}

    def identity(self, meta):
        return {"mode": "synthetic", "sample_ids": meta.sample_id.tolist(), "configuration": EXP.configuration(EXP.MODEL_REVISION, False, "cuda")}

    def filled_cache(self, out, signature, meta):
        con = EXP.init_cache(out / "checkpoint_camadas_ssl.sqlite", signature)
        for index, row in enumerate(meta.itertuples(index=False)):
            EXP.store(con, row, np.full((len(EXP.LAYERS), EXP.DIM), index, np.float32), self.diagnostics())
        return con

    def test_index_contract_and_weighted_pooling_use5_12_16_not_adjacent_blocks(self):
        chunks = [np.arange(2, dtype=np.float32), np.arange(5, dtype=np.float32)]
        def processor(chunk, **kwargs):
            self.assertEqual(kwargs, {"sampling_rate": 16000, "return_tensors": "pt", "return_attention_mask": True, "padding": False})
            return {"input_values": ArrayTensor(chunk)}
        def model(input_values, **kwargs):
            self.assertEqual(kwargs, {"output_hidden_states": True, "return_dict": True})
            n = len(input_values.array)
            states = tuple(ArrayTensor(np.repeat((np.arange(n, dtype=np.float32) + layer)[None, :, None], EXP.WIDTH, axis=2)) for layer in range(25))
            return SimpleNamespace(hidden_states=states, last_hidden_state=states[24])
        vectors, control, count = EXP.infer_vectors(chunks, SimpleNamespace(inference_mode=nullcontext), processor, model, "synthetic")
        self.assertEqual(count, 7)
        self.assertEqual(vectors.dtype, np.float32)
        for index, layer in enumerate(EXP.LAYERS):
            values = np.concatenate([np.arange(2)+layer, np.arange(5)+layer])
            expected = np.repeat(np.asarray([values.mean(), values.std()], dtype=np.float32), EXP.WIDTH)
            np.testing.assert_array_equal(vectors[index], expected)
        values = np.concatenate([np.arange(2)+24, np.arange(5)+24])
        np.testing.assert_array_equal(control, np.repeat(np.asarray([values.mean(), values.std()], dtype=np.float32), EXP.WIDTH))
        self.assertEqual(EXP.LAYERS, (5, 12, 16))

    def test_hidden_state_count_shapes_and_last_control_are_strict(self):
        states = [ArrayTensor(np.zeros((1, 3, EXP.WIDTH), np.float32)) for _ in range(25)]
        valid = SimpleNamespace(hidden_states=states, last_hidden_state=states[-1])
        self.assertEqual(set(EXP.hidden_tensors(valid)), {5, 12, 16, 24})
        for output in (SimpleNamespace(hidden_states=states[:-1], last_hidden_state=states[-1]),
                       SimpleNamespace(hidden_states=states, last_hidden_state=ArrayTensor(np.zeros((2, 3, EXP.WIDTH)))),
                       SimpleNamespace(hidden_states=states, last_hidden_state=ArrayTensor(np.zeros((1, 3, 7))))):
            with self.assertRaises(ValueError): EXP.hidden_tensors(output)
        mismatch = states.copy(); mismatch[12] = ArrayTensor(np.zeros((1, 2, EXP.WIDTH)))
        with self.assertRaises(ValueError): EXP.hidden_tensors(SimpleNamespace(hidden_states=mismatch, last_hidden_state=states[-1]))

    def test_control20_checks_values_frames_audio_and_reports_nonexact(self):
        reference = np.ones(EXP.DIM, np.float32)
        diag = self.diagnostics()
        exact = EXP.check_reference20(reference.copy(), reference, diag, diag)
        self.assertTrue(exact["control24_exact"])
        nearby = reference.copy(); nearby[0] += 1e-6
        result = EXP.check_reference20(nearby, reference, diag, diag)
        self.assertFalse(result["control24_exact"])
        self.assertGreater(result["control24_max_abs_error"], 0)
        with self.assertRaises(ValueError): EXP.check_reference20(reference+.01, reference, diag, diag)
        for key, value in (("hidden_frames", 10), ("chunk_count", 3), ("speech_ratio", .9)):
            changed = dict(diag, **{key: value})
            with self.assertRaises(ValueError): EXP.check_reference20(reference, reference, changed, diag)

    def test_reference_alignment_order_labels_names_and_configuration_are_checked(self):
        meta = self.metadata()
        embedded = {key: meta[key].to_numpy(dtype=str) for key in ("sample_id", "participant_id", "task")}
        matrix = np.ones((len(meta), EXP.DIM), np.float32)
        identity = {"mode": "completo", "sample_ids": meta.sample_id.tolist(), "configuration": EXP.EXP20.configuration(EXP.MODEL_REVISION, False, "cuda")}
        args = (meta, meta.copy(), matrix, embedded, EXP.EXP20.feature_names(), identity)
        EXP.validate_reference(*args, real=False)
        changed = meta.copy(); changed.loc[0, "y"] = 0
        with self.assertRaises(ValueError): EXP.validate_reference(meta, changed, *args[2:], real=False)
        changed = {**embedded, "sample_id": embedded["sample_id"][::-1]}
        with self.assertRaises(ValueError): EXP.validate_reference(meta, meta, matrix, changed, *args[4:], real=False)
        identity = json.loads(json.dumps(identity)); identity["configuration"]["max_chunk_samples"] = 42
        with self.assertRaises(ValueError): EXP.validate_reference(*args[:-1], identity, real=False)

    def test_cache_binds_ids_vectors_diagnostics_and_signature(self):
        meta = self.metadata()
        with TemporaryDirectory() as tmp:
            out = Path(tmp); con = self.filled_cache(out, "signature", meta)
            found = EXP.validate_cache(con, meta)
            np.testing.assert_array_equal(found["A004"][0], np.ones((3, EXP.DIM), np.float32))
            con.execute("UPDATE features SET diagnostics=? WHERE sample_id='A004'", ('{}',)); con.commit()
            with self.assertRaises(ValueError): EXP.validate_cache(con, meta)
            con.close()
            with self.assertRaises(ValueError): EXP.init_cache(out / "checkpoint_camadas_ssl.sqlite", "another")
        with TemporaryDirectory() as tmp:
            out = Path(tmp); con = self.filled_cache(out, "signature", meta)
            con.execute("UPDATE features SET vectors=? WHERE sample_id='A004'", (b'broken',)); con.commit()
            with self.assertRaises(ValueError): EXP.validate_cache(con, meta)
            con.close()

    def test_export_separate_layers_order_dtype_and_all_hashes(self):
        meta = self.metadata(); ident = self.identity(meta)
        with TemporaryDirectory() as tmp:
            out = Path(tmp); con = self.filled_cache(out, "signature", meta)
            found = EXP.validate_cache(con, meta); con.close()
            EXP.EXP20.atomic_text(out / "assinatura_execucao.json", EXP.EXP20.json_text(ident))
            EXP.EXP20.atomic_text(out / "PROTOCOLO.txt", EXP.protocol(ident))
            EXP.export(out, meta, found, ident, 1., 2, {})
            with np.load(out / EXP.EXPORTS[0], allow_pickle=False) as z:
                np.testing.assert_array_equal(z["sample_id"], meta.sample_id)
                np.testing.assert_array_equal(z["hidden_layer_indices"], [5, 12, 16])
                for layer in EXP.LAYERS:
                    x = z[f"embeddings_layer{layer:02d}"]
                    self.assertEqual(x.shape, (2, EXP.DIM)); self.assertEqual(x.dtype, np.float32)
                    np.testing.assert_array_equal(x[:, 0], [0, 1])
                    np.testing.assert_array_equal(z[f"feature_names_layer{layer:02d}"], EXP.feature_names(layer))
            marker = json.loads((out / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
            self.assertEqual(set(marker), set(EXP.EXPORTS))
            self.assertTrue(all(EXP.EXP20.sha(out/name) == digest for name, digest in marker.items()))
            with self.assertRaises(ValueError): EXP.export(out, meta, {"A004": found["A004"]}, ident, 1, 1, {})

    def test_complete_resume_checks_integrity_without_encoder_or_wav(self):
        meta = self.metadata(); ident = self.identity(meta)
        text = EXP.EXP20.json_text(ident); signature = hashlib.sha256(text.encode("utf8")).hexdigest()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); out = root / "new"; out.mkdir()
            con = self.filled_cache(out, signature, meta); found = EXP.validate_cache(con, meta); con.close()
            EXP.EXP20.atomic_text(out / "assinatura_execucao.json", text)
            EXP.EXP20.atomic_text(out / "PROTOCOLO.txt", EXP.protocol(ident))
            EXP.export(out, meta, found, ident, 1, 2, {})
            before = {name: (EXP.EXP20.sha(out/name), (out/name).stat().st_mtime_ns) for name in EXP.EXPORTS}
            with patch.object(EXP, "previous_campaign_complete", return_value={}), patch.object(EXP.EXP20, "load_encoder", side_effect=AssertionError("no encoder")), patch.object(EXP.EXP18, "read_signal48", side_effect=AssertionError("no WAV")):
                EXP.run(root, out, meta, {}, ident, Path("unused"), False, "cuda")
            self.assertEqual(before, {name: (EXP.EXP20.sha(out/name), (out/name).stat().st_mtime_ns) for name in EXP.EXPORTS})
            (out / EXP.EXPORTS[0]).write_bytes(b"tampered")
            with patch.object(EXP, "previous_campaign_complete", return_value={}), patch.object(EXP.EXP20, "load_encoder", side_effect=AssertionError("no encoder")):
                with self.assertRaises(ValueError): EXP.run(root, out, meta, {}, ident, Path("unused"), False, "cuda")

    def test_guard_refuses_active_previous_campaign_before_creating_output(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); out = root / "output"
            ident = self.identity(self.metadata())
            with patch.object(EXP.EXP20, "load_encoder", side_effect=AssertionError("no encoder")):
                with self.assertRaises(ValueError): EXP.run(root, out, self.metadata(), {}, ident, Path("unused"), False, "cuda")
            self.assertFalse(out.exists())
            status = root / "results/experiments/campanha_25_26_execucao/STATUS.json"
            status.parent.mkdir(parents=True); status.write_text(json.dumps({"phase": "executar26"}), encoding="utf8")
            with self.assertRaises(ValueError): EXP.previous_campaign_complete(root)


if __name__ == "__main__":
    unittest.main()
