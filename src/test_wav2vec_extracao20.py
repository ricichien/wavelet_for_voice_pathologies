# -*- coding: utf-8 -*-
"""Verificacoes numericas e de integridade20, sem torch/modelo/WAV/rede."""
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location("exp20_test", Path(__file__).with_name("20_extrair_wav2vec_avfad.py"))
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


class Extraction20Tests(unittest.TestCase):
    def test_encoder_rejects_missing_weights_and_unknown_heads(self):
        EXP.validate_loading_info({"unexpected_keys": ["quantizer.codevectors", "project_q.weight", "wav2vec2.project_hid.bias"]})
        for info in ({"missing_keys": ["encoder.layers.0.weight"]},
                     {"mismatched_keys": ["feature_extractor.weight"]},
                     {"unexpected_keys": ["lm_head.weight"]},
                     {"unexpected_keys": ["wav2vec2.encoder.unknown"]}):
            with self.assertRaises(RuntimeError):
                EXP.validate_loading_info(info)

    def test_model_origin_binds_revision_and_every_file(self):
        revision = "a" * 40
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            config = {"model_type": "wav2vec2", "hidden_size": 1024, "num_hidden_layers": 24}
            preprocessor = {"sampling_rate": 16000, "do_normalize": True}
            EXP.atomic_text(out / "config.json", EXP.json_text(config))
            EXP.atomic_text(out / "preprocessor_config.json", EXP.json_text(preprocessor))
            (out / "pytorch_model.bin").write_bytes(b"synthetic")
            (out / "README.md").write_bytes(b"synthetic model")
            files = [p for p in out.iterdir()]
            provenance = {"model_id": EXP.MODEL_ID, "model_revision": revision,
                          "files": {p.name: {"sha256": EXP.sha(p), "size_bytes": p.stat().st_size} for p in files}}
            EXP.atomic_text(out / "ORIGEM_MODELO.json", EXP.json_text(provenance))
            selected, safe = EXP.model_files(out, revision)
            self.assertFalse(safe)
            self.assertIn(out / "ORIGEM_MODELO.json", selected)
            self.assertIn(out / "README.md", selected)
            with self.assertRaises(ValueError):
                EXP.model_files(out, "b" * 40)
            (out / "pytorch_model.bin").write_bytes(b"tampered!")
            with self.assertRaises(ValueError):
                EXP.model_files(out, revision)

    def test_chunks_preserve_all_samples_without_short_tail(self):
        for n in (400, 401, 127999, 128000, 128001, 256001, 16000 * 277):
            x = np.arange(n, dtype=np.float32)
            chunks = EXP.split_chunks(x)
            np.testing.assert_array_equal(np.concatenate(chunks), x)
            self.assertTrue(all(400 <= len(c) <= 128000 for c in chunks))
            self.assertLessEqual(max(map(len, chunks)) - min(map(len, chunks)), 1)
            self.assertEqual(len(chunks), (n + 127999) // 128000)
        with self.assertRaises(ValueError):
            EXP.split_chunks(np.zeros(399))

    def test_frame_pooling_weights_frame_counts_not_chunks(self):
        rng = np.random.default_rng(20)
        frames = rng.normal(100, 0.5, size=(131, 7))
        pooled = EXP.FrameMoments(7)
        for block in (frames[:1], frames[1:118], frames[118:]):
            pooled.update(block)
        expected = np.concatenate([frames.mean(axis=0), frames.std(axis=0)]).astype(np.float32)
        np.testing.assert_array_equal(pooled.vector(), expected)
        self.assertEqual(pooled.count, len(frames))
        pooled.update(np.ones((1, 7)) * 100)
        self.assertEqual(pooled.count, len(frames) + 1)

    def test_pooling_rejects_nonfinite_empty_wrong_dimension(self):
        pooled = EXP.FrameMoments(3)
        with self.assertRaises(ValueError):
            pooled.vector()
        for x in (np.zeros((0, 3)), np.zeros((2, 4)), np.full((1, 3), np.nan)):
            with self.assertRaises(ValueError):
                pooled.update(x)

    def test_cache_rejects_signature_and_vector_tampering(self):
        meta = pd.DataFrame({"sample_id": ["A004"], "participant_id": ["A"], "task": ["004"]})
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.sqlite"
            con = EXP.init_cache(path, "a")
            vector = np.arange(EXP.DIM, dtype=np.float32)
            blob = vector.tobytes()
            con.execute("INSERT INTO features VALUES (?,?,?,?,?,?)", ("A004", "A", "004", blob, hashlib.sha256(blob).hexdigest(), "{}"))
            con.commit()
            np.testing.assert_array_equal(EXP.validate_cache(con, meta)["A004"][0], vector)
            con.execute("UPDATE features SET vector=?", (b"broken",)); con.commit()
            with self.assertRaises(ValueError):
                EXP.validate_cache(con, meta)
            con.close()
            with self.assertRaises(ValueError):
                EXP.init_cache(path, "b")

    def test_export_alignment_and_hashed_signature(self):
        meta = pd.DataFrame({"sample_id": ["B004", "A004"], "participant_id": ["B", "A"], "task": ["004", "004"], "y": [1, 0]})
        diag = {"hidden_frames": 3, "chunk_count": 1, "vad_fallback": 0}
        found = {"A004": (np.ones(EXP.DIM, dtype=np.float32), diag),
                 "B004": (np.zeros(EXP.DIM, dtype=np.float32), diag)}
        ident = {"mode": "synthetic", "configuration": {"model_revision": "a" * 40}}
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            EXP.atomic_text(out / "assinatura_execucao.json", EXP.json_text(ident))
            EXP.export(out, meta, found, ident, 1, 1, 2, {})
            with np.load(out / EXP.EXPORTS[0], allow_pickle=False) as data:
                np.testing.assert_array_equal(data["sample_id"], ["B004", "A004"])
                np.testing.assert_array_equal(data["embeddings"][:, 0], [0, 1])
                self.assertEqual(data["embeddings"].dtype, np.float32)
            hashes = json.loads((out / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf-8"))
            self.assertEqual(set(hashes), set(EXP.EXPORTS))
            self.assertTrue(all(EXP.sha(out / name) == digest for name, digest in hashes.items()))
            with self.assertRaises(ValueError):
                EXP.export(out, meta, {"A004": found["A004"]}, ident, 1, 1, 1, {})


if __name__ == "__main__":
    unittest.main()
