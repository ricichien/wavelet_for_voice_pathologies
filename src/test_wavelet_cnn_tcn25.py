# -*- coding: utf-8 -*-
"""Testes sintéticos do25; nenhum WAV, rótulo ou ajuste AVFAD."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp25_tests", HERE / "25_treinar_wavelet_cnn_tcn_avfad.py")
EXP25 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP25)


class TemporalNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        EXP25.configure_torch("cpu", threads=1)
        rng = np.random.default_rng(27)
        cls.meta = pd.DataFrame([{"sample_id": f"S{i:03d}_{j}", "participant_id": f"S{i:03d}",
                                  "task": str(j).zfill(3), "y": int(i % 4 != 0)}
                                 for i in range(40) for j in range(2 + int(i % 5 != 0))])
        n = len(cls.meta)
        cls.inputs = {"raw": rng.normal(size=(n, 480, 8)).astype(np.float32),
                      "rms": rng.normal(size=(n, 480, 8)).astype(np.float32),
                      "summaries_rms": rng.normal(size=(n, 1920)).astype(np.float32)}
        cls.train, cls.test = next(EXP25.EXP13.row_splits(cls.meta, 5, 20261105))
        cls.options = {**EXP25.DEFAULT_OPTIONS, "max_epochs": 2, "min_epochs": 1,
                       "patience": 1, "batch_people": 8, "steps": 8}
        cls.result = EXP25.train_arm(cls.inputs, cls.meta, cls.train, cls.test, 20261105, 1,
                                    EXP25.ARMS[0], "cpu", cls.options)

    def compare_fits(self, other):
        for key in ("state", "selection_state"):
            for name in self.result[key]:
                torch.testing.assert_close(self.result[key][name], other[key][name], rtol=0, atol=0)
        for key in self.result["arrays"]:
            np.testing.assert_array_equal(self.result["arrays"][key], other["arrays"][key])
        for key, value in self.result["selection"].items():
            if key != "seconds": self.assertEqual(value, other["selection"][key])
        pd.testing.assert_frame_equal(self.result["frames"]["historico.csv"], other["frames"]["historico.csv"])
        pd.testing.assert_frame_equal(self.result["frames"]["predicoes_treino_gravacao.csv"], other["frames"]["predicoes_treino_gravacao.csv"])

    def test_interpolation_endpoints_identity_and_single_window(self):
        x = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
        result = EXP25.interpolate_sequence(x, 5)
        np.testing.assert_array_equal(result[:, 0], x[0])
        np.testing.assert_array_equal(result[:, -1], x[-1])
        np.testing.assert_array_equal(result[:, 1], .5 * (x[0] + x[1]))
        np.testing.assert_array_equal(EXP25.interpolate_sequence(x, 3), x.T)
        np.testing.assert_array_equal(EXP25.interpolate_sequence(x[:1], 5), np.repeat(x[:1].T, 5, axis=1))
        with self.assertRaises(ValueError): EXP25.interpolate_sequence(np.full((3, 4), np.nan), 5)

    def test_patient_loss_is_invariant_to_duplicating_its_recordings(self):
        logits = torch.tensor([-.5, .5, 3.], requires_grad=True)
        original = EXP25.person_logits(logits, [2, 1])
        repeated = EXP25.person_logits(torch.tensor([-.5, .5, -.5, .5, 3., 3.]), [4, 2])
        torch.testing.assert_close(original, repeated, rtol=0, atol=0)
        original.sum().backward()
        torch.testing.assert_close(logits.grad, torch.tensor([.5, .5, 1.]), rtol=0, atol=0)

    def test_models_are_compact_and_recording_independent_in_eval(self):
        for arm in EXP25.ARMS:
            model = EXP25.model_for(arm).eval()
            self.assertLess(sum(p.numel() for p in model.parameters()), 80000)
            self.assertFalse(any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules()))
            x = torch.tensor(self.inputs[EXP25.ARM_INPUT[arm]][:3])
            with torch.inference_mode():
                self.assertEqual(tuple(model.encode(x).shape), (3, 32))
                a, b = model(x)[0], model(x[:1])[0]
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)

    def test_raw_RMS_initialization_is_paired_within_family(self):
        for a, b in ((EXP25.ARMS[0], EXP25.ARMS[1]), (EXP25.ARMS[2], EXP25.ARMS[3])):
            seed = EXP25.arm_seed(20261105, 1, a)
            self.assertEqual(seed, EXP25.arm_seed(20261105, 1, b))
            EXP25.seed_everything(seed)
            left = EXP25.model_for(a)
            EXP25.seed_everything(seed)
            right = EXP25.model_for(b)
            self.assertEqual(EXP25.state_digest(left.state_dict()), EXP25.state_digest(right.state_dict()))

    def test_outer_features_cannot_change_preprocessing_epochs_or_model(self):
        changed = {key: value.copy() for key, value in self.inputs.items()}
        for value in changed.values(): value[self.test] = 1e5
        other = EXP25.train_arm(changed, self.meta, self.train, self.test, 20261105, 1, EXP25.ARMS[0], "cpu", self.options)
        self.compare_fits(other)

    def test_outer_labels_cannot_change_preprocessing_epochs_or_model(self):
        changed = self.meta.copy()
        changed.loc[self.test, "y"] = 1 - changed.loc[self.test, "y"]
        other = EXP25.train_arm(self.inputs, changed, self.train, self.test, 20261105, 1, EXP25.ARMS[0], "cpu", self.options)
        self.compare_fits(other)

    def test_internal_holdout_cannot_fit_its_own_scaler(self):
        fit, validation = EXP25.internal_split(self.meta, self.train, 20261105, 1)
        x = self.inputs["raw"].copy()
        original = EXP25.fit_scaler(x, fit)
        x[validation] = 1e8
        altered = EXP25.fit_scaler(x, fit)
        for key in original: np.testing.assert_array_equal(original[key], altered[key])
        flattened = self.inputs["raw"][fit].transpose(0, 2, 1).reshape(-1, 480).astype(np.float64)
        np.testing.assert_allclose(original["mean"], flattened.mean(axis=0), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(original["scale"], flattened.std(axis=0), rtol=1e-12, atol=1e-12)

    def test_all_patient_groups_are_disjoint(self):
        fit, validation = EXP25.internal_split(self.meta, self.train, 20261105, 1)
        groups = [set(self.meta.iloc[rows].participant_id) for rows in (fit, validation, self.test)]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        EXP25.validate_result(self.result, self.meta, self.train, self.test, 20261105, 1, EXP25.ARMS[0], self.options)
        with self.assertRaises(ValueError):
            EXP25.train_arm(self.inputs, self.meta, np.append(self.train, self.test[0]), self.test,
                            20261105, 1, EXP25.ARMS[0], "cpu", self.options)

    def test_corrupt_epoch_and_person_aggregation_are_rejected(self):
        other = copy.deepcopy(self.result)
        other["selection"]["selected_epoch"] = 999
        with self.assertRaises(ValueError):
            EXP25.validate_result(other, self.meta, self.train, self.test, 20261105, 1, EXP25.ARMS[0], self.options)
        other = copy.deepcopy(self.result)
        other["frames"]["predicoes.csv"]["score"] += 1e-3
        with self.assertRaises(AssertionError):
            EXP25.validate_result(other, self.meta, self.train, self.test, 20261105, 1, EXP25.ARMS[0], self.options)

    def test_checkpoint_roundtrip_and_corruption(self):
        with TemporaryDirectory(prefix="avfad25_test_") as temporary:
            destination = Path(temporary) / "arm"
            EXP25.save_checkpoint(destination, self.result, "synthetic")
            loaded = EXP25.load_checkpoint(destination, "synthetic")
            EXP25.validate_result(loaded, self.meta, self.train, self.test, 20261105, 1, EXP25.ARMS[0], self.options)
            for key in self.result["state"]:
                torch.testing.assert_close(loaded["state"][key], self.result["state"][key], rtol=0, atol=0)
            with self.assertRaises(ValueError): EXP25.load_checkpoint(destination, "other")
            with (destination / "modelo_refit.pt").open("ab") as stream: stream.write(b"tamper")
            with self.assertRaises(ValueError): EXP25.load_checkpoint(destination, "synthetic")

    def test_memmap_interpolation_cache_is_label_blind_and_checks_hashes(self):
        with TemporaryDirectory(prefix="avfad25_cache_") as temporary:
            root = Path(temporary)
            data = root / "data"; data.mkdir()
            meta = pd.DataFrame({"sample_id": ["A", "B"], "y": [0, 1]})
            offsets = np.asarray([0, 2, 5], dtype=np.int64)
            raw = np.arange(5 * 480, dtype=np.float32).reshape(5, 480)
            np.savez_compressed(data / "sequencias_wpt.npz", data_raw=raw, data_rms=raw * .1)
            cache = root / "checkpoints/cache25"
            with patch.object(EXP25, "ROOT", root):
                original = EXP25.interpolation_cache(data, meta, offsets, 8, cache)
                np.testing.assert_array_equal(original["raw"][1], EXP25.interpolate_sequence(raw[2:], 8))
                changed = meta.copy(); changed.y = 1 - changed.y
                again = EXP25.interpolation_cache(data, changed, offsets, 8, cache)
                np.testing.assert_array_equal(again["rms"], original["rms"])
                del original, again
                with (cache / "raw_8.npy").open("ab") as stream: stream.write(b"tamper")
                with self.assertRaises(ValueError): EXP25.interpolation_cache(data, meta, offsets, 8, cache)


if __name__ == "__main__":
    unittest.main()
