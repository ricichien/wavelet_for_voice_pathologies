# -*- coding: utf-8 -*-
"""Contratos e isolamento de folds do21; somente dados sinteticos."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp21_under_test", HERE / "21_avaliar_wpt_wav2vec_avfad.py")
EXP21 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP21)


class HybridExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(21161105)
        cls.meta = pd.DataFrame([{"participant_id": f"S{p:03d}", "sample_id": f"S{p:03d}_{task}",
                                  "task": task, "y": int(p % 4 != 0), "matrix_row": p * 3 + t}
                                 for p in range(36) for t, task in enumerate(("004", "005", "006"))])
        cls.matrices = {"wpt": rng.normal(size=(len(cls.meta), 1920)).astype("float32"),
                        "wav2vec": rng.normal(size=(len(cls.meta), 2048)).astype("float32")}
        cls.names = {"wpt": np.asarray([f"WPT__sym5__TEST__{j:04d}" for j in range(1920)]),
                     "wav2vec": EXP21.embedding_names()}
        cls.embedded = {key: cls.meta[key].to_numpy(dtype=str) for key in ("sample_id", "participant_id", "task")}
        cls.train, cls.test = next(EXP21.EXP13.row_splits(cls.meta, 5, 20261105))
        cls.frames, cls.parameters = EXP21.evaluate_fold(cls.matrices, cls.meta, cls.names, cls.train, cls.test,
                                                        20261105, 1, lambda _: None)

    def test_alignment_rejects_independent_identity_and_label_corruption(self):
        EXP21.validate_alignment(self.matrices, self.meta, self.names, self.meta, self.embedded, real=False)
        for kind in ("order", "label", "sample_id", "embedded", "duplicate", "row_index", "names"):
            other = self.meta.copy(deep=True)
            embedded = {key: value.copy() for key, value in self.embedded.items()}
            names = {key: value.copy() for key, value in self.names.items()}
            if kind == "order":
                other = other.iloc[::-1].reset_index(drop=True)
            elif kind == "label":
                other.loc[other.participant_id == "S000", "y"] = 1
            elif kind == "sample_id":
                other.loc[0, "sample_id"] = "OUTRO"
            elif kind == "embedded":
                embedded["participant_id"][0] = "OUTR"
            elif kind == "duplicate":
                other.loc[0, "sample_id"] = other.sample_id.iloc[1]
            elif kind == "row_index":
                other.loc[0, "matrix_row"] = 20
            else:
                names["wav2vec"] = names["wav2vec"][::-1]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                EXP21.validate_alignment(self.matrices, self.meta, names, other, embedded, real=False)

    def test_encoder_checkpoint_and_pooling_must_be_fixed(self):
        correct = {"model_id": EXP21.MODEL_ID, "sample_rate": 16000, "frozen": True,
                   "hidden_layer": "last", "pooling": "mean_std", "hidden_size": 1024,
                   "model_revision": "a" * 40}
        EXP21.validate_model_configuration({"configuration": correct})
        for key, invalid in (("model_revision", "main"), ("frozen", False), ("frozen", 1),
                             ("hidden_layer", 12), ("pooling", "mean"), ("sample_rate", 48000)):
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                EXP21.validate_model_configuration({"configuration": {**correct, key: invalid}})

    def test_test_features_cannot_change_any_fitted_parameter_or_training_prediction(self):
        changed = {key: value.copy() for key, value in self.matrices.items()}
        for i, matrix in enumerate(changed.values()):
            matrix[self.test] = (2e5 + i * 5e4) + matrix[self.test] * 800
        frames, parameters = EXP21.evaluate_fold(changed, self.meta, self.names, self.train, self.test,
                                                 20261105, 1, lambda _: None)
        self.assertEqual(set(parameters), set(self.parameters))
        for key in parameters:
            np.testing.assert_array_equal(parameters[key], self.parameters[key], err_msg=key)
        pd.testing.assert_frame_equal(frames["atributos.csv"], self.frames["atributos.csv"])
        pd.testing.assert_frame_equal(frames["predicoes_treino_participante.csv"], self.frames["predicoes_treino_participante.csv"])
        pd.testing.assert_frame_equal(frames["predicoes_treino_gravacao.csv"], self.frames["predicoes_treino_gravacao.csv"])

    def test_test_labels_cannot_change_models_or_scores(self):
        changed_meta = self.meta.copy(deep=True)
        changed_meta.loc[self.test, "y"] = 1 - changed_meta.loc[self.test, "y"]
        frames, parameters = EXP21.evaluate_fold(self.matrices, changed_meta, self.names, self.train, self.test,
                                                 20261105, 1, lambda _: None)
        for key in parameters:
            np.testing.assert_array_equal(parameters[key], self.parameters[key], err_msg=key)
        for filename in ("predicoes.csv", "predicoes_treino_participante.csv",
                         "predicoes_teste_gravacao.csv", "predicoes_treino_gravacao.csv"):
            np.testing.assert_array_equal(frames[filename].score, self.frames[filename].score)
        pd.testing.assert_frame_equal(frames["atributos.csv"], self.frames["atributos.csv"])

    def test_checkpoint_roundtrip_and_tampering(self):
        EXP21.validate(self.frames, self.parameters, self.meta, self.train, self.test, self.names, 20261105, 1)
        with TemporaryDirectory(prefix="avfad21_test_") as temp:
            destination = Path(temp) / "fold"
            EXP21.save_checkpoint(destination, self.frames, self.parameters, "synthetic")
            frames, parameters = EXP21.load_checkpoint(destination, "synthetic")
            EXP21.validate(frames, parameters, self.meta, self.train, self.test, self.names, 20261105, 1)
            for key in parameters:
                np.testing.assert_array_equal(parameters[key], self.parameters[key])
            with self.assertRaises(ValueError):
                EXP21.load_checkpoint(destination, "outro")
            with (destination / "predicoes.csv").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaises(ValueError):
                EXP21.load_checkpoint(destination, "synthetic")

    def test_record_to_participant_aggregation_is_enforced(self):
        frames = {key: value.copy(deep=True) for key, value in self.frames.items()}
        # Todas as decisoes e rankings ficam iguais; uma metrica recalculada
        # isoladamente nao detectaria que a margem deixou de ser a media.
        frames["predicoes.csv"]["score"] += 1e-5
        source = frames["predicoes.csv"]
        for arm in EXP21.ARMS:
            mask = (frames["metricas.csv"].arm == arm) & (frames["metricas.csv"].phase == "teste") & (frames["metricas.csv"].unit == "participante")
            for key, value in EXP21.EXP17.measures(source[source.arm == arm]).items():
                frames["metricas.csv"].loc[mask, key] = value
        with self.assertRaises(AssertionError):
            EXP21.validate(frames, self.parameters, self.meta, self.train, self.test, self.names, 20261105, 1)

    def test_hybrid_uses_the_exact_training_only_preprocessors_of_both_branches(self):
        for modality, arm in zip(EXP21.MODALITIES, EXP21.ARMS[:2]):
            for key in ("imputer", "constant_mask", "selector_scores", "selector_mask", "scaler_mean", "scaler_scale"):
                np.testing.assert_array_equal(self.parameters[f"{arm}__{key}"], self.parameters[f"C_HIBRIDO__{modality}__{key}"])
        self.assertEqual(self.parameters["C_HIBRIDO__support_vectors"].shape[1], 256)
        self.assertEqual(self.frames["atributos.csv"].groupby("arm").size().to_dict(),
                         {"A_WPT": 128, "B_WAV2VEC": 128, "C_HIBRIDO": 256, "D_WAV2VEC_256": 256})

    def test_budget_control_matches_hybrid_dimension_and_energy_scaling(self):
        prefix = "D_WAV2VEC_256__"
        var, sel = self.parameters[prefix + "constant_mask"], self.parameters[prefix + "selector_mask"]
        values = self.matrices["wav2vec"][self.train][:, var][:, sel].astype("float64")
        z = (values - self.parameters[prefix + "scaler_mean"]) / self.parameters[prefix + "scaler_scale"]
        normalized = z / np.sqrt(2.0)
        self.assertEqual(normalized.shape[1], 256)
        self.assertAlmostEqual(float(np.mean(np.square(normalized).sum(axis=1))), 128.0, places=5)
        support = self.parameters[prefix + "support_indices"]
        np.testing.assert_allclose(self.parameters[prefix + "support_vectors"], normalized[support], atol=3e-7, rtol=1e-6)
        # O classificador salvo opera sobre embeddings escalados por sqrt2;
        # verificar suas margens detecta omissao ou duplicacao dessa divisao.
        scores = normalized @ self.parameters[prefix + "coef"].reshape(-1) + self.parameters[prefix + "intercept"][0]
        recorded = self.frames["predicoes_treino_gravacao.csv"].query("arm == 'D_WAV2VEC_256'").score
        np.testing.assert_allclose(scores, recorded, atol=1e-6, rtol=1e-6)

    def test_isolated_branches_keep_the_original_reference_pipeline(self):
        for arm, modality in zip(EXP21.ARMS[:2], EXP21.MODALITIES):
            model = EXP21.EXP17.model_for("A_STANDARD_LINEAR", len(self.train))
            model.fit(self.matrices[modality][self.train], self.meta.y.to_numpy()[self.train])
            scores = model.decision_function(self.matrices[modality][self.test])
            stored = self.frames["predicoes_teste_gravacao.csv"].query("arm == @arm").score
            np.testing.assert_array_equal(scores, stored)
            for key, value in EXP21.EXP17.parameters(model).items():
                np.testing.assert_array_equal(value, self.parameters[f"{arm}__{key}"])

    def test_participant_overlap_is_rejected(self):
        repeated = np.append(self.train, self.test[0])
        with self.assertRaises(ValueError):
            EXP21.evaluate_fold(self.matrices, self.meta, self.names, repeated, self.test, 20261105, 1, lambda _: None)


if __name__ == "__main__":
    unittest.main()
