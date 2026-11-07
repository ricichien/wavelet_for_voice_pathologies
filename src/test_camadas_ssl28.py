# -*- coding: utf-8 -*-
"""Isolamento e replay sintéticos do experimento28; sem áudio ou ajuste AVFAD."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp28_test", HERE / "28_avaliar_camadas_ssl_wavelet_avfad.py")
E = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(E)


class LayerSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(28161105)
        cls.meta = pd.DataFrame([{"participant_id": f"S{p:03d}", "sample_id": f"S{p:03d}_{task}",
                                 "task": task, "y": int(p % 4 != 0), "matrix_row": p*3+t}
                                for p in range(36) for t, task in enumerate(("004", "005", "006"))])
        cls.matrices = {"wpt": rng.normal(size=(len(cls.meta), 1920)).astype(np.float32)}
        cls.names = {"wpt": np.asarray([f"WPT__sym5__TEST__{i:04d}" for i in range(1920)])}
        for layer in E.LAYERS:
            cls.matrices[f"ssl{layer:02d}"] = rng.normal(size=(len(cls.meta), 2048)).astype(np.float32)
            cls.names[f"ssl{layer:02d}"] = E.EXP21.embedding_names() if layer == 24 else E.intermediate_names(layer)
        cls.train, cls.test = next(E.EXP13.row_splits(cls.meta, 5, 20261105))
        cls.grid = (.001, 1.)
        cls.frames, cls.arrays = E.evaluate_fold(cls.matrices, cls.meta, cls.names, cls.train, cls.test, 20261105, 1, c_grid=cls.grid)

    def assert_same_fitted(self, frames, arrays):
        self.assertEqual(set(arrays), set(self.arrays))
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, self.arrays[key], err_msg=key)
        for key in ("candidatos_internos.csv", "predicoes_oof_internas.csv", "particoes_internas.csv", "selecoes.csv", "atributos.csv"):
            pd.testing.assert_frame_equal(frames[key], self.frames[key])

    def test_outer_features_cannot_change_fit_or_search(self):
        changed = {key: value.copy() for key, value in self.matrices.items()}
        for matrix in changed.values():
            matrix[self.test] = 2e5 + matrix[self.test]*300.
        frames, arrays = E.evaluate_fold(changed, self.meta, self.names, self.train, self.test, 20261105, 1, c_grid=self.grid)
        self.assert_same_fitted(frames, arrays)
        for key in ("predicoes_treino_participante.csv", "predicoes_treino_gravacao.csv"):
            pd.testing.assert_frame_equal(frames[key], self.frames[key])

    def test_outer_labels_cannot_change_fit_search_or_margins(self):
        changed = self.meta.copy(); changed.loc[self.test, "y"] = 1-changed.loc[self.test, "y"]
        frames, arrays = E.evaluate_fold(self.matrices, changed, self.names, self.train, self.test, 20261105, 1, c_grid=self.grid)
        self.assert_same_fitted(frames, arrays)
        for key in ("predicoes.csv", "predicoes_teste_gravacao.csv", "predicoes_treino_participante.csv"):
            np.testing.assert_array_equal(frames[key].score, self.frames[key].score)

    def test_inner_validation_cannot_fit_its_own_preprocessors(self):
        tr, va = next(E.EXP13.row_splits(self.meta.iloc[self.train].reset_index(drop=True), 3, 20261206))
        tr, va = self.train[tr], self.train[va]
        y = self.meta.y.to_numpy(); _, _, before = E.prepare(self.matrices, y, tr, va)
        changed = {key: value.copy() for key, value in self.matrices.items()}
        for matrix in changed.values(): matrix[va] = matrix[va]*1000+1e5
        ynew = y.copy(); ynew[va] = 1-ynew[va]
        _, _, after = E.prepare(changed, ynew, tr, va)
        for key in before: np.testing.assert_array_equal(after[key], before[key])

    def test_ties_use_lower_C_then_predeclared_layer_order(self):
        candidates = pd.DataFrame([{"candidate_id": E.candidate_id(rep, c), "representation": rep,
                                    "C": c, "layer": E.layer_of(rep), "inner_BA": .6}
                                   for rep in E.ARM_REPS[E.ARMS[6]] for c in E.C_GRID])
        selected = E.choose(candidates, E.ARM_REPS[E.ARMS[6]])
        self.assertEqual(selected.C, 1e-4); self.assertEqual(selected.layer, 24)
        candidates.loc[candidates.layer == 5, "inner_BA"] += .5e-12
        self.assertEqual(E.choose(candidates, E.ARM_REPS[E.ARMS[6]]).layer, 24)
        candidates.loc[candidates.layer == 5, "inner_BA"] += 1e-7
        self.assertEqual(E.choose(candidates, E.ARM_REPS[E.ARMS[6]]).layer, 5)

    def test_all_partitions_candidates_and_fixed_choices_validate(self):
        E.validate(self.frames, self.arrays, self.meta, self.train, self.test, self.names, 20261105, 1, c_grid=self.grid)
        inner = self.frames["predicoes_oof_internas.csv"]
        self.assertEqual(inner.candidate_id.nunique(), 12*len(self.grid))
        self.assertFalse(inner.duplicated(["candidate_id", "participant_id"]).any())
        self.assertFalse(set(inner.participant_id) & set(self.meta.iloc[self.test].participant_id))
        for row in self.frames["selecoes.csv"].query("choice_source == 'fixed_protocol'").itertuples():
            self.assertEqual(row.representation, E.FIXED[row.arm]); self.assertEqual(row.C, 1.)

    def test_tampered_choice_and_inner_scores_are_rejected(self):
        for corruption in ("choice", "score", "nan"):
            frames = {key: value.copy(deep=True) for key, value in self.frames.items()}
            if corruption == "choice": frames["selecoes.csv"].loc[5, "C"] = 42.
            else:
                index = frames["predicoes_oof_internas.csv"].index[0]
                original = frames["predicoes_oof_internas.csv"].loc[index, "score"]
                frames["predicoes_oof_internas.csv"].loc[index, "score"] = -original if corruption == "score" else np.nan
            with self.subTest(corruption=corruption), self.assertRaises((ValueError, AssertionError)):
                E.validate(frames, self.arrays, self.meta, self.train, self.test, self.names, 20261105, 1, c_grid=self.grid)

    def test_participant_overlap_is_rejected_before_any_fit(self):
        with self.assertRaises(ValueError):
            E.evaluate_fold(self.matrices, self.meta, self.names, np.append(self.train, self.test[0]), self.test, 20261105, 1, c_grid=self.grid)

    def test_fixed_controls_reproduce21_and_reg24_reproduces22(self):
        matrices21 = {"wpt": self.matrices["wpt"], "wav2vec": self.matrices["ssl24"]}
        old, _ = E.EXP21.fit_fold_models(matrices21, self.meta.y.to_numpy(), self.train, self.test)
        actual_records = self.frames["predicoes_teste_gravacao.csv"]
        for arm, old_arm in zip(E.ARMS[:2], E.EXP21.ARMS[:2]):
            actual = actual_records[actual_records.arm == arm].score.to_numpy()
            np.testing.assert_array_equal(actual, old[old_arm][1])
        frames22, arrays22 = E.EXP22.evaluate_fold(matrices21, self.meta,
                                               {"wpt": self.names["wpt"], "wav2vec": self.names["ssl24"]},
                                               self.train, self.test, 20261105, 1, c_grid=self.grid)
        chosen28 = self.frames["selecoes.csv"].query("arm == 'F_L24_C128_REG'").iloc[0]
        chosen22 = frames22["selecoes.csv"].query("arm == 'B_WAV2VEC_REG'").iloc[0]
        self.assertEqual(chosen28.C, chosen22.C)
        np.testing.assert_array_equal(actual_records.query("arm == 'F_L24_C128_REG'").score,
                                      frames22["predicoes_teste_gravacao.csv"].query("arm == 'B_WAV2VEC_REG'").score)
        for key in ("coef", "intercept", "support_indices", "support_original_rows", "class_weight"):
            np.testing.assert_array_equal(self.arrays[f"F_L24_C128_REG__{key}"], arrays22[f"B_WAV2VEC_REG__{key}"])

    def test_hybrid_and_budget_have_expected_scaling(self):
        train_values, _, _ = E.prepare(self.matrices, self.meta.y.to_numpy(), self.train, self.test)
        for layer in E.LAYERS:
            actual = E.represented(train_values, f"HYB128_L{layer:02d}")
            expected = np.concatenate([train_values["WPT128"], train_values[f"SSL128_L{layer:02d}"]], axis=1)/np.sqrt(2.)
            np.testing.assert_array_equal(actual, expected)
            for rep in (f"HYB128_L{layer:02d}", f"SSL256_L{layer:02d}"):
                self.assertEqual(E.represented(train_values, rep).shape[1], 256)
                self.assertAlmostEqual(float(np.mean(np.square(E.represented(train_values, rep)).sum(axis=1))), 128., places=4)

    def test_checkpoint_replay_and_tamper_detection(self):
        with TemporaryDirectory(prefix="avfad28_synthetic_") as temp:
            out = Path(temp)/"fold"
            E.save_checkpoint(out, self.frames, self.arrays, "synthetic")
            frames, arrays = E.load_checkpoint(out, "synthetic")
            E.validate(frames, arrays, self.meta, self.train, self.test, self.names, 20261105, 1, c_grid=self.grid)
            for key, value in arrays.items(): np.testing.assert_array_equal(value, self.arrays[key])
            with self.assertRaises(ValueError): E.load_checkpoint(out, "other")
            with (out/"candidatos_internos.csv").open("ab") as stream: stream.write(b"\n")
            with self.assertRaises(ValueError): E.load_checkpoint(out, "synthetic")

    def test_multilayer_ID_label_configuration_and_control_contract(self):
        ident = {"mode": "completo", "configuration": {"model_id": E.EXP21.MODEL_ID,
                "model_revision": "1a640f32ac3e39899438a2931f9924c02f080a54", "frozen": True,
                "sample_rate": 16000, "hidden_layer": "multiple_intermediate", "hidden_layers": [5,12,16],
                "feature_dimension": 6144, "feature_dimension_per_layer": 2048, "pooling": "mean_std",
                "num_hidden_layers": 24, "hidden_size": 1024, "control_layer": 24,
                "control_reference": "complete_exp20_same_recording", "intermediate_final_layer_norm": False,
                "control_atol": 2e-5, "control_rtol": 2e-5, "output_hidden_states": True}}
        meta27 = self.meta.copy()
        meta27["control24_allclose"], meta27["control24_exact"] = True, True
        meta27["hidden_frames"], meta27["control24_reference_frames"] = 100, 100
        meta27["control24_max_abs_error"], meta27["control24_mean_abs_error"] = 0., 0.
        embedded = {key: self.meta[key].to_numpy() for key in ("sample_id", "participant_id", "task")}
        E.validate_multilayer(self.matrices, self.meta, self.names, meta27, embedded, ident, np.asarray([5,12,16]), real=False)
        for corruption in ("IDs", "labels", "control", "encoder", "layers", "names"):
            changed = meta27.copy(); config = copy.deepcopy(ident); ids = {key:value.copy() for key,value in embedded.items()}; names = {key:value.copy() for key,value in self.names.items()}
            indices = np.asarray([5,12,16])
            if corruption == "IDs": ids["sample_id"] = ids["sample_id"][::-1]
            elif corruption == "labels": changed.loc[0,"y"] = 1-changed.loc[0,"y"]
            elif corruption == "control": changed.loc[0,"control24_allclose"] = False
            elif corruption == "encoder": config["configuration"]["frozen"] = False
            elif corruption == "layers": indices = np.asarray([5,12,18])
            elif corruption == "names": names["ssl05"] = names["ssl05"][::-1]
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                E.validate_multilayer(self.matrices, self.meta, names, changed, ids, config, indices, real=False)

    def test_recording_person_mean_margins_are_enforced(self):
        frames = {key:value.copy(deep=True) for key,value in self.frames.items()}
        frames["predicoes.csv"]["score"] += 1e-5
        for arm in E.ARMS:
            mask = (frames["metricas.csv"].arm==arm)&(frames["metricas.csv"].phase=="teste")&(frames["metricas.csv"].unit=="participante")
            source = frames["predicoes.csv"].query("arm == @arm")
            for key,value in E.EXP17.measures(source).items(): frames["metricas.csv"].loc[mask,key] = value
        with self.assertRaises(AssertionError): E.validate(frames,self.arrays,self.meta,self.train,self.test,self.names,20261105,1,c_grid=self.grid)


if __name__ == "__main__":
    unittest.main()
