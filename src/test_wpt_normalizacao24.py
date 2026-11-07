# -*- coding: utf-8 -*-
"""Contratos de alinhamento e isolamento da ablação24; somente dados sintéticos."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp24_under_test", HERE / "24_avaliar_normalizacao_wpt_avfad.py")
EXP24 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXP24)


class Normalization24Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(24161106)
        cls.meta = pd.DataFrame([{"participant_id": f"S{p:03d}", "sample_id": f"S{p:03d}_{task}",
                                  "task": task, "y": int(p % 4 != 0), "matrix_row": 3*p+t}
                                 for p in range(36) for t, task in enumerate(("004", "005", "006"))])
        cls.feature_names = np.asarray([f"WPT__sym5__L6__N{n:03d}__{descriptor}__{stat}"
                                        for stat in ("mean", "std", "q10", "q90") for n in range(96)
                                        for descriptor in ("rel_energy", "log_energy", "entropy", "kurtosis", "zcr")])
        cls.raw = rng.normal(size=(len(cls.meta), 1920)).astype("float32")
        cls.rms = (cls.raw + rng.normal(0, .2, cls.raw.shape)).astype("float32")
        mask = EXP24.no_log_mask(cls.feature_names)
        cls.matrices = dict(zip(EXP24.ARMS, (cls.raw, cls.rms, cls.raw[:, mask])))
        cls.names = dict(zip(EXP24.ARMS, (cls.feature_names, cls.feature_names, cls.feature_names[mask])))
        cls.embedded = {key: cls.meta[key].to_numpy(dtype=str) for key in ("sample_id", "participant_id", "task")}
        cls.train, cls.test = next(EXP24.EXP13.row_splits(cls.meta, 5, 20261105))
        cls.frames, cls.parameters = EXP24.evaluate_fold(cls.matrices, cls.meta, cls.names, cls.train, cls.test, 20261105, 1)

    def test_alignment_and_gain_configuration_reject_corruption(self):
        EXP24.validate_alignment(self.raw, self.meta, self.feature_names, self.rms, self.meta, self.feature_names, self.embedded, real=False)
        configuration = {"sample_rate": 48000, "target_rms": .1, "normalization": "global_active_rms",
                         "vad_before_normalization": True, "wavelet": "sym5", "levels": [5, 6]}
        EXP24.validate_configuration({"configuration": configuration, "mode": "completo"})
        for key, invalid in (("sample_rate", 16000), ("target_rms", 1.), ("normalization", "per_window"), ("vad_before_normalization", False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                EXP24.validate_configuration({"configuration": {**configuration, key: invalid}, "mode": "completo"})
        with self.assertRaises(ValueError): EXP24.validate_configuration({"configuration": configuration, "mode": "piloto_3"})
        for kind in ("order", "label", "embedded", "row", "names"):
            other, embedded, names = self.meta.copy(), {k:v.copy() for k,v in self.embedded.items()}, self.feature_names.copy()
            if kind == "order": other = other.iloc[::-1].reset_index(drop=True)
            elif kind == "label": other.loc[other.participant_id == "S000", "y"] = 1
            elif kind == "embedded": embedded["sample_id"][0] = "OTHER"
            elif kind == "row": other.loc[0, "matrix_row"] = 17
            else: names = names[::-1]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                EXP24.validate_alignment(self.raw, self.meta, self.feature_names, self.rms, other, names, embedded, real=False)

    def test_removing_log_energy_is_exactly_the_whole_384_column_block(self):
        mask = EXP24.no_log_mask(self.feature_names)
        self.assertEqual((~mask).sum(), 384)
        self.assertEqual(mask.sum(), 1536)
        self.assertFalse(any("__log_energy__" in n for n in self.names[EXP24.ARMS[2]]))
        np.testing.assert_array_equal(self.matrices[EXP24.ARMS[2]], self.raw[:, mask])

    def test_external_feature_mutation_changes_no_fitted_parameter_or_training_score(self):
        changed = {key:value.copy() for key,value in self.matrices.items()}
        for value in changed.values(): value[self.test] = 1e6 + value[self.test]*900
        frames, parameters = EXP24.evaluate_fold(changed, self.meta, self.names, self.train, self.test, 20261105, 1)
        for key in parameters: np.testing.assert_array_equal(parameters[key], self.parameters[key], err_msg=key)
        for filename in ("atributos.csv", "predicoes_treino_participante.csv", "predicoes_treino_gravacao.csv"):
            pd.testing.assert_frame_equal(frames[filename], self.frames[filename])

    def test_external_labels_cannot_change_fit_or_scores(self):
        changed = self.meta.copy(); changed.loc[self.test, "y"] = 1-changed.loc[self.test, "y"]
        frames, parameters = EXP24.evaluate_fold(self.matrices, changed, self.names, self.train, self.test, 20261105, 1)
        for key in parameters: np.testing.assert_array_equal(parameters[key], self.parameters[key], err_msg=key)
        for filename in ("predicoes.csv", "predicoes_teste_gravacao.csv", "predicoes_treino_gravacao.csv"):
            np.testing.assert_array_equal(frames[filename].score, self.frames[filename].score)

    def test_raw_control_reproduces_previous_pipeline_exactly(self):
        model = EXP24.EXP17.model_for("A_STANDARD_LINEAR", len(self.train))
        model.fit(self.raw[self.train], self.meta.y.to_numpy()[self.train])
        actual = self.frames["predicoes_teste_gravacao.csv"].query("arm == 'A_RAW'").score
        np.testing.assert_array_equal(actual, model.decision_function(self.raw[self.test]))
        for key, value in EXP24.EXP17.parameters(model).items():
            np.testing.assert_array_equal(value, self.parameters[f"A_RAW__{key}"])

    def test_checkpoint_roundtrip_and_tampering(self):
        EXP24.validate(self.frames, self.parameters, self.meta, self.train, self.test, self.names, 20261105, 1)
        with TemporaryDirectory(prefix="avfad24_") as temp:
            path = Path(temp)/"fold"
            EXP24.save_checkpoint(path, self.frames, self.parameters, "synthetic")
            frames, parameters = EXP24.load_checkpoint(path, "synthetic")
            EXP24.validate(frames, parameters, self.meta, self.train, self.test, self.names, 20261105, 1)
            with self.assertRaises(ValueError): EXP24.load_checkpoint(path, "wrong")
            with (path/"predicoes.csv").open("ab") as stream: stream.write(b"\n")
            with self.assertRaises(ValueError): EXP24.load_checkpoint(path, "synthetic")

    def test_mean_margin_aggregation_cannot_be_silently_changed(self):
        frames = {key:value.copy(deep=True) for key,value in self.frames.items()}
        frames["predicoes.csv"].score += 1e-5
        for arm in EXP24.ARMS:
            mask = (frames["metricas.csv"].arm==arm)&(frames["metricas.csv"].phase=="teste")&(frames["metricas.csv"].unit=="participante")
            for key,value in EXP24.EXP17.measures(frames["predicoes.csv"].query("arm == @arm")).items(): frames["metricas.csv"].loc[mask,key] = value
        with self.assertRaises(AssertionError): EXP24.validate(frames, self.parameters, self.meta, self.train, self.test, self.names, 20261105, 1)

    def test_participant_overlap_is_rejected_before_fit(self):
        with self.assertRaises(ValueError):
            EXP24.evaluate_fold(self.matrices, self.meta, self.names, np.append(self.train,self.test[0]), self.test, 20261105, 1)


if __name__ == "__main__":
    unittest.main()
