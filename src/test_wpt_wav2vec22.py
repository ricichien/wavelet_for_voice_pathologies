# -*- coding: utf-8 -*-
"""Isolamento da CV aninhada22; apenas dados sinteticos, nenhum WAV."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp22_under_test", HERE / "22_regularizar_wpt_wav2vec_avfad.py")
EXP22 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP22)


class NestedHybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(22161105)
        cls.meta = pd.DataFrame([{"participant_id": f"S{p:03d}", "sample_id": f"S{p:03d}_{task}",
                                  "task": task, "y": int(p % 4 != 0), "matrix_row": p*3+t}
                                 for p in range(36) for t, task in enumerate(("004", "005", "006"))])
        cls.matrices = {"wpt": rng.normal(size=(len(cls.meta),1920)).astype("float32"),
                        "wav2vec": rng.normal(size=(len(cls.meta),2048)).astype("float32")}
        cls.names = {"wpt": np.asarray([f"WPT__sym5__TEST__{j:04d}" for j in range(1920)]),
                     "wav2vec": EXP22.EXP21.embedding_names()}
        cls.train, cls.test = next(EXP22.EXP13.row_splits(cls.meta,5,20261105))
        cls.grid = (.001,1.)
        cls.frames, cls.parameters = EXP22.evaluate_fold(cls.matrices,cls.meta,cls.names,cls.train,cls.test,
                                                        20261105,1,c_grid=cls.grid)

    def test_outer_test_features_do_not_change_search_selection_or_any_fitted_parameter(self):
        changed = {key: value.copy() for key,value in self.matrices.items()}
        for i,matrix in enumerate(changed.values()): matrix[self.test] = 2e5+i*5e4+matrix[self.test]*800
        frames,parameters = EXP22.evaluate_fold(changed,self.meta,self.names,self.train,self.test,20261105,1,c_grid=self.grid)
        self.assertEqual(set(parameters),set(self.parameters))
        for key in parameters: np.testing.assert_array_equal(parameters[key],self.parameters[key],err_msg=key)
        for filename in ("candidatos_internos.csv","predicoes_oof_internas.csv","particoes_internas.csv","selecoes.csv",
                         "predicoes_treino_participante.csv","predicoes_treino_gravacao.csv","atributos.csv"):
            pd.testing.assert_frame_equal(frames[filename],self.frames[filename])

    def test_outer_test_labels_do_not_change_search_models_or_scores(self):
        changed = self.meta.copy(deep=True); changed.loc[self.test,"y"] = 1-changed.loc[self.test,"y"]
        frames,parameters = EXP22.evaluate_fold(self.matrices,changed,self.names,self.train,self.test,20261105,1,c_grid=self.grid)
        for key in parameters: np.testing.assert_array_equal(parameters[key],self.parameters[key],err_msg=key)
        for filename in ("candidatos_internos.csv","predicoes_oof_internas.csv","particoes_internas.csv","selecoes.csv","atributos.csv"):
            pd.testing.assert_frame_equal(frames[filename],self.frames[filename])
        for filename in ("predicoes.csv","predicoes_treino_participante.csv","predicoes_teste_gravacao.csv","predicoes_treino_gravacao.csv"):
            np.testing.assert_array_equal(frames[filename].score,self.frames[filename].score)

    def test_inner_heldout_features_and_labels_cannot_fit_its_own_preprocessor(self):
        local_train,local_val = next(EXP22.EXP13.row_splits(self.meta.iloc[self.train].reset_index(drop=True),3,20261206))
        train,val = self.train[local_train],self.train[local_val]
        y = self.meta.y.to_numpy()
        _,_,first = EXP22.prepare(self.matrices,y,train,val)
        changed = {key:value.copy() for key,value in self.matrices.items()}
        for value in changed.values(): value[val] = value[val]*300+10000
        new_y = y.copy(); new_y[val] = 1-new_y[val]
        _,_,second = EXP22.prepare(changed,new_y,train,val)
        for key in first: np.testing.assert_array_equal(first[key],second[key])

    def test_ties_prefer_stronger_regularization_then_equal_fusion(self):
        candidates = pd.DataFrame([{"candidate_id":EXP22.candidate_id(rep,c),"representation":rep,
                                    "C":c,"alpha":EXP22.ALPHA[rep],"inner_BA":.6}
                                   for rep in ("HYB025","HYB050","HYB075") for c in (.0001,.001,1.)])
        chosen = EXP22.choose(candidates,EXP22.ARM_REPS[EXP22.ARMS[4]])
        self.assertEqual(chosen.C,.0001); self.assertEqual(chosen.alpha,.5)
        candidates = candidates[candidates.alpha!=.5]
        self.assertEqual(EXP22.choose(candidates,EXP22.ARM_REPS[EXP22.ARMS[4]]).alpha,.25)
        candidates.loc[candidates.representation=="HYB075","inner_BA"] = .61
        self.assertEqual(EXP22.choose(candidates,EXP22.ARM_REPS[EXP22.ARMS[4]]).alpha,.75)

    def test_inner_outer_group_partitions_and_candidate_oof_cover_exact_people(self):
        EXP22.validate(self.frames,self.parameters,self.meta,self.train,self.test,self.names,20261105,1,c_grid=self.grid)
        inner = self.frames["predicoes_oof_internas.csv"]
        self.assertEqual(inner.candidate_id.nunique(),6*len(self.grid))
        self.assertFalse(inner.duplicated(["candidate_id","participant_id"]).any())
        self.assertFalse(set(inner.participant_id)&set(self.meta.iloc[self.test].participant_id))
        self.assertEqual(len(inner),6*len(self.grid)*self.meta.iloc[self.train].participant_id.nunique())

    def test_selected_regularization_is_determined_only_by_inner_oof(self):
        frames = {key:value.copy(deep=True) for key,value in self.frames.items()}
        frames["selecoes.csv"].loc[0,"C"] = 1234.
        with self.assertRaises(ValueError): EXP22.validate(frames,self.parameters,self.meta,self.train,self.test,self.names,20261105,1,c_grid=self.grid)

    def test_inner_score_cannot_disagree_with_decision_even_when_BA_is_unchanged(self):
        original = self.frames["predicoes_oof_internas.csv"]
        index = original.index[np.flatnonzero(original.score.to_numpy() != 0)[0]]
        for corruption in ("opposite_margin", "nonfinite"):
            frames = {key:value.copy(deep=True) for key,value in self.frames.items()}
            inner = frames["predicoes_oof_internas.csv"]
            cid = inner.loc[index,"candidate_id"]
            before = EXP22.EXP17.measures(inner[inner.candidate_id==cid])["BA"]
            inner.loc[index,"score"] = -inner.loc[index,"score"] if corruption=="opposite_margin" else np.nan
            # BA tabelada depende das decisoes: inverter somente a margem nao
            # muda a BA e nao pode ser detectado apenas recalculando essa metrica.
            np.testing.assert_array_equal(inner.predicao,original.predicao)
            if corruption=="opposite_margin":
                self.assertEqual(EXP22.EXP17.measures(inner[inner.candidate_id==cid])["BA"],before)
            with self.subTest(corruption=corruption), self.assertRaisesRegex(ValueError,"Score/decisao interna"):
                EXP22.validate(frames,self.parameters,self.meta,self.train,self.test,self.names,20261105,1,c_grid=self.grid)

    def test_participant_overlap_is_rejected_before_search(self):
        with self.assertRaises(ValueError): EXP22.evaluate_fold(self.matrices,self.meta,self.names,np.append(self.train,self.test[0]),self.test,20261105,1,c_grid=self.grid)

    def test_equal_weight_and_C1_reproduce_all_four_fixed_pipelines(self):
        y = self.meta.y.to_numpy()
        tr,te,_ = EXP22.prepare(self.matrices,y,self.train,self.test)
        scores,_ = EXP22.EXP21.fit_fold_models(self.matrices,y,self.train,self.test)
        for old_arm,rep in zip(EXP22.EXP21.ARMS,("WPT128","WAV128","HYB050","WAV256")):
            model = EXP22.fit_classifier(EXP22.represented(tr,rep),y[self.train],1.)
            actual = model.decision_function(EXP22.represented(te,rep))
            np.testing.assert_array_equal(actual,scores[old_arm][1],err_msg=old_arm)
        np.testing.assert_array_equal(EXP22.represented(tr,"HYB050"),np.concatenate([tr['WPT128'],tr['WAV128']],axis=1)/np.sqrt(2.))

    def test_weighted_fusion_and_budget_control_have_fixed_total_energy(self):
        tr,_,_ = EXP22.prepare(self.matrices,self.meta.y.to_numpy(),self.train,self.test)
        for rep in EXP22.REPRESENTATIONS:
            z = EXP22.represented(tr,rep)
            self.assertAlmostEqual(float(np.mean(np.square(z).sum(axis=1))),128.,places=4)
            self.assertEqual(z.shape[1],128 if rep in ('WPT128','WAV128') else 256)

    def test_checkpoint_roundtrip_signature_and_tampering(self):
        with TemporaryDirectory(prefix="avfad22_test_") as temp:
            destination = Path(temp)/"fold"
            EXP22.save_checkpoint(destination,self.frames,self.parameters,"synthetic")
            frames,parameters = EXP22.load_checkpoint(destination,"synthetic")
            EXP22.validate(frames,parameters,self.meta,self.train,self.test,self.names,20261105,1,c_grid=self.grid)
            for key in parameters: np.testing.assert_array_equal(parameters[key],self.parameters[key])
            with self.assertRaises(ValueError): EXP22.load_checkpoint(destination,"other")
            with (destination/"candidatos_internos.csv").open("ab") as stream: stream.write(b"\n")
            with self.assertRaises(ValueError): EXP22.load_checkpoint(destination,"synthetic")

    def test_recording_to_person_margin_mean_is_enforced(self):
        frames = {key:value.copy(deep=True) for key,value in self.frames.items()}
        frames['predicoes.csv']['score'] += 1e-5
        for arm in EXP22.ARMS:
            mask = (frames['metricas.csv'].arm==arm)&(frames['metricas.csv'].phase=='teste')&(frames['metricas.csv'].unit=='participante')
            source = frames['predicoes.csv'].query('arm == @arm')
            for key,value in EXP22.EXP17.measures(source).items(): frames['metricas.csv'].loc[mask,key] = value
        with self.assertRaises(AssertionError): EXP22.validate(frames,self.parameters,self.meta,self.train,self.test,self.names,20261105,1,c_grid=self.grid)


if __name__ == "__main__":
    unittest.main()
