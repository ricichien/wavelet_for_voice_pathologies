# -*- coding: utf-8 -*-
"""Testes artificiais do auditor30; nenhum dado ou classificador AVFAD."""
from __future__ import annotations
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("audit30_tests",HERE/"verificar_frase_vogal30.py")
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)
EXP=AUDIT.EXP30


class Audit30Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(1030)
        cls.records=pd.DataFrame([{"participant_id":f"FAKE{i:03d}","sample_id":f"FAKE{i:03d}{j+4:03d}",
                                   "task":f"{j+4:03d}","y":int(i%4!=0)} for i in range(40) for j in range(2+int(i%5!=0))])
        cls.record_matrices={key:rng.normal(size=(len(cls.records),1920 if key=="wpt" else 2048)).astype(np.float32)
                             for key in ("wpt","ssl24","ssl05","ssl12","ssl16")}
        cls.record_names={key:np.asarray([f"{key}__{j}" for j in range(value.shape[1])]) for key,value in cls.record_matrices.items()}
        cls.matrices,cls.meta,cls.names,cls.mapping=EXP.aggregate_sentences(cls.record_matrices,cls.records,cls.record_names,real=False)
        cls.vowel_meta=cls.meta[["participant_id","y","matrix_row"]].copy()
        cls.vowel_meta["sample_id"]=cls.vowel_meta.participant_id+"002";cls.vowel_meta["task"]="002"
        cls.vowel={key:cls.vowel_meta[key].to_numpy() for key in ("sample_id","participant_id","task")}
        cls.vowel["hidden_layer_indices"]=np.asarray([24,5,12,16])
        cls.vowel["wavelet_raw"]=rng.normal(size=(len(cls.meta),1920)).astype(np.float32)
        cls.vowel["wavelet_feature_names"]=cls.record_names["wpt"]
        cls.matrices["wpt_vow"]=cls.vowel["wavelet_raw"]
        cls.names["wpt_vow"]=np.asarray(["VOW__"+str(n) for n in cls.vowel["wavelet_feature_names"]])
        for layer in (24,5,12,16):
            key=f"embeddings_layer{layer:02d}";cls.vowel[key]=rng.normal(size=(len(cls.meta),2048)).astype(np.float32)
            cls.vowel[f"feature_names_layer{layer:02d}"]=cls.record_names[f"ssl{layer:02d}"]
            cls.matrices[f"ssl_vow{layer:02d}"]=cls.vowel[key]
            cls.names[f"ssl_vow{layer:02d}"]=np.asarray(["VOW__"+str(n) for n in cls.vowel[f"feature_names_layer{layer:02d}"]])
        processors={branch:(modality,4 if "256" in branch else 2) for branch,(modality,_) in EXP.PREPROCESSORS.items()}
        cls.patch=patch.object(EXP,"PREPROCESSORS",processors);cls.patch.start()
        cls.grid=(.01,1.);cls.train,cls.test=next(AUDIT.independent_splits(cls.meta,5,20261105))
        cls.frames,cls.arrays=EXP.evaluate_fold(cls.matrices,cls.meta,cls.names,cls.train,cls.test,20261105,1,c_grid=cls.grid)

    @classmethod
    def tearDownClass(cls):cls.patch.stop()

    def test_all_nine_arms_pass_independent_reconstruction(self):
        totals=AUDIT.blank_totals()
        AUDIT.audit_checkpoint(self.matrices,self.meta,self.names,self.train,self.test,20261105,1,self.frames,self.arrays,totals,self.grid)
        self.assertEqual(totals["classifiers"],9);self.assertEqual(totals["preprocessors"],72)
        self.assertEqual(totals["test_margins"],9*len(self.test));self.assertEqual(totals["train_margins"],9*len(self.train))
        self.assertLess(totals["max_margin_error"],1e-7)

    def test_checkpoint_roundtrip_passes_independent_reconstruction(self):
        with TemporaryDirectory() as folder:
            p=Path(folder)/"checkpoint";EXP.save_checkpoint(p,self.frames,self.arrays,"synthetic30")
            frames,arrays=EXP.load_checkpoint(p,"synthetic30")
            AUDIT.audit_checkpoint(self.matrices,self.meta,self.names,self.train,self.test,20261105,1,frames,arrays,AUDIT.blank_totals(),self.grid)

    def test_independent_aggregation_checks_equal_weight_and_all_sources(self):
        AUDIT.verify_aggregation(self.matrices,self.meta,self.names,self.mapping,self.record_matrices,self.records,self.record_names,self.vowel,self.vowel_meta)
        self.assertEqual(self.mapping.groupby("participant_id").size().nunique(),2)

    def test_wrong_sentence_weight_is_rejected(self):
        mapping=self.mapping.copy();mapping.loc[0,"weight"]+=.01
        with self.assertRaises(AssertionError):
            AUDIT.verify_aggregation(self.matrices,self.meta,self.names,mapping,self.record_matrices,self.records,self.record_names,self.vowel,self.vowel_meta)

    def test_vowel_from_different_person_is_rejected(self):
        vowel={key:value.copy() for key,value in self.vowel.items()};vowel["wavelet_raw"][[0,1]]=vowel["wavelet_raw"][[1,0]]
        with self.assertRaises(AssertionError):
            AUDIT.verify_aggregation(self.matrices,self.meta,self.names,self.mapping,self.record_matrices,self.records,self.record_names,vowel,self.vowel_meta)

    def test_scaling_and_budget_layouts(self):
        values={branch:np.ones((2,k),np.float32) for branch,(_,k) in EXP.PREPROCESSORS.items()}
        for rep in EXP.REPRESENTATIONS:
            a=AUDIT.representation(values,rep);b=EXP.represented(values,rep)
            np.testing.assert_array_equal(a,b);self.assertEqual(a.dtype,b.dtype)
            np.testing.assert_allclose((a*a).sum(axis=1),2.,atol=1e-6,rtol=0)
        self.assertEqual(AUDIT.representation(values,"HYB_SENT_VOW512_L05").shape[1],AUDIT.representation(values,"SSL_SENT_VOW512_L05").shape[1])

    def test_corrupt_classifier_coefficient_is_rejected(self):
        arrays={key:value.copy() for key,value in self.arrays.items()};arrays[f"{EXP.ARMS[0]}__coef"]+=.2
        with self.assertRaises((AssertionError,ValueError)):
            AUDIT.audit_checkpoint(self.matrices,self.meta,self.names,self.train,self.test,20261105,1,self.frames,arrays,AUDIT.blank_totals(),self.grid)

    def test_person_train_statistics_are_enforced(self):
        arrays={key:value.copy() for key,value in self.arrays.items()};arrays["outer__WPT_SENT128__scaler_mean"]+=.1
        with self.assertRaises(AssertionError):
            AUDIT.verify_preprocessor(self.matrices["wpt_sent"],self.meta.y.to_numpy(),self.train,arrays,"outer__WPT_SENT128__",2)

    def test_recording_metrics_are_rejected(self):
        frames={key:value.copy() for key,value in self.frames.items()};frames["metricas.csv"]["unit"]="gravacao"
        with self.assertRaises((AssertionError,ValueError)):
            AUDIT.audit_checkpoint(self.matrices,self.meta,self.names,self.train,self.test,20261105,1,frames,self.arrays,AUDIT.blank_totals(),self.grid)

    def test_tie_break_handles_wavelet_and_layer_order(self):
        waves=pd.DataFrame([{"representation":"WPT_SENT128","candidate_id":str(c),"layer":0,"C":c,"inner_BA":.6} for c in (1.,.01)])
        self.assertEqual(AUDIT.independent_winner(waves,["WPT_SENT128"]).C,.01)
        layers=pd.DataFrame([{"representation":f"SSL_SENT128_L{l:02d}","candidate_id":str(l),"layer":l,"C":.01,"inner_BA":.6+1e-13*int(l==5)} for l in (5,12,16,24)])
        self.assertEqual(AUDIT.independent_winner(layers,layers.representation).layer,24)

    def test_incomplete_run_cannot_create_a_report(self):
        with TemporaryDirectory() as folder:
            p=Path(folder);report=p/"report.txt"
            with self.assertRaises(AssertionError):AUDIT.audit(p,p,p,p,p,report)
            self.assertFalse(report.exists());self.assertFalse(report.with_suffix(".json").exists())


if __name__=="__main__":unittest.main(verbosity=2)
