# -*- coding: utf-8 -*-
"""Contratos e isolamento26. Somente dados sintéticos e uma época de CPU."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("exp26_under_test",HERE/"26_hibrido_wavelet_temporal_wav2vec.py")
EXP26=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(EXP26)


class Hybrid26Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        EXP26.EXP25.configure_torch("cpu",threads=1)
        rng=np.random.default_rng(26161106)
        cls.meta=pd.DataFrame([{"sample_id":f"S{person:03d}_{task}","participant_id":f"S{person:03d}","task":task,
                               "y":int(person%4!=0),"matrix_row":person*2+t}
                              for person in range(40) for t,task in enumerate(("004","005"))])
        cls.inputs={"rms":rng.normal(size=(len(cls.meta),480,16)).astype("float32"),
                    "wav":rng.normal(size=(len(cls.meta),2048)).astype("float32")}
        cls.train,cls.test=next(EXP26.EXP13.row_splits(cls.meta,5,20261105))
        cls.options={**EXP26.DEFAULT_OPTIONS,"max_epochs":1,"min_epochs":1,"patience":1,"steps":16,"batch_people":8}
        cls.arm=EXP26.ARMS[2]
        cls.result=EXP26.train_arm(cls.inputs,cls.meta,cls.train,cls.test,20261105,1,cls.arm,"cpu",cls.options)

    def test_model_shapes_parameter_control_and_both_hybrid_branches_receive_gradients(self):
        EXP26.EXP25.seed_everything(17)
        counts={}
        for arm in EXP26.ARMS:
            model=EXP26.model_for(arm,self.options)
            counts[arm]=sum(p.numel() for p in model.parameters())
            seq=torch.from_numpy(self.inputs['rms'][:3]) if EXP26.uses_wavelet(arm) else None
            wav=torch.from_numpy(self.inputs['wav'][:3])
            self.assertEqual(tuple(model(seq,wav).shape),(3,))
            self.assertEqual(tuple(model.encode(seq,wav).shape),(3,64 if EXP26.uses_wavelet(arm) else 32))
            if EXP26.uses_wavelet(arm):
                model(seq,wav).sum().backward()
                self.assertTrue(any(p.grad is not None and torch.any(p.grad!=0) for p in model.wavelet.parameters()))
                self.assertTrue(any(p.grad is not None and torch.any(p.grad!=0) for p in model.wav.parameters()))
        self.assertGreater(counts[EXP26.ARMS[1]],max(counts[EXP26.ARMS[2]],counts[EXP26.ARMS[3]]))

    def test_reproducible_seed_recreates_initial_and_trained_weights(self):
        result=EXP26.train_arm(self.inputs,self.meta,self.train,self.test,20261105,1,self.arm,"cpu",self.options)
        self.assertEqual(result['selection']['initial_state_sha256'],self.result['selection']['initial_state_sha256'])
        self.assertEqual(result['selection']['selected_epoch'],self.result['selection']['selected_epoch'])
        for key in result['state']:torch.testing.assert_close(result['state'][key],self.result['state'][key],rtol=0,atol=0)
        pd.testing.assert_frame_equal(result['frames']['predicoes.csv'],self.result['frames']['predicoes.csv'])

    def test_outer_heldout_features_change_no_scaler_epoch_or_trained_weight(self):
        changed={key:value.copy() for key,value in self.inputs.items()}
        for value in changed.values():value[self.test]=1e5+value[self.test]*300
        result=EXP26.train_arm(changed,self.meta,self.train,self.test,20261105,1,self.arm,"cpu",self.options)
        self.assertEqual(result['selection']['selected_epoch'],self.result['selection']['selected_epoch'])
        self.assertEqual(result['selection']['best_validation_loss'],self.result['selection']['best_validation_loss'])
        for key in result['arrays']:np.testing.assert_array_equal(result['arrays'][key],self.result['arrays'][key],err_msg=key)
        for key in result['state']:torch.testing.assert_close(result['state'][key],self.result['state'][key],rtol=0,atol=0)
        for key in result['selection_state']:torch.testing.assert_close(result['selection_state'][key],self.result['selection_state'][key],rtol=0,atol=0)
        pd.testing.assert_frame_equal(result['frames']['predicoes_treino_participante.csv'],self.result['frames']['predicoes_treino_participante.csv'])

    def test_outer_heldout_labels_change_no_selection_or_trained_weight(self):
        meta=self.meta.copy();meta.loc[self.test,'y']=1-meta.loc[self.test,'y']
        result=EXP26.train_arm(self.inputs,meta,self.train,self.test,20261105,1,self.arm,"cpu",self.options)
        self.assertEqual(result['selection']['selected_epoch'],self.result['selection']['selected_epoch'])
        for key in result['arrays']:np.testing.assert_array_equal(result['arrays'][key],self.result['arrays'][key])
        for key in result['state']:torch.testing.assert_close(result['state'][key],self.result['state'][key],rtol=0,atol=0)
        np.testing.assert_array_equal(result['frames']['predicoes.csv'].score,self.result['frames']['predicoes.csv'].score)

    def test_embedding_scaler_is_fit_only_to_training_recordings(self):
        scalers=EXP26.fit_scalers(self.inputs,self.train,self.arm)
        expected=self.inputs['wav'][self.train].astype('float64')
        np.testing.assert_allclose(scalers['wav']['mean'],expected.mean(axis=0),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(scalers['wav']['scale'],expected.std(axis=0),atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(scalers['wav']['train_rows'],self.train)
        np.testing.assert_array_equal(scalers['rms']['train_rows'],self.train)

    def test_each_arm_completes_train_validate_and_can_replay_outer_scores(self):
        for arm in EXP26.ARMS:
            with self.subTest(arm=arm):
                result = self.result if arm == self.arm else EXP26.train_arm(
                    self.inputs,self.meta,self.train,self.test,20261105,1,arm,'cpu',self.options)
                EXP26.validate_result(result,self.meta,self.train,self.test,20261105,1,arm,self.options)
                model = EXP26.model_for(arm,self.options)
                model.load_state_dict(result['state'],strict=True)
                modalities = ('wav','rms') if EXP26.uses_wavelet(arm) else ('wav',)
                scalers = {modality:{key:result['arrays'][f'outer__{modality}__{key}']
                                      for key in ('mean','scale','train_rows','n_frames')}
                           for modality in modalities}
                replay = EXP26.predict_records(model,self.inputs,self.test,scalers,'cpu')
                observed = result['frames']['predicoes_teste_gravacao.csv'].set_index('sample_id')
                np.testing.assert_allclose(replay,observed.loc[self.meta.iloc[self.test].sample_id,'score'],atol=0,rtol=0)

    def test_wav_only_training_does_not_read_wavelet_values(self):
        arm = EXP26.ARMS[0]
        result = EXP26.train_arm(self.inputs,self.meta,self.train,self.test,20261105,1,arm,'cpu',self.options)
        changed = {**self.inputs,'rms':np.full_like(self.inputs['rms'],np.nan)}
        other = EXP26.train_arm(changed,self.meta,self.train,self.test,20261105,1,arm,'cpu',self.options)
        for key in result['state']:
            torch.testing.assert_close(result['state'][key],other['state'][key],rtol=0,atol=0)
        pd.testing.assert_frame_equal(result['frames']['predicoes.csv'],other['frames']['predicoes.csv'])

    def test_participant_mean_logits_precedes_weighted_BCE_and_counts_all_phrases(self):
        logits=torch.tensor([-2.,4.,3.],requires_grad=True)
        actual=EXP26.EXP25.person_logits(logits,[2,1])
        torch.testing.assert_close(actual,torch.tensor([1.,3.]))
        labels=torch.tensor([0.,1.]);weights=torch.tensor([2.,.5])
        correct=(torch.nn.functional.binary_cross_entropy_with_logits(actual,labels,reduction='none')*weights).mean()
        wrong=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,0.,1.]))
        self.assertFalse(torch.isclose(correct,wrong))
        correct.backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_participant_overlap_is_rejected_before_training(self):
        with self.assertRaises(ValueError):EXP26.train_arm(self.inputs,self.meta,np.append(self.train,self.test[0]),self.test,20261105,1,self.arm,'cpu',self.options)

    def test_embedding_identity_alignment_and_frozen_revision_are_strict(self):
        configuration={"model_id":EXP26.EXP21.MODEL_ID,"sample_rate":16000,"frozen":True,"hidden_layer":"last",
                       "pooling":"mean_std","hidden_size":1024,"model_revision":EXP26.MODEL_REVISION}
        embedded={key:self.meta[key].to_numpy(dtype=str) for key in ('sample_id','participant_id','task')}
        EXP26.validate_embedding_alignment(self.inputs['wav'],self.meta,embedded,EXP26.EXP21.embedding_names(),self.meta,{'configuration':configuration},real=False)
        for bad in ('order','label','revision','frozen'):
            other=self.meta.copy();config=configuration.copy()
            if bad=='order':other=other.iloc[::-1].reset_index(drop=True)
            elif bad=='label':other.loc[other.participant_id=='S000','y']=1
            elif bad=='revision':config['model_revision']='f'*40
            else:config['frozen']=False
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                EXP26.validate_embedding_alignment(self.inputs['wav'],other,embedded,EXP26.EXP21.embedding_names(),self.meta,{'configuration':config},real=False)

    def test_checkpoint_roundtrip_uses_tensor_only_loading_and_rejects_tampering(self):
        EXP26.validate_result(self.result,self.meta,self.train,self.test,20261105,1,self.arm,self.options)
        with TemporaryDirectory(prefix='avfad26_test_') as temporary:
            folder=Path(temporary)/'arm';EXP26.save_checkpoint(folder,self.result,'synthetic')
            loaded=EXP26.load_checkpoint(folder,'synthetic')
            EXP26.validate_result(loaded,self.meta,self.train,self.test,20261105,1,self.arm,self.options)
            for key in loaded['state']:torch.testing.assert_close(loaded['state'][key],self.result['state'][key],rtol=0,atol=0)
            with self.assertRaises(ValueError):EXP26.load_checkpoint(folder,'other')
            with (folder/'modelo_refit.pt').open('ab') as stream:stream.write(b'x')
            with self.assertRaises(ValueError):EXP26.load_checkpoint(folder,'synthetic')

    def test_epoch_and_mean_margin_decisions_cannot_be_changed_silently(self):
        frames={key:value.copy(deep=True) for key,value in self.result['frames'].items()}
        result={**self.result,'frames':frames,'selection':dict(self.result['selection'])}
        result['selection']['selected_epoch']=2
        with self.assertRaises(ValueError):EXP26.validate_result(result,self.meta,self.train,self.test,20261105,1,self.arm,self.options)
        result['selection']=dict(self.result['selection']);frames['predicoes.csv'].score+=1e-4
        for key,value in EXP26.EXP17.measures(frames['predicoes.csv']).items():
            frames['metricas.csv'].loc[(frames['metricas.csv'].phase=='teste')&(frames['metricas.csv'].unit=='participante'),key]=value
        with self.assertRaises(AssertionError):EXP26.validate_result(result,self.meta,self.train,self.test,20261105,1,self.arm,self.options)


if __name__=='__main__':unittest.main()
