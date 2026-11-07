# -*- coding: utf-8 -*-
"""Testes sintéticos de unidade por pessoa, fusão e isolamento do30."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("exp30_tests", HERE / "30_avaliar_frase_vogal_wavelet_avfad.py")
E = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(E)


class PersonFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(30161107)
        cls.sentence_meta = pd.DataFrame([{"participant_id": f"P{i:03d}", "sample_id": f"P{i:03d}{t:03d}",
                                          "task": f"{t:03d}", "y": int(i % 4 != 0)}
                                         for i in range(36) for t in range(4, 6+int(i % 3 != 0))])
        cls.sentences = {"wpt": rng.normal(size=(len(cls.sentence_meta),1920)).astype(np.float32)}
        cls.sentence_names = {"wpt": np.asarray([f"WPT__sym5__TEST__{i:04d}" for i in range(1920)])}
        for layer in E.LAYERS:
            cls.sentences[f"ssl{layer:02d}"] = rng.normal(size=(len(cls.sentence_meta),2048)).astype(np.float32)
            cls.sentence_names[f"ssl{layer:02d}"] = E.EXP21.embedding_names() if layer==24 else E.EXP28.intermediate_names(layer)
        matrices,cls.meta,names,cls.mapping = E.aggregate_sentences(cls.sentences,cls.sentence_meta,cls.sentence_names,real=False)
        cls.vowel_meta=cls.meta[["participant_id","y","matrix_row"]].copy()
        cls.vowel_meta["sample_id"],cls.vowel_meta["task"]=cls.meta.vowel_sample_id,"002"
        cls.payload={key:cls.vowel_meta[key].to_numpy() for key in ("sample_id","participant_id","task")}
        cls.payload.update(hidden_layer_indices=np.asarray(E.LAYERS),wavelet_raw=rng.normal(size=(36,1920)).astype(np.float32),
                           wavelet_rms=rng.normal(size=(36,1920)).astype(np.float32),wavelet_feature_names=cls.sentence_names["wpt"])
        for layer in E.LAYERS:
            cls.payload[f"embeddings_layer{layer:02d}"]=rng.normal(size=(36,2048)).astype(np.float32)
            cls.payload[f"feature_names_layer{layer:02d}"]=cls.sentence_names[f"ssl{layer:02d}"]
        cls.ident={"mode":"completo","sample_ids":cls.vowel_meta.sample_id.tolist(),"configuration":{
            "task":"002","task_definition":"sustained_a_all_recorded_repetitions_VAD_not_Praat_segment",
            "model_id":E.EXP21.MODEL_ID,"model_revision":"1a640f32ac3e39899438a2931f9924c02f080a54","frozen":True,
            "sample_rate":16000,"hidden_layers":[24,5,12,16],"pooling":"mean_std","wavelet_raw_rms":True,
            "wavelet":"sym5","wavelet_levels":[5,6],"wavelet_source_rate":48000,"wavelet_windows":[.4,.2],
            "wavelet_summaries":["mean","std","q10","q90"],"wavelet_RMS_target":.1,
            "control_reference":"new_audio_pilot_forward27_vs20_no_sentence_equivalence","full_requires_signed_pilot":True,
            "feature_dimension":12032,"feature_dimensions":{"wavelet_raw":1920,"wavelet_rms":1920,**{f"embeddings_layer{x:02d}":2048 for x in E.LAYERS}}}}
        cls.matrices,cls.names=E.validate_vowel(matrices,cls.meta,names,cls.vowel_meta,cls.payload,cls.ident,real=False)
        cls.train,cls.test=next(E.EXP13.row_splits(cls.meta,5,20261105));cls.grid=(.001,1.)
        cls.frames,cls.arrays=E.evaluate_fold(cls.matrices,cls.meta,cls.names,cls.train,cls.test,20261105,1,c_grid=cls.grid)

    def assert_fitted_equal(self,frames,arrays):
        self.assertEqual(set(arrays),set(self.arrays))
        for key in arrays:np.testing.assert_array_equal(arrays[key],self.arrays[key],err_msg=key)
        for key in ("candidatos_internos.csv","predicoes_oof_internas.csv","particoes_internas.csv","selecoes.csv","atributos.csv","amostras_suporte.csv"):
            pd.testing.assert_frame_equal(frames[key],self.frames[key])

    def test_person_means_use_every_sentence_with_equal_weights(self):
        self.assertEqual(len(self.meta),36);self.assertEqual(len(E.PREPROCESSORS),18)
        for person in self.meta.itertuples():
            indices=np.flatnonzero(self.sentence_meta.participant_id.to_numpy()==person.participant_id)
            np.testing.assert_array_equal(self.matrices['wpt_sent'][person.matrix_row],self.sentences['wpt'][indices].mean(axis=0,dtype=np.float64).astype(np.float32))
            sub=self.mapping[self.mapping.participant_id==person.participant_id]
            self.assertAlmostEqual(sub.weight.sum(),1.);np.testing.assert_array_equal(sub.source_row,indices)
            np.testing.assert_array_equal(sub.weight,np.full(len(indices),1/len(indices)))

    def test_person_means_do_not_use_labels_or_other_people(self):
        changed=self.sentence_meta.copy();pid=self.meta.iloc[self.test[0]].participant_id
        changed.loc[changed.participant_id==pid,'y']=1-changed.loc[changed.participant_id==pid,'y']
        matrices,_,_,_=E.aggregate_sentences(self.sentences,changed,self.sentence_names,real=False)
        for key in matrices:np.testing.assert_array_equal(matrices[key],self.matrices[key])
        features={k:v.copy() for k,v in self.sentences.items()};rows=np.flatnonzero(self.sentence_meta.participant_id.to_numpy()==pid)
        for value in features.values():value[rows]+=1000
        matrices,_,_,_=E.aggregate_sentences(features,self.sentence_meta,self.sentence_names,real=False)
        for key in matrices:np.testing.assert_array_equal(matrices[key][self.train],self.matrices[key][self.train])

    def test_outer_features_cannot_change_models_or_choices(self):
        changed={key:value.copy() for key,value in self.matrices.items()}
        for value in changed.values():value[self.test]=value[self.test]*500+1e5
        frames,arrays=E.evaluate_fold(changed,self.meta,self.names,self.train,self.test,20261105,1,c_grid=self.grid)
        self.assert_fitted_equal(frames,arrays)
        pd.testing.assert_frame_equal(frames['predicoes_treino_participante.csv'],self.frames['predicoes_treino_participante.csv'])

    def test_outer_labels_and_clinical_metadata_cannot_change_models_or_choices(self):
        changed=self.meta.copy();changed.loc[self.test,'y']=1-changed.loc[self.test,'y'];changed['age']=999;changed['sex']='X';changed['visit_year']=1900
        frames,arrays=E.evaluate_fold(self.matrices,changed,self.names,self.train,self.test,20261105,1,c_grid=self.grid)
        self.assert_fitted_equal(frames,arrays)
        for key in ('predicoes.csv','predicoes_treino_participante.csv'):np.testing.assert_array_equal(frames[key].score,self.frames[key].score)

    def test_inner_validation_does_not_fit_its_own_preprocessor(self):
        tr,va=next(E.EXP13.row_splits(self.meta.iloc[self.train].reset_index(drop=True),3,20261206));tr,va=self.train[tr],self.train[va]
        y=self.meta.y.to_numpy();_,_,before=E.prepare(self.matrices,y,tr,va)
        changed={k:v.copy() for k,v in self.matrices.items()}
        for value in changed.values():value[va]=1e4+300*value[va]
        ynew=y.copy();ynew[va]=1-ynew[va];_,_,after=E.prepare(changed,ynew,tr,va)
        for key in before:np.testing.assert_array_equal(before[key],after[key])

    def test_layouts_and_scaling_are_predefined(self):
        values={branch:np.full((3,k),i+1,dtype=float) for i,(branch,(_,k)) in enumerate(E.PREPROCESSORS.items())}
        self.assertEqual(len(E.REPRESENTATIONS)*len(E.C_GRID),135)
        for rep in E.REPRESENTATIONS:
            branches=E.branches_for(rep);expected=np.concatenate([values[b] for b in branches],axis=1)
            if len(branches)>1:expected/=2 if rep.startswith(('HYB_SENT_VOW512','SSL_SENT_VOW512')) else np.sqrt(2.)
            np.testing.assert_array_equal(E.represented(values,rep),expected)
            if '_L' in rep:
                layers={int(branch.rsplit('_L',1)[1]) for branch in branches if '_L' in branch}
                self.assertEqual(layers,{E.layer_of(rep)})
        tr,_,_=E.prepare(self.matrices,self.meta.y.to_numpy(),self.train,self.test)
        for rep in E.REPRESENTATIONS:self.assertAlmostEqual(float(np.mean(np.square(E.represented(tr,rep)).sum(axis=1))),128.,places=4)

    def test_ties_use_lower_C_then_layer_order(self):
        candidates=pd.DataFrame([{'candidate_id':E.candidate_id(rep,c),'representation':rep,'layer':E.layer_of(rep),'C':c,'inner_BA':.6}
                                 for rep in E.ARM_REPS[E.ARMS[5]] for c in E.C_GRID])
        winner=E.choose(candidates,E.ARM_REPS[E.ARMS[5]]);self.assertEqual(winner.C,1e-4);self.assertEqual(winner.layer,24)
        candidates.loc[candidates.layer==5,'inner_BA']+=1e-7
        self.assertEqual(E.choose(candidates,E.ARM_REPS[E.ARMS[5]]).layer,5)

    def test_partitions_person_counts_weights_and_choices_validate(self):
        E.validate(self.frames,self.arrays,self.meta,self.train,self.test,self.names,20261105,1,c_grid=self.grid)
        self.assertEqual(self.frames['predicoes_oof_internas.csv'].candidate_id.nunique(),27*len(self.grid))
        self.assertTrue(self.frames['metricas.csv'].unit.eq('participante').all());self.assertEqual(len(self.frames['metricas.csv']),18)
        self.assertEqual(len(self.frames['predicoes.csv']),len(self.test)*9)

    def test_duplicate_people_and_overlap_are_rejected(self):
        changed=self.meta.copy();changed.loc[1,'participant_id']=changed.loc[0,'participant_id']
        with self.assertRaises(ValueError):E.evaluate_fold(self.matrices,changed,self.names,self.train,self.test,20261105,1,c_grid=self.grid)
        with self.assertRaises(ValueError):E.evaluate_fold(self.matrices,self.meta,self.names,np.append(self.train,self.test[0]),self.test,20261105,1,c_grid=self.grid)

    def test_vowel_contract_rejects_identity_and_configuration_changes(self):
        for kind in ('label','id','task','mode','rms_as_input','names'):
            meta=self.vowel_meta.copy();payload={k:v.copy() for k,v in self.payload.items()};ident=copy.deepcopy(self.ident)
            if kind=='label':meta.loc[0,'y']=1-meta.loc[0,'y']
            elif kind=='id':payload['participant_id']=payload['participant_id'][::-1]
            elif kind=='task':meta['task']='001'
            elif kind=='mode':ident['mode']='piloto_3'
            elif kind=='rms_as_input':ident['configuration']['wavelet']='db4'
            elif kind=='names':payload['wavelet_feature_names']=payload['wavelet_feature_names'][::-1]
            matrices={k:v.copy() for k,v in self.matrices.items() if 'vow' not in k};names={k:v.copy() for k,v in self.names.items() if 'vow' not in k}
            with self.subTest(kind=kind),self.assertRaises(ValueError):E.validate_vowel(matrices,self.meta,names,meta,payload,ident,real=False)

    def test_vowel_RMS_does_not_enter_classifier_inputs(self):
        payload={k:v.copy() for k,v in self.payload.items()};payload['wavelet_rms'][:]=1e5
        matrices={k:v.copy() for k,v in self.matrices.items() if 'vow' not in k};names={k:v.copy() for k,v in self.names.items() if 'vow' not in k}
        after,_=E.validate_vowel(matrices,self.meta,names,self.vowel_meta,payload,self.ident,real=False)
        for key in after:np.testing.assert_array_equal(after[key],self.matrices[key])

    def test_checkpoint_and_choice_tamper_are_rejected(self):
        with TemporaryDirectory(prefix='avfad30_synth_') as temp:
            folder=Path(temp)/'fold';E.save_checkpoint(folder,self.frames,self.arrays,'synthetic')
            frames,arrays=E.load_checkpoint(folder,'synthetic');E.validate(frames,arrays,self.meta,self.train,self.test,self.names,20261105,1,c_grid=self.grid)
            with self.assertRaises(ValueError):E.load_checkpoint(folder,'other')
            frames['selecoes.csv'].loc[0,'C']=123
            with self.assertRaises(ValueError):E.validate(frames,arrays,self.meta,self.train,self.test,self.names,20261105,1,c_grid=self.grid)
            with (folder/'metricas.csv').open('ab') as stream:stream.write(b'\n')
            with self.assertRaises(ValueError):E.load_checkpoint(folder,'synthetic')


if __name__=='__main__':unittest.main()
