# -*- coding: utf-8 -*-
"""Testes23 sem arquivos AVFAD e sem treinamento."""
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

SPEC = importlib.util.spec_from_file_location("exp23_test", Path(__file__).with_name("23_extrair_wpt_rms_sequencias_avfad.py"))
E = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(E)


class Extraction23Tests(unittest.TestCase):
    def test_global_rms_preserves_relative_amplitude_and_reaches_target(self):
        regions = [np.asarray([.1, -.1] * 50, dtype=np.float32), np.asarray([.3, -.3] * 50, dtype=np.float32)]
        normalized, diag = E.normalize_regions(regions)
        self.assertAlmostEqual(diag['active_rms_after'], .1, places=7)
        self.assertAlmostEqual(float(np.std(normalized[1])/np.std(normalized[0])), 3., places=5)
        np.testing.assert_allclose(normalized[0], regions[0]*diag['rms_global_gain'], rtol=1e-6)
        other, _ = E.normalize_regions([r*8 for r in regions])
        for a,b in zip(normalized,other): np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-6)

    def test_vad_called_once_before_gain_and_raw_summary_matches_legacy(self):
        rng = np.random.default_rng(23)
        regions = [rng.normal(0,.02,25000).astype(np.float32),rng.normal(0,.04,35000).astype(np.float32)]
        signal = np.r_[regions[0],np.zeros(10000),regions[1]].astype(np.float32)
        windows = E.LEGACY.make_windows(regions,48000)
        expected = E.EXP18.summarize_windows(windows,2)
        with patch.object(E.LEGACY,'detect_speech_regions',return_value=(regions,False)) as vad:
            raw,rms,vector,diag,control = E.extract_signal48(signal,expected,2)
        vad.assert_called_once()
        self.assertEqual(vad.call_args.args[1],48000)
        np.testing.assert_array_equal(E.summarize(raw),expected)
        np.testing.assert_array_equal(E.summarize(rms),vector)
        self.assertEqual(raw.shape,rms.shape)
        self.assertEqual(raw.shape[1],480)
        self.assertEqual(diag['window_count'],len(windows))
        self.assertTrue(control['equivalent']); self.assertEqual(control['max_abs_error'],0.)

    def test_quality_detects_channel_cancellation_and_fullscale_without_exclusion(self):
        a=np.asarray([1.,-.5,.5,-1.],dtype=np.float32)
        q=E.audio_quality(np.column_stack([a,-a]),48000)
        self.assertEqual(q['channels_original'],2)
        self.assertEqual(q['mono_to_channel_ac_rms_ratio'],0.)
        self.assertAlmostEqual(q['min_channel_correlation'],-1.)
        self.assertEqual(q['flag_possible_channel_cancellation'],1)
        self.assertEqual(q['flag_any_full_scale_sample'],1)
        self.assertAlmostEqual(q['max_clip_fraction'],.5)
        same=E.audio_quality(np.column_stack([a,a]),48000)
        self.assertAlmostEqual(same['mono_to_channel_ac_rms_ratio'],1.)
        self.assertEqual(same['flag_possible_channel_cancellation'],0)

    def test_reader_operations_match_exp18_exactly(self):
        rng=np.random.default_rng(230)
        audio=rng.normal(.001,.1,size=(4800,2)).astype(np.float32)
        expected=audio.mean(axis=1,dtype=np.float64).astype(np.float32)
        expected=expected-np.mean(expected,dtype=np.float64)
        actual,q=E.process_audio48(audio,48000)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(q['source_samples'],4800)

    def test_blob_cache_signature_shape_and_tampering(self):
        selected=pd.DataFrame({'sample_id':['A004'],'participant_id':['A'],'task':['004'],'row_index':[0]})
        raw=np.arange(960,dtype=np.float32).reshape(2,480)/100
        rms=raw*.5
        vec=E.summarize(rms)
        rb,rh=E.encode(raw); nb,nh=E.encode(rms); vb,vh=E.encode(vec)
        with TemporaryDirectory(prefix='avfad23_test_') as tmp:
            path=Path(tmp)/'new.sqlite'; con=E.init_cache(path,'same')
            diag={'window_count':2,'active_rms_after':.1}; control={'equivalent':True,'max_abs_error':0.}
            with con: con.execute('INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',('A004','A','004',2,rb,nb,vb,rh,nh,vh,json.dumps(diag),'{}',json.dumps(control)))
            self.assertEqual(set(E.validate_cache(con,selected,np.asarray([E.summarize(raw)]))),{'A004'})
            with self.assertRaises(ValueError): E.decode(rb,'wrong',(2,480))
            with self.assertRaises(ValueError): E.decode(rb,rh,(1,480))
            with con: con.execute('UPDATE features SET raw=?',(b'broken',))
            with self.assertRaises(ValueError): E.validate_cache(con,selected)
            con.close()
            with self.assertRaises(ValueError): E.init_cache(path,'different')

    def test_export_ragged_order_ids_and_hashes(self):
        selected=pd.DataFrame({'sample_id':['B004','A004'],'participant_id':['B','A'],'task':['004','004'],'y':[1,0],'row_index':[1,0]})
        names,base=E.EXP18.feature_names()
        with TemporaryDirectory(prefix='avfad23_export_') as tmp:
            out=Path(tmp); con=E.init_cache(out/'cache.sqlite','signature'); found={}
            E.atomic_text(out/'assinatura_execucao.json',E.json_text({'mode':'synthetic'}))
            for i,row in enumerate(selected.itertuples()):
                raw=np.full((i+2,480),i+.25,dtype=np.float32); rms=raw*.5; vec=E.summarize(rms)
                rb,rh=E.encode(raw); nb,nh=E.encode(rms); vb,vh=E.encode(vec)
                diag={'window_count':len(raw),'active_rms_after':.1}; q={'flag_any_full_scale_sample':0,'flag_possible_channel_cancellation':0,'flag_duration_over_60s':0}
                control={'equivalent':True,'max_abs_error':0.}
                with con: con.execute('INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',(row.sample_id,row.participant_id,row.task,len(raw),rb,nb,vb,rh,nh,vh,json.dumps(diag),json.dumps(q),json.dumps(control)))
                found[row.sample_id]={'windows':len(raw),'diagnostics':diag,'quality':q,'control':control}
            E.export(out,selected,found,con,names,base,{'mode':'synthetic'},1.,.5,2)
            with np.load(out/'sequencias_wpt.npz',allow_pickle=False) as data:
                np.testing.assert_array_equal(data['sample_id'],selected.sample_id)
                np.testing.assert_array_equal(data['offsets'],[0,2,5])
                np.testing.assert_array_equal(data['data_raw'][:2],np.full((2,480),.25,dtype=np.float32))
                self.assertEqual(data['data_rms'].shape,(5,480))
            with np.load(out/'features_wpt_rms.npz',allow_pickle=False) as data:
                np.testing.assert_array_equal(data['participant_id'],selected.participant_id)
                self.assertEqual(data['wavelet'].shape,(2,1920))
            hashes=json.loads((out/'ARTEFATOS_COMPLETOS.json').read_text(encoding='utf8'))
            self.assertEqual(set(hashes),set(E.EXPORTS))
            self.assertTrue(all(E.sha(out/name)==digest for name,digest in hashes.items()))
            con.close()

    def test_complete_resume_never_rereads_audio_and_rejects_changed_marker(self):
        selected=pd.DataFrame({'sample_id':['A004','B004'],'participant_id':['A','B'],
                               'task':['004','004'],'y':[0,1],'row_index':[0,1]})
        names,base=E.EXP18.feature_names()
        raw=np.arange(960,dtype=np.float32).reshape(2,480)/100
        rms=raw*.5
        vector=E.summarize(rms)
        reference=np.stack([E.summarize(raw)]*2)
        diag={'window_count':2,'active_rms_after':.1}
        quality={'flag_any_full_scale_sample':0,'flag_possible_channel_cancellation':0,'flag_duration_over_60s':0}
        control={'equivalent':True,'max_abs_error':0.,'rtol':2e-5,'atol':2e-5}
        identity={'mode':'synthetic'}
        with TemporaryDirectory(prefix='avfad23_resume_') as tmp:
            out=Path(tmp)/'output'
            with patch.object(E,'read_audio48',return_value=(np.ones(100,dtype=np.float32),quality)), \
                 patch.object(E,'extract_signal48',return_value=(raw,rms,vector,diag,control)):
                E.run(Path(tmp),out,selected,reference,names,base,identity,2)
            before={name:E.sha(out/name) for name in E.EXPORTS}
            with patch.object(E,'read_audio48',side_effect=AssertionError('Audio relido')):
                E.run(Path(tmp),out,selected,reference,names,base,identity,2)
            self.assertEqual(before,{name:E.sha(out/name) for name in E.EXPORTS})
            E.atomic_text(out/'ARTEFATOS_COMPLETOS.json','{}')
            with self.assertRaisesRegex(ValueError,'Marcador'):
                E.run(Path(tmp),out,selected,reference,names,base,identity,2)


if __name__=='__main__': unittest.main(verbosity=2)
