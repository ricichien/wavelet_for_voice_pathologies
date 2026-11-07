# -*- coding: utf-8 -*-
"""Testes sintéticos da auditoria posterior; nenhum cache AVFAD é lido."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location("audit_under_test",HERE/"verificar_redes25_26.py")
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)


class NetworkAuditTests(unittest.TestCase):
    def test_two_pass_statistics_match_training_welford_for_temporal_and_static_inputs(self):
        rng=np.random.default_rng(261106)
        rows=np.arange(0,101,2)
        for shape in ((101,8),(101,8,16)):
            values=rng.normal(size=shape).astype(np.float32)
            values[:,0]=3.25
            expected=AUDIT.EXP25.fit_scaler(values,rows)
            actual=AUDIT.independently_scaled(values,rows)
            np.testing.assert_allclose(actual['mean'],expected['mean'],atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(actual['scale'],expected['scale'],atol=1e-12,rtol=1e-12)
            np.testing.assert_array_equal(actual['n_frames'],expected['n_frames'])
            np.testing.assert_array_equal(actual['train_rows'],rows)
            changed=values.copy();changed[np.setdiff1d(np.arange(101),rows)]=np.nan
            other=AUDIT.independently_scaled(changed,rows)
            np.testing.assert_array_equal(other['mean'],actual['mean'])
            np.testing.assert_array_equal(other['scale'],actual['scale'])

    def test_independent_folds_and_internal_groups_reproduce_frozen_protocol(self):
        meta=pd.DataFrame([{'participant_id':f'S{i:03d}','y':int(i%4!=0)} for i in range(40) for _ in range(2)])
        for fold,((train,test),(expected_train,expected_test)) in enumerate(zip(AUDIT.splits(meta,20261105),AUDIT.EXP25.EXP13.row_splits(meta,5,20261105)),1):
            np.testing.assert_array_equal(train,expected_train)
            np.testing.assert_array_equal(test,expected_test)
            fit,validation=AUDIT.internal(meta,train,20261105,fold)
            expected_fit,expected_validation=AUDIT.EXP25.internal_split(meta,train,20261105,fold)
            np.testing.assert_array_equal(fit,expected_fit);np.testing.assert_array_equal(validation,expected_validation)
            groups=[set(meta.iloc[rr].participant_id) for rr in (fit,validation,test)]
            self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])

    def test_history_rejects_wrong_epoch_missing_refit_and_early_stop_violation(self):
        options={'min_epochs':2,'max_epochs':10,'min_delta':1e-4,'patience':2}
        history=pd.DataFrame([{'phase':'selecao','epoch':i,'train_loss':.5,'validation_loss':loss}
                              for i,loss in enumerate((.9,.8,.7,.69999,.70001),1)]+
                             [{'phase':'refit','epoch':i,'train_loss':.5,'validation_loss':np.nan} for i in range(1,4)])
        selected={'selected_epoch':3,'selection_epochs_run':5,'best_validation_loss':.7}
        AUDIT.check_history(history,selected,options)
        with self.assertRaises(AssertionError):AUDIT.check_history(history,{**selected,'selected_epoch':4},options)
        with self.assertRaises(AssertionError):AUDIT.check_history(history.iloc[:-1],selected,options)
        extra=pd.concat([history[history.phase=='selecao'],pd.DataFrame([{'phase':'selecao','epoch':6,'train_loss':.5,'validation_loss':.8}]),history[history.phase=='refit']],ignore_index=True)
        with self.assertRaises(AssertionError):AUDIT.check_history(extra,{**selected,'selection_epochs_run':6},options)

    def test_independent_metrics_use_confusion_counts_and_threshold_zero(self):
        frame=pd.DataFrame({'y':[0,0,1,1],'score':[-2.,0.,2.,-1.],'predicao':[0,1,1,0]})
        actual=AUDIT.metrics(frame)
        self.assertEqual(actual['BA'],.5);self.assertEqual(actual['TN'],1);self.assertEqual(actual['FP'],1)
        self.assertEqual(actual['FN'],1);self.assertEqual(actual['TP'],1)
        frame.loc[1,'predicao']=0
        with self.assertRaises(AssertionError):AUDIT.metrics(frame)

    def test_bootstrap_keeps_identical_seed_repetitions_in_the_same_person_cluster(self):
        rows=[]
        for arm,pred in (('A',[0,1,1,0]),('B',[0,0,1,1])):
            for i,(label,decision) in enumerate(zip((0,0,1,1),pred)):
                rows.append({'seed':1,'arm':arm,'participant_id':f'S{i}','y':label,'predicao':decision})
        raw=pd.DataFrame(rows)
        one=AUDIT.bootstrap(raw,[1],{'A':1,'B':-1})
        two=AUDIT.bootstrap(pd.concat([raw,raw.assign(seed=2)],ignore_index=True),[1,2],{'A':1,'B':-1})
        self.assertEqual(one,two);self.assertEqual(one['delta_BA'],-.5)
        self.assertGreaterEqual(one['IC95_inferior'],-1.)
        self.assertLessEqual(one['IC95_superior'],0.)

    def test_reaggregation_detects_modified_person_scores_and_metric_rows(self):
        meta=pd.DataFrame([{'sample_id':f'S{i:03d}_{j}','participant_id':f'S{i:03d}','task':str(j),'y':int(i%4!=0)}
                           for i in range(40) for j in range(2)])
        train,test=next(AUDIT.splits(meta,20261105))
        frames=AUDIT.EXP25.prediction_frames(meta,train,test,np.linspace(-2.,2.,len(train)),np.linspace(-2.,2.,len(test)),'A',20261105,1)
        auditor=AUDIT.Auditor(Path('.'));auditor.meta=meta
        auditor.check_predictions({'frames':frames},train,test,20261105,1,'A')
        changed=copy.deepcopy(frames);changed['predicoes.csv'].score+=.1
        with self.assertRaises(AssertionError):auditor.check_predictions({'frames':changed},train,test,20261105,1,'A')
        changed=copy.deepcopy(frames);changed['metricas.csv'].loc[0,'BA']+=.01
        with self.assertRaises(AssertionError):auditor.check_predictions({'frames':changed},train,test,20261105,1,'A')

    def test_incomplete_experiment_refuses_before_loading_data_or_using_gpu(self):
        with TemporaryDirectory(prefix='audit25_26_') as temporary:
            auditor=AUDIT.Auditor(Path(temporary),'cuda')
            with patch.object(auditor,'load_data',side_effect=RuntimeError('must not load')) as loader:
                with self.assertRaisesRegex(AssertionError,'ainda não completo'):auditor.audit(25)
                loader.assert_not_called()

    def test_hash_map_rejects_file_tampering_and_path_traversal(self):
        with TemporaryDirectory(prefix='audit_hash_') as temporary:
            folder=Path(temporary);path=folder/'x.txt';path.write_text('ok',encoding='utf8')
            digest=AUDIT.sha(path)
            auditor=AUDIT.Auditor(folder);auditor.verify_map(folder,{'x.txt':digest})
            path.write_text('altered',encoding='utf8')
            # Um auditor novo não compartilha a memoização de leitura anterior.
            with self.assertRaises(AssertionError):AUDIT.Auditor(folder).verify_map(folder,{'x.txt':digest})
            with self.assertRaises(AssertionError):auditor.verify_map(folder,{'../x.txt':digest})

    def test_end_to_end_audit_of_synthetic_complete_outputs_without_optimizer_updates(self):
        exp=AUDIT.EXP25
        exp.configure_torch('cpu',threads=1)
        rng=np.random.default_rng(251126)
        meta=pd.DataFrame([{'sample_id':f'P{i:03d}_{j}','participant_id':f'P{i:03d}','task':str(j),'y':int(i%4!=0)}
                           for i in range(40) for j in range(2)])
        inputs={'raw':rng.normal(size=(len(meta),480,8)).astype(np.float32),
                'rms':rng.normal(size=(len(meta),480,8)).astype(np.float32)}
        options={**exp.DEFAULT_OPTIONS,'steps':8,'max_epochs':1,'min_epochs':1,'patience':1}
        arms=exp.ARMS[:2]
        contrasts={'B-A_synthetic':{arms[1]:1,arms[0]:-1}}
        with TemporaryDirectory(prefix='audit_integrated_') as temporary,patch.object(exp,'ARMS',arms),patch.object(exp,'CONTRASTS',contrasts):
            root=Path(temporary);out=root/exp.OUT_REL;out.mkdir(parents=True)
            protocol=exp.protocol(options)
            identity={'configuration':{'arms':list(arms),'seeds':list(exp.SEEDS),**options},'code':{},'inputs':{},
                      'protocol_sha256':hashlib.sha256(protocol.encode()).hexdigest()}
            exp.EXP17.text_write(out/'PROTOCOLO.txt',protocol)
            exp.EXP17.json_write(out/'assinatura_execucao.json',identity)
            exp.EXP17.csv_write(out/'particoes_participantes.csv',exp.EXP19.fold_table(meta,exp.SEEDS))
            signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            # Apenas inferência sobre dados artificiais: nenhuma atualização de pesos.
            with patch.object(exp,'train_epoch',return_value=0.):
                for seed in exp.SEEDS:
                    for fold,(train,test) in enumerate(exp.EXP13.row_splits(meta,5,seed),1):
                        for arm in arms:
                            result=exp.train_arm(inputs,meta,train,test,seed,fold,arm,'cpu',options)
                            exp.save_checkpoint(out/'checkpoints'/f'seed_{seed}_fold_{fold:02d}'/arm,result,signature)
            exp.consolidate(out,meta,exp.SEEDS,arms,lambda _:None)
            auditor=AUDIT.Auditor(root);auditor.inputs=inputs;auditor.meta=meta
            result=auditor.audit(25)
            self.assertEqual(result['checkpoints'],60)
            self.assertEqual(auditor.totals['replayed_models'],2)
            self.assertEqual(auditor.totals['checkpoint_files_hashed'],60*len(exp.ARTIFACTS))


if __name__=='__main__':unittest.main()
