# -*- coding: utf-8 -*-
"""Busca de modelos Wavelet com CV ANINHADA, avaliada por participante.

SEM EXTRAIR WAV. Pre-processamento, representacao, atributos, classificador
(e opcionalmente limiar) escolhidos SOMENTE no treino de cada fold externo.
Saidas em pasta separada, checkpoint por fold+braco.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir
import argparse
import collections
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from avfad_core import (AVFADError,load_data,input_signature,representations,
    participant_table,make_folds,transform,row_weights,model_spec,build_model,aggregate_participants,
    scores,collapse,threshold_from_inner,metrics,atomic_json)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dados',default='.',help='Pasta dos arquivos .npz e metadata (padrao: atual)')
    p.add_argument('--saida',default='resultados_otimizacao_wavelet',help='Pasta exclusiva de resultados')
    p.add_argument('--preset',choices=['teste','rapido','completo'],default='rapido')
    p.add_argument('--baselines',choices=['nenhum','mfcc','compare','todos'],default='mfcc')
    p.add_argument('--limiar',choices=['interno','zero'],default='interno',help='Escolher corte com CV interna ou fixar 0')
    p.add_argument('--coorte',default=None,help='CSV participant_id e opcional pair_id, dentro da coorte HSA')
    p.add_argument('--tarefa',choices=['todas','004','005','006','007','008','009'],default='todas')
    p.add_argument('--agregacao',choices=['frases','media_pessoa','media_desvio_pessoa'],default='frases',help='Amostras por frase ou estatisticas entre frases por participante')
    p.add_argument('--folds-externos',type=int,default=5)
    p.add_argument('--folds-internos',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--somente-verificar',action='store_true')
    return p.parse_args()


def print_summary(m,matrices,persons,options):
    print(f'Gravacoes: {len(m)}, participantes: {len(persons)}, classes: {persons.y.value_counts().to_dict()}',flush=True)
    print('Matrizes:',', '.join(f'{k} {v[0].shape}' for k,v in matrices.items()),flush=True)
    print(f'Preset: {options.preset} | baselines: {options.baselines} | limiar: {options.limiar}',flush=True)


def evaluate_arm(arm,options,meta,persons,matrices,wave_reps,outer_fold,train_pids,test_pids,output,signature):
    file=output/f'checkpoint_{arm}_fold{outer_fold:02d}.json'
    if file.exists():
        prior=json.loads(file.read_text(encoding='utf-8'))
        if prior.get('signature') != signature:
            raise AVFADError(f'Checkpoint diferente em {file}; use outra --saida para outra configuracao.')
        print(f'  {arm} fold {outer_fold}: checkpoint existente',flush=True)
        return prior
    p_train=persons[persons.participant_id.isin(train_pids)].copy()
    if p_train.y.value_counts().min() < options.folds_internos:
        raise AVFADError('Nao ha exemplos suficientes de ambas as classes para CV interna.')
    train_mask=meta.participant_id.isin(train_pids).to_numpy()
    test_mask=meta.participant_id.isin(test_pids).to_numpy()
    outer_train_meta=meta.loc[train_mask].reset_index(drop=True)
    outer_test_meta=meta.loc[test_mask].reset_index(drop=True)
    if arm=='WAVELET':
        X=matrices['WAVELET'][0]
        rep_idx=wave_reps
    elif arm=='MFCC':
        X=matrices['MFCC'][0];rep_idx={'MFCC':np.arange(X.shape[1])}
    elif arm=='COMPARE':
        X=matrices['COMPARE'][0];rep_idx={'ComParE2016':np.arange(X.shape[1])}
    else:
        raise AVFADError('Braco desconhecido')
    x_outer=X[train_mask]
    all_specs=model_spec(options.preset)
    kvals={'teste':[32],'rapido':[64,128],'completo':[32,64,128,256]}[options.preset]
    inner_splits=list(make_folds(p_train,options.folds_internos,options.seed+outer_fold*37))
    results=collections.defaultdict(list)
    start=time.monotonic()
    n_fit=0
    for rep,ix in rep_idx.items():
        for kval in sorted(set(min(k,len(ix)) for k in kvals)):
            for inner_num,(inner_train_ids,inner_test_ids) in enumerate(inner_splits,1):
                tr=outer_train_meta.participant_id.isin(inner_train_ids).to_numpy()
                te=outer_train_meta.participant_id.isin(inner_test_ids).to_numpy()
                train_meta=outer_train_meta.loc[tr].reset_index(drop=True)
                valid_meta=outer_train_meta.loc[te].reset_index(drop=True)
                xtr,xva,actual_k=transform(x_outer[np.ix_(tr,ix)],x_outer[np.ix_(te,ix)],train_meta.y.to_numpy(),kval)
                w=row_weights(train_meta)
                for model_name,kind,c in all_specs:
                    model=build_model(kind,c)
                    model.fit(xtr,train_meta.y.to_numpy(),sample_weight=w)
                    collapsed=collapse(valid_meta,scores(model,xva))
                    results[(rep,kval,model_name)].append(collapsed)
                    n_fit+=1
            print(f'    {arm} fold {outer_fold}: {rep}, k={kval} avaliado',flush=True)
    # Escolha de representacao + k + classificador + corte a partir de OOF INTERNO.
    ranking=[]
    for (rep,kval,model_name),inner_data in results.items():
        inner_pred=pd.concat(inner_data,ignore_index=True)
        if len(inner_pred)!=len(p_train) or inner_pred.participant_id.duplicated().any():
            raise AVFADError('CV interna nao produziu uma predicao unica por participante.')
        threshold=threshold_from_inner(inner_pred.y.to_numpy(),inner_pred.score.to_numpy(),options.limiar)
        mm=metrics(inner_pred.y.to_numpy(),inner_pred.score.to_numpy(),threshold)
        ranking.append({'representation':rep,'k':int(kval),'model':model_name,
                        'inner_ba':mm['ba'],'inner_auc':mm['auc'],
                        'inner_threshold':float(threshold), 'inner_sensitivity':mm['sensitivity'],
                        'inner_specificity':mm['specificity']})
    ranking.sort(key=lambda z:(-z['inner_ba'],-z['inner_auc'],abs(z['inner_threshold']),z['representation'],z['k'],z['model']))
    best=ranking[0]
    rep,kval=best['representation'],best['k']
    model_name=best['model']
    kind,c=next((kind,c) for mn,kind,c in all_specs if mn==model_name)
    ix=rep_idx[rep]
    full_tr,full_te,selected_n=transform(x_outer[:,ix],X[np.ix_(test_mask,ix)],outer_train_meta.y.to_numpy(),kval)
    model=build_model(kind,c)
    model.fit(full_tr,outer_train_meta.y.to_numpy(),sample_weight=row_weights(outer_train_meta))
    pred=collapse(outer_test_meta,scores(model,full_te))
    threshold=float(best['inner_threshold'])
    pred['pred']=(pred.score>=threshold).astype(int)
    pred['threshold']=threshold
    pred['arm']=arm
    pred['fold']=outer_fold
    pred['representation']=rep
    pred['model']=model_name
    pred['k']=int(selected_n)
    test_metrics=metrics(pred.y.to_numpy(),pred.score.to_numpy(),threshold)
    # Alem da predicao ajustada, guardar a avaliacao fixa em zero para transparencia.
    zero_metrics=metrics(pred.y.to_numpy(),pred.score.to_numpy(),0.0)
    artifact={'signature':signature,'arm':arm,'fold':outer_fold,'selected':best,
              'train_n_people':len(p_train),'test_n_people':len(pred),'inner_candidates':len(ranking),
              'top5_inner':ranking[:5], 'out_metrics':test_metrics, 'out_zero':zero_metrics,
              'predictions':pred.to_dict(orient='records'),
              'n_fit':n_fit,'duration_sec':round(time.monotonic()-start,2)}
    atomic_json(file,artifact)
    print(f'  {arm} fold {outer_fold}: {rep} {model_name} k={selected_n} '
          f'corte={threshold:.3f} | BA externo={test_metrics["ba"]:.3f} '
          f'| {n_fit} ajustes | {artifact["duration_sec"]:.1f}s',flush=True)
    return artifact


def main():
    o=parse_args()
    if o.folds_externos<2 or o.folds_internos<2:
        raise AVFADError('Folds devem ser >= 2.')
    root=Path(o.dados).resolve()
    output=Path(o.saida)
    if not output.is_absolute():output=asset(root, output)
    base= o.baselines != 'nenhum'
    meta,matrices,paths=load_data(root,require_baselines=base)
    if o.tarefa!='todas':
        mask=meta.task.eq(o.tarefa).to_numpy()
        meta=meta.loc[mask].reset_index(drop=True)
        matrices={k:(x[mask],names) for k,(x,names) in matrices.items()}
    if o.agregacao!='frases':
        meta,matrices=aggregate_participants(meta,matrices,o.agregacao)
    cohort_path=Path(o.coorte).resolve() if o.coorte else None
    persons=participant_table(meta,cohort_path)
    # Filtrar a matriz por coorte somente apos verificar os grupos.
    keep=meta.participant_id.isin(persons.participant_id).to_numpy()
    meta=meta.loc[keep].reset_index(drop=True)
    matrices={k:(x[keep],names) for k,(x,names) in matrices.items()}
    reps=representations(matrices['WAVELET'][1],o.preset)
    arms=['WAVELET']
    if o.baselines in ('mfcc','todos'):arms.append('MFCC')
    if o.baselines in ('compare','todos'):arms.append('COMPARE')
    for arm in arms:
        if arm not in matrices:raise AVFADError(f'Baseline {arm} nao disponivel no NPZ.')
    if persons.y.value_counts().min() < o.folds_externos*o.folds_internos // 2:
        print('AVISO: poucos participantes minoritarios para o numero de folds solicitado.',flush=True)
    signature_settings={k:v for k,v in vars(o).items() if k not in ('somente_verificar','saida','dados')}
    signature={'inputs':input_signature(paths+([cohort_path] if cohort_path else [])),
               'settings':signature_settings,'script_version':'nested-wavelet-2026-11-05-v1'}
    print_summary(meta,matrices,persons,o)
    print('Representacoes wavelet:',{k:len(v) for k,v in reps.items()},flush=True)
    if o.somente_verificar:
        print('Arquivos e alinhamento validados. Nenhum treinamento realizado.',flush=True)
        return
    output.mkdir(parents=True,exist_ok=True)
    signature_file=output/'configuracao_execucao.json'
    if signature_file.exists() and json.loads(signature_file.read_text(encoding='utf-8'))!=signature:
        raise AVFADError(f'Saida existente com outro experimento: {output}. Escolha outro --saida.')
    atomic_json(signature_file,signature)
    folds=list(make_folds(persons,o.folds_externos,o.seed))
    full=[]
    started=time.monotonic()
    for fold,(train_ids,test_ids) in enumerate(folds,1):
        overlap=set(train_ids)&set(test_ids)
        if overlap:raise AVFADError(f'Vazamento entre treino/teste em fold {fold}.')
        print(f'\n=== FOLD EXTERNO {fold}/{len(folds)}: treino={len(train_ids)} pessoas, teste={len(test_ids)} ===',flush=True)
        for arm in arms:
            art=evaluate_arm(arm,o,meta,persons,matrices,reps,fold,train_ids,test_ids,output,signature)
            full.append(art)
        # Exportacao parcial segura apos cada fold externo terminado.
        write_aggregates(output,full,arms,persons,final=False)
    write_aggregates(output,full,arms,persons,final=True)
    print(f'\nConcluido: {output} | tempo total desta execucao: {(time.monotonic()-started)/60:.1f} min',flush=True)
    print('Use: python 02_resumir_resultados.py --saida "'+str(output)+'"',flush=True)


def write_aggregates(output,artifacts,arms,persons,final=False):
    entries=[];fold_rows=[]
    for art in artifacts:
        entries.extend(art['predictions'])
        fold_rows.append({'arm':art['arm'],'fold':art['fold'],**art['selected'],
                          **{f'external_{k}':v for k,v in art['out_metrics'].items()},
                          'n_people':art['test_n_people'],'fits':art['n_fit'],'seconds':art['duration_sec']})
    preds=pd.DataFrame(entries)
    folds=pd.DataFrame(fold_rows)
    if not preds.empty:
        preds.to_csv(output/'predicoes_por_participante.csv',index=False,encoding='utf-8-sig')
    if not folds.empty:
        folds.to_csv(output/'escolhas_por_fold.csv',index=False,encoding='utf-8-sig')
    if final:
        for arm in arms:
            subset=preds[preds.arm==arm]
            if subset.participant_id.duplicated().any() or len(subset)!=len(persons):
                raise AVFADError('As predicoes externas nao cobrem todos os participantes uma vez por braco.')
        rows=[]
        for arm in arms:
            d=preds[preds.arm==arm]
            rows.append({'arm':arm,'n_pessoas':len(d),**metrics(d.y,d.score,0.0),
                         'ba_com_limiar_interno':metrics(d.y,(d.score-d.threshold),0.0)['ba'],
                         'sens_com_limiar_interno':metrics(d.y,(d.score-d.threshold),0.0)['sensitivity'],
                         'esp_com_limiar_interno':metrics(d.y,(d.score-d.threshold),0.0)['specificity']})
        pd.DataFrame(rows).to_csv(output/'resumo_oof_preliminar.csv',index=False,encoding='utf-8-sig')
        print('\n=== RESUMO OOF: limiar selecionado internamente ===',flush=True)
        for r in rows:
            print(f'  {r["arm"]:7s}: BA {r["ba_com_limiar_interno"]:.3f} | '
                  f'Sens {r["sens_com_limiar_interno"]:.3f} | Esp {r["esp_com_limiar_interno"]:.3f} | '
                  f'AUC {r["auc"]:.3f}',flush=True)


if __name__=='__main__':
    try:main()
    except (AVFADError,FileNotFoundError,ValueError) as e:
        raise SystemExit(f'ERRO: {e}')
