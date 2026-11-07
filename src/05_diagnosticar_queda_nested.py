# -*- coding: utf-8 -*-
"""Diagnostico da queda WPT sym5 no nested CV (sem WAVs; somente .npz).

Fixa WPT sym5, k=128, SVC linear C=1, limiar=0. Varia SOMENTE:
- folds externos: StratifiedGroupKFold (benchmark anterior) / StratifiedKFold
  no nivel de participante (otimizacao aninhada recente);
- pesos: class_weight='balanced' (benchmark anterior) / pesos balanceados
  por participante (otimizacao aninhada recente).
Compara com o resultado da selecao nested, quando CSV estiver disponivel.

Resultados de diagnostico; NAO sao um novo teste externo independente.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC
from avfad_core import (load_data, representations, transform, row_weights,
                        collapse, metrics)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dados',default='.',help='Pasta do NPZ/CSV de origem')
    ap.add_argument('--saida',default='resultados_diagnostico_queda_nested')
    ap.add_argument('--comparar-nested',default='resultados_otimizacao_wavelet/predicoes_por_participante.csv')
    a=ap.parse_args()
    root=Path(a.dados).resolve()
    dest=Path(a.saida)
    if not dest.is_absolute():dest=asset(root, dest)
    dest.mkdir(parents=True,exist_ok=True)
    meta,mats,paths=load_data(root,require_baselines=False)
    people=meta[['participant_id','y']].drop_duplicates().sort_values('participant_id').reset_index(drop=True)
    if len(people)!=411 or set(people.y.unique())!={0,1}:raise RuntimeError('Coorte diferente da esperada')
    reps=representations(mats['WAVELET'][1],'teste')
    x=mats['WAVELET'][0]
    wi=reps['WPT_sym5']
    print(f'WPT_sym5: {len(wi)} atributos; amostras {len(meta)}; pessoas {len(people)}',flush=True)
    print('CONFIGURACAO FIXA: WPT_sym5, SelectKBest 128, SVC linear C=1, limiar zero',flush=True)
    print('Comparacao entre dois esquemas de folds e dois balanceamentos. SEM escolher vencedor.',flush=True)
    reports=[]; predictions=[]; fold_groups={}
    for split_type in ('historico_SGKF','recente_SKF'):
        if split_type=='historico_SGKF':
            s=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
            folds=list(s.split(np.zeros((len(people),1),dtype=np.uint8),people.y.to_numpy(),groups=people.participant_id.to_numpy()))
        else:
            s=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
            folds=list(s.split(people.participant_id,people.y))
        pid_to_fold={str(people.iloc[ix].participant_id):n for n,(_,te) in enumerate(folds,1) for ix in te}
        fold_groups[split_type]=pid_to_fold
        for weight_type in ('class_weight_historico','sample_weight_recente'):
            observed=[]
            fold_rows=[]
            for fold,(trp,tep) in enumerate(folds,1):
                ptrain=people.iloc[trp].participant_id
                ptest=people.iloc[tep].participant_id
                tr=meta.participant_id.isin(ptrain).to_numpy()
                te=meta.participant_id.isin(ptest).to_numpy()
                assert not (set(meta.loc[tr,'participant_id']) & set(meta.loc[te,'participant_id']))
                xtrain,xtest,k=transform(x[np.ix_(tr,wi)],x[np.ix_(te,wi)],meta.loc[tr,'y'].to_numpy(),128)
                if weight_type=='class_weight_historico':
                    model=SVC(kernel='linear',C=1.0,class_weight='balanced',cache_size=512)
                    model.fit(xtrain,meta.loc[tr,'y'].to_numpy())
                else:
                    model=SVC(kernel='linear',C=1.0,cache_size=512)
                    model.fit(xtrain,meta.loc[tr,'y'].to_numpy(),sample_weight=row_weights(meta.loc[tr].reset_index(drop=True)))
                out=collapse(meta.loc[te].reset_index(drop=True),model.decision_function(xtest))
                out['fold']=fold
                observed.append(out)
                mm=metrics(out.y.to_numpy(),out.score.to_numpy(),0)
                fold_rows.append({'fold':fold,'test_people':len(out),'normal':int(sum(out.y==0)),
                                  'pathological':int(sum(out.y==1)),**mm})
                print(f'  {split_type} | {weight_type} | fold {fold}: BA={mm["ba"]:.3f}, sens={mm["sensitivity"]:.3f}, esp={mm["specificity"]:.3f}',flush=True)
            all_oof=pd.concat(observed,ignore_index=True)
            assert len(all_oof)==411 and not all_oof.participant_id.duplicated().any()
            mm=metrics(all_oof.y.to_numpy(),all_oof.score.to_numpy(),0)
            print(f'=> {split_type} | {weight_type}: BA={mm["ba"]:.4f} AUC={mm["auc"]:.4f} Sens={mm["sensitivity"]:.4f} Esp={mm["specificity"]:.4f}',flush=True)
            reports.append({'folds':split_type,'pesos':weight_type,**mm})
            all_oof['folds']=split_type
            all_oof['pesos']=weight_type
            predictions.append(all_oof)
            pd.DataFrame(fold_rows).to_csv(dest/f'folds_{split_type}_{weight_type}.csv',index=False,encoding='utf-8-sig')
    summary=pd.DataFrame(reports)
    summary.to_csv(dest/'resumo_diagnostico.csv',index=False,encoding='utf-8-sig')
    pd.concat(predictions,ignore_index=True).to_csv(dest/'predicoes_diagnostico.csv',index=False,encoding='utf-8-sig')
    h=fold_groups['historico_SGKF'];n=fold_groups['recente_SKF']
    disagreements=sum(h[pid]!=n[pid] for pid in h)
    # Porcentagem de discordancia no NUMERO do fold nao e uma medida invariante a permutacao dos rotulos.
    # A matriz de intersecao abaixo permite comparar as particoes concretas.
    intersection=pd.crosstab(pd.Series(h,name='fold_historico'),pd.Series(n,name='fold_recente'))
    intersection.to_csv(dest/'intersecao_folds.csv',encoding='utf-8-sig')
    print('\n=== FOLDS HISTORICO vs RECENTE: participantes em comum por fold ===',flush=True)
    print(intersection.to_string(),flush=True)
    npred=Path(a.comparar_nested)
    if not npred.is_absolute():npred=asset(root, npred)
    if npred.exists():
        nested=pd.read_csv(npred)
        nested=nested[nested.arm=='WAVELET'].copy()
        if len(nested)==411:
            y=nested.y.to_numpy(dtype=int);score=nested.score.to_numpy(dtype=float)
            base=metrics(y,score,0.0)
            tuned=metrics(y,score-nested.threshold.to_numpy(dtype=float),0.0)
            print('\nNested CV, MESMAS matrizes (arquivo ja gerado; selecao interna):',flush=True)
            print(f'  corte zero BA={base["ba"]:.4f}; corte interno BA={tuned["ba"]:.4f}',flush=True)
            print('  Compare com o resultado historico_SGKF+class_weight e recente_SKF+sample_weight fixos acima.',flush=True)
    print(f'\nSalvo em: {dest}',flush=True)
    print('OBS: diagnostico exploratorio. Confirmacao requer avaliacao independente/nested preespecificada.',flush=True)

if __name__=='__main__':main()
