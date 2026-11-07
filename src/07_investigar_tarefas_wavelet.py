# -*- coding: utf-8 -*-
"""AVFAD CAPE-V: diagnostico da informacao WPT por tarefa (exploratorio).

Usa SOMENTE matrizes NPZ ja extraidas; nao le WAVs. Modelo predefinido:
WPT sym5, SelectKBest(128), StandardScaler, SVC linear C=1,
class_weight='balanced', limiar 0. Todos os ajustes apenas no treino.

As mesmas pessoas e folds sao usadas em TODOS os modos para comparabilidade.
Os resultados nao constituem teste clinico externo e nao autorizam escolher
'a melhor frase' e reportar sua performance como estimativa imparcial.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from avfad_core import load_data, representations, transform, collapse, metrics

TASKS = ('004','005','006','007','008','009')
MODES = (*TASKS,'TODAS_POOLED','MEDIA_PESSOA','MEDIA_DP_PESSOA')
VERSION = '1.0'


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados',default='.',help='Pasta com NPZ e CSV de entrada')
    p.add_argument('--saida',default='resultados_07_tarefas_wavelet',help='Pasta de saida isolada')
    p.add_argument('--repeticoes',type=int,default=5,help='Seeds consecutivas, padrao 5')
    p.add_argument('--primeira-seed',type=int,default=101)
    p.add_argument('--incluir-mfcc',action='store_true',help='Tambem avaliar MFCC; Wavelet continua como objetivo')
    a=p.parse_args()
    if a.repeticoes<1 or a.repeticoes>30:p.error('Use 1 a 30 repeticoes')
    if a.primeira_seed<0:p.error('A primeira seed deve ser >=0')
    return a


def fingerprint(paths,opts):
    items=[]
    for f in paths:
        f=Path(f);s=f.stat();items.append((str(f.resolve()),s.st_size,s.st_mtime_ns))
    return hashlib.sha256(json.dumps({'ver':VERSION,'inputs':items,'opcoes':opts},sort_keys=True).encode()).hexdigest()


def dataset(meta,x,mode,ids):
    """Monta entradas sem rótulo, apenas a partir de gravacoes de cada pessoa."""
    if mode in TASKS:
        sel=meta.task.to_numpy()==mode
        return meta.loc[sel].reset_index(drop=True),x[sel]
    if mode=='TODAS_POOLED':
        return meta,x
    positions={pid:np.flatnonzero(meta.participant_id.to_numpy()==pid) for pid in ids}
    features=[]
    for pid in ids:
        a=x[positions[pid]]
        if a.shape[0]!=6:raise RuntimeError(f'Pessoa {pid} sem 6 frases')
        if mode=='MEDIA_PESSOA':
            feat=np.mean(a,axis=0)
        elif mode=='MEDIA_DP_PESSOA':
            feat=np.r_[np.mean(a,axis=0),np.std(a,axis=0)]
        else:
            raise RuntimeError(mode)
        features.append(feat)
    meta_p=meta.drop_duplicates('participant_id').set_index('participant_id').loc[ids].reset_index()
    meta_p=meta_p.assign(sample_id=lambda d:d.participant_id+'__'+mode,task=mode)
    return meta_p, np.asarray(features,dtype=np.float32)


def atomic_csv(df,p):
    p=Path(p)
    tmp=p.with_suffix(p.suffix+'.tmp')
    df.to_csv(tmp,index=False,encoding='utf-8-sig')
    tmp.replace(p)


def run_one(meta,x,people,mode,seed,arm):
    persons=people.participant_id.to_numpy()
    y=people.y.to_numpy()
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
    pred=[];fold_rows=[]
    one_meta,one_x=dataset(meta,x,mode,persons)
    if len(one_meta.participant_id.unique())!=len(people):
        raise RuntimeError(f'{mode}: pessoas faltando')
    for fold,(ptr,pte) in enumerate(cv.split(persons,y),1):
        train_ids=set(persons[ptr]);test_ids=set(persons[pte])
        if train_ids & test_ids: raise RuntimeError('Pessoa em treino e teste')
        tr=one_meta.participant_id.isin(train_ids).to_numpy()
        te=one_meta.participant_id.isin(test_ids).to_numpy()
        xt,xv,n=transform(one_x[tr],one_x[te],one_meta.loc[tr,'y'].to_numpy(),128)
        model=SVC(kernel='linear',C=1.0,class_weight='balanced',cache_size=512)
        model.fit(xt,one_meta.loc[tr,'y'].to_numpy())
        result=collapse(one_meta.loc[te].reset_index(drop=True),model.decision_function(xv))
        if set(result.participant_id)!=test_ids:raise RuntimeError('Pessoas esperadas nao encontradas')
        result['arm']=arm;result['modo']=mode;result['seed']=seed;result['fold']=fold
        fm=metrics(result.y,result.score,0.0)
        fold_rows.append(dict(arm=arm,modo=mode,seed=seed,fold=fold,n_pessoas=len(result),
                              n_normais=int((result.y==0).sum()),n_patologicos=int((result.y==1).sum()),
                              k_usado=n,**fm))
        pred.append(result)
    frame=pd.concat(pred,ignore_index=True)
    if frame.participant_id.duplicated().any() or len(frame)!=len(people):
        raise RuntimeError('Predicao OOF incompleta/duplicada')
    m=metrics(frame.y,frame.score,0.0)
    return frame,pd.DataFrame(fold_rows),dict(arm=arm,modo=mode,seed=seed,n_pessoas=len(people),**m)


def main():
    a=cli();root=Path(a.dados).resolve();dest=Path(a.saida)
    if not dest.is_absolute():dest=asset(root, dest)
    dest.mkdir(parents=True,exist_ok=True)
    ck=dest/'checkpoints';ck.mkdir(exist_ok=True)
    meta,mats,paths=load_data(root,require_baselines=a.incluir_mfcc)
    wave_idx=representations(mats['WAVELET'][1],'teste')['WPT_sym5']
    # As analises por frase devem usar EXATAMENTE a mesma coorte de 410 pessoas.
    counts=meta.groupby('participant_id').task.nunique()
    ids=sorted(counts[counts==6].index)
    meta=meta[meta.participant_id.isin(ids)].reset_index(drop=True)
    valid=meta.groupby('participant_id').task.agg(lambda g:set(g)==set(TASKS))
    if len(ids)!=410 or not valid.all() or len(meta)!=2460:
        raise RuntimeError('Esperava 410 pessoas com 6 tarefas exatas, 2460 gravacoes')
    people=meta[['participant_id','y']].drop_duplicates().sort_values('participant_id').reset_index(drop=True)
    if people.y.value_counts().min()<5:raise RuntimeError('Poucas pessoas para 5 folds')
    arms={}
    # Selecionar a mesma ordem do manifesto original para manter alinhamento NPZ.
    # Recarregar a metadata original via ids para manter a ordem exata do NPZ.
    original_meta=pd.read_csv(asset(root, 'metadata_wavelet_temporal_avfad_reduzido.csv'))
    original_meta['participant_id']=original_meta.participant_id.astype(str).str.strip().str.upper()
    keep=original_meta.participant_id.isin(ids).to_numpy()
    original_order=original_meta.loc[keep,'sample_id'].astype(str).str.strip().str.upper().to_numpy()
    if not np.array_equal(meta.sample_id.to_numpy(),original_order):
        raise RuntimeError('Desalinhamento apos filtro de coorte')
    arms['WPT_sym5']=np.asarray(mats['WAVELET'][0][keep][:,wave_idx],dtype=np.float32)
    if a.incluir_mfcc:
        arms['MFCC']=np.asarray(mats['MFCC'][0][keep],dtype=np.float32)
    opts={'repeticoes':a.repeticoes,'primeira-seed':a.primeira_seed,'modos':MODES,'arms':list(arms),
          'k':128,'classifier':'SVC_linear_C1_balanced','threshold':0.0,'complete_cases':410}
    sig=fingerprint(paths,opts)
    cfg=dest/'configuracao.json'
    if cfg.exists():
        old=json.loads(cfg.read_text(encoding='utf8'))
        if old.get('assinatura')!=sig:
            raise RuntimeError('Pasta de saida pertence a outra configuracao. Use --saida novo.')
    else:
        cfg.write_text(json.dumps({'assinatura':sig,'parametros':opts,'arquivos':[str(p) for p in paths]},
                                  indent=2,ensure_ascii=False),encoding='utf8')
    print('Coorte comum:',len(people),'pessoas;',len(meta),'frases; normal=',sum(people.y==0),
          'patologico=',sum(people.y==1),flush=True)
    print('Folds fixos por pessoa em cada seed; modelo WPT predefinido; sem escolher vencedor.',flush=True)
    for arm,x in arms.items():
        for seed in range(a.primeira_seed,a.primeira_seed+a.repeticoes):
            for mode in MODES:
                name=f'{arm}_seed{seed}_{mode}'
                out=ck/(name+'_resultado.csv');predfile=ck/(name+'_predicoes.csv');foldfile=ck/(name+'_folds.csv')
                if all(p.exists() for p in [out,predfile,foldfile]):
                    print(name,'checkpoint',flush=True);continue
                p,f,r=run_one(meta,x,people,mode,seed,arm)
                atomic_csv(p,predfile);atomic_csv(f,foldfile);atomic_csv(pd.DataFrame([r]),out)
                print(f'{name}: BA={r["ba"]:.4f} AUC={r["auc"]:.4f} sens={r["sensitivity"]:.3f} esp={r["specificity"]:.3f}',flush=True)
    rs=[];ps=[];fs=[]
    for arm in arms:
        for seed in range(a.primeira_seed,a.primeira_seed+a.repeticoes):
            for mode in MODES:
                name=f'{arm}_seed{seed}_{mode}'
                rs.append(pd.read_csv(ck/(name+'_resultado.csv'),dtype={'modo':str}))
                ps.append(pd.read_csv(ck/(name+'_predicoes.csv'),dtype={'modo':str}))
                fs.append(pd.read_csv(ck/(name+'_folds.csv'),dtype={'modo':str}))
    r=pd.concat(rs,ignore_index=True);p=pd.concat(ps,ignore_index=True);f=pd.concat(fs,ignore_index=True)
    atomic_csv(r,dest/'desempenho_por_repeticao.csv')
    atomic_csv(p,dest/'predicoes_por_participante.csv')
    atomic_csv(f,dest/'desempenho_por_fold.csv')
    agg=r.groupby(['arm','modo']).agg(ba_media=('ba','mean'),ba_min=('ba','min'),ba_max=('ba','max'),
                                     ba_dp=('ba','std'),auc_media=('auc','mean'),
                                     sens_media=('sensitivity','mean'),esp_media=('specificity','mean')).reset_index()
    atomic_csv(agg,dest/'resumo_tarefas.csv')
    msg=['# AVFAD — Estudo exploratorio de tarefas e agregacao WPT', '',
         f'- Pessoas com todas as seis tarefas: {len(people)}; frases: {len(meta)}.',
         f'- Normais: {(people.y==0).sum()}; patologicos: {(people.y==1).sum()}.',
         f'- Repeticoes: {a.repeticoes}; sementes: {a.primeira_seed} a {a.primeira_seed+a.repeticoes-1}.',
         '- Modelo FIXO: WPT sym5, SVC linear C=1, balanceado, SelectKBest(128), limiar zero.',
         '- Mesmos folds e pessoas para todas as variantes; selecao, scaler e modelo ajustados dentro de cada treino.',
         '- Tarefas individuais treinam em uma frase por pessoa; pooled treina nas seis e agrega score por pessoa.',
         '- MEDIA_PESSOA e MEDIA_DP_PESSOA treinam em uma linha agregada por pessoa.',
         '- AVISO: nao selecionar a melhor tarefa dos mesmos folds e divulga-la como resultado de generalizacao.',
         '- Repeticoes reutilizam as mesmas pessoas; desvio entre seeds nao e intervalo de confianca.',
         '- As representacoes Wavelet usam VAD/janelas proprios; comparacao MFCC nao isola exclusivamente a transformada.',
         '', '## Desempenho medio entre sementes', '',
         '| Representacao | Modo | BA media | BA min | BA max | AUC media | Sens media | Esp media |',
         '|---|---|---:|---:|---:|---:|---:|---:|']
    for z in agg.itertuples(index=False):
        msg.append(f'| {z.arm} | {z.modo} | {z.ba_media:.4f} | {z.ba_min:.4f} | {z.ba_max:.4f} | {z.auc_media:.4f} | {z.sens_media:.4f} | {z.esp_media:.4f} |')
    msg+=['','## Leitura recomendada','',
          'Interpretar quais tarefas aparentam carregar sinal e se agregar atributos ajuda sem aumentar a capacidade do classificador.',
          'Resultado alto em tarefa escolhida depois de ver a tabela e exploratorio, nao evidencia final.',
          'A proxima etapa deve testar uma hipotese preespecificada em validacao aninhada ou coorte externa.']
    (dest/'RELATORIO_07.md').write_text('\n'.join(msg)+'\n',encoding='utf8')
    print('\n=== RESUMO DE TODAS AS VARIANTES ===')
    for z in agg.itertuples(index=False):
        print(f'{z.arm:9s} | {str(z.modo):15s} | BA media={z.ba_media:.4f} range={z.ba_min:.4f}..{z.ba_max:.4f} | AUC={z.auc_media:.4f}',flush=True)
    print('Relatorio:',dest/'RELATORIO_07.md')
    print('Nao escolha melhor frase/seed para reportar sem nova validacao independente.')

if __name__=='__main__':main()
