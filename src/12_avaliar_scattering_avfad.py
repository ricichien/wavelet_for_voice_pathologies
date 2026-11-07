# -*- coding: utf-8 -*-
"""Benchmark PREDEFINIDO: Scattering ordem 1+2 vs WPT sym5 vs MFCC, AVFAD HSA.
Cinco folds estratificados por PESSOA, seeds repetidas, fold-preprocessing.
Nao seleciona o melhor resultado como estimativa final imparcial.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold,SelectKBest,f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,confusion_matrix,accuracy_score
import warnings
from scattering_avfad_common import load_manifest, signature, VERSION


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados',default='.')
    p.add_argument('--extracao',default='resultados_10_scattering_full')
    p.add_argument('--saida',default='resultados_12_avaliacao_scattering')
    p.add_argument('--repeticoes',type=int,default=5)
    p.add_argument('--primeira-seed',type=int,default=101)
    p.add_argument('--ablacao',action='store_true',help='Exploratorio: testar S1 e S2 isoladamente')
    a=p.parse_args()
    if not 1<=a.repeticoes<=20:p.error('1..20 repeticoes')
    return a


def read_metadata(path):
    m=pd.read_csv(path,dtype={'sample_id':str,'participant_id':str,'task':str})
    m['sample_id']=m.sample_id.str.strip().str.upper()
    m['participant_id']=m.participant_id.str.strip().str.upper()
    m['y']=m.y.astype(int)
    if m.sample_id.duplicated().any():raise ValueError(f'Duplicados em {path}')
    return m


def aligned_matrix(source_meta,source_matrix,target_meta):
    if set(source_meta.sample_id)!=set(target_meta.sample_id):
        raise ValueError(f'Matrizes com conjuntos diferentes de sample_id: faltam {len(set(target_meta.sample_id)-set(source_meta.sample_id))}')
    by_id=source_meta.set_index('sample_id')
    aligned=by_id.loc[target_meta.sample_id]
    if (aligned.participant_id.to_numpy()!=target_meta.participant_id.to_numpy()).any() or (aligned.y.to_numpy()!=target_meta.y.to_numpy()).any():
        raise ValueError('Atributos, pessoa e rotulo inconsistentes apos alinhamento')
    pos=pd.Series(np.arange(len(source_meta)),index=source_meta.sample_id).loc[target_meta.sample_id].to_numpy()
    return np.asarray(source_matrix[pos],dtype=np.float32)


def load_matrices(root,extract):
    meta=load_manifest(root)
    emeta=read_metadata(extract/'metadata_scattering_avfad.csv')
    if len(emeta)!=len(meta):
        raise ValueError('Esta extracao NAO eh completa (piloto?). Rode 11_extrair_scattering_avfad.py sem --piloto-pessoas.')
    with np.load(extract/'features_scattering_avfad.npz',allow_pickle=False) as d:
        scat=aligned_matrix(emeta,d['scattering'],meta)
        snames=np.asarray(d['feature_names']).astype(str)
    if scat.shape[1]!=len(snames):raise ValueError('Dimensoes scattering e feature_names nao batem')
    orders={x.split('_p',1)[0] for x in snames}
    if not {'S1','S2'}.issubset(orders):raise ValueError('Nao encontrei caminhos S1 e S2')
    with np.load(asset(root, 'features_wavelet_temporal_avfad_reduzido.npz'),allow_pickle=False) as d:
        wpt_all=np.asarray(d['wavelet'],dtype=np.float32)
        wnames=np.asarray(d['feature_names']).astype(str)
    wm=read_metadata(asset(root, 'metadata_wavelet_temporal_avfad_reduzido.csv'))
    wpt_all=aligned_matrix(wm,wpt_all,meta)
    iw=np.flatnonzero(np.char.startswith(wnames.astype(str),'WPT__sym5__'))
    if len(iw)!=1920:raise ValueError(f'Esperava 1920 atributos WPT sym5, encontrei {len(iw)}')
    bm=read_metadata(asset(root, 'metadata_capev_avfad.csv'))
    with np.load(asset(root, 'features_capev_compare_avfad.npz'),allow_pickle=False) as d:
        acoustic=np.asarray(d['acoustic'],dtype=np.float32)
        names=np.asarray(d['acoustic_names']).astype(str)
    im=np.flatnonzero(np.asarray([n.startswith(('mfcc_','delta_','delta2_')) for n in names]))
    if len(im)!=150:raise ValueError(f'Esperava 150 atributos MFCC, achei {len(im)}')
    mfcc=aligned_matrix(bm,acoustic[:,im],meta)
    matrices={
      'SCAT_S1_S2_SVC_linear':(scat,'linear'),
      'WPT_sym5_SVC_linear':(wpt_all[:,iw],'linear'),
      'MFCC_SVC_RBF':(mfcc,'rbf'),
    }
    return meta,matrices,scat,snames


def scaled_features(a,b,y,k=128):
    im=SimpleImputer(strategy='median',keep_empty_features=True)
    at=im.fit_transform(a);bt=im.transform(b)
    var=VarianceThreshold(threshold=0.0)
    at=var.fit_transform(at);bt=var.transform(bt)
    kk=min(k,at.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        sel=SelectKBest(score_func=f_classif,k=kk)
        at=sel.fit_transform(at,y);bt=sel.transform(bt)
    sc=StandardScaler()
    return sc.fit_transform(at),sc.transform(bt)


def summarize(y,score):
    y=np.asarray(y,dtype=int);s=np.asarray(score,dtype=float)
    pred=(s>=0).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'ba':float(balanced_accuracy_score(y,pred)),
            'auc':float(roc_auc_score(y,s)),
            'accuracy':float(accuracy_score(y,pred)),
            'sens':float(tp/(tp+fn)), 'esp':float(tn/(tn+fp)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def atomic_csv(df,p):
    tmp=p.with_suffix('.csv.tmp');df.to_csv(tmp,index=False,encoding='utf-8-sig');tmp.replace(p)


def main():
    a=arguments();root=Path(a.dados).resolve()
    ext=Path(a.extracao)
    if not ext.is_absolute():ext=asset(root, ext)
    dest=Path(a.saida)
    if not dest.is_absolute():dest=asset(root, dest)
    dest.mkdir(parents=True,exist_ok=True)
    checkpoints=dest/'checkpoints';checkpoints.mkdir(exist_ok=True)
    meta,mat,scat,names=load_matrices(root,ext)
    if a.ablacao:
        s1=np.char.startswith(names,'S1_');s2=np.char.startswith(names,'S2_')
        mat['SCAT_S1_ONLY_EXPLORATORIO']=(scat[:,s1],'linear')
        mat['SCAT_S2_ONLY_EXPLORATORIO']=(scat[:,s2],'linear')
    persons=meta[['participant_id','y']].drop_duplicates().sort_values('participant_id').reset_index(drop=True)
    if len(persons)!=411:raise ValueError('Participantes diferentes de 411')
    params={'modelos':list(mat),'repeticoes':a.repeticoes,'primeira_seed':a.primeira_seed,
            'folds':5,'k':128,'SVC_C':1,'class_weight':'balanced','threshold':0,
            'CV':'StratifiedKFold sobre participantes','protocolo':'predefinido'}
    sig=signature(root,[ext/'features_scattering_avfad.npz',ext/'metadata_scattering_avfad.csv',
                        asset(root, 'features_wavelet_temporal_avfad_reduzido.npz'),
                        asset(root, 'features_capev_compare_avfad.npz')],params)
    cfg_file=dest/'configuracao_avaliacao.json'
    if cfg_file.exists():
        if json.loads(cfg_file.read_text('utf-8')).get('assinatura')!=sig:
            raise RuntimeError('Pasta de resultados ja possui outro protocolo. Use --saida NOVO_DIRETORIO')
    else:
        cfg_file.write_text(json.dumps({'assinatura':sig,'versao':VERSION,'parametros':params},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Pessoas=411 (82 normais,329 patologicos) | Scattering',scat.shape,'| modelos',list(mat))
    print('Dados pareados por pessoa, selecao de 128 atributos dentro do treino, class_weight=balanced, corte=0.')
    seed_results=[]; all_predictions=[];fold_rows=[]
    for seed in range(a.primeira_seed,a.primeira_seed+a.repeticoes):
        split=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for mod,(X,kernel) in mat.items():
            frames=[]
            for f,(p_train,p_test) in enumerate(split.split(persons.participant_id,persons.y),1):
                output=checkpoints/f'seed_{seed}_{mod}_fold_{f}.csv'
                if output.exists():
                    fold_pred=pd.read_csv(output,dtype={'participant_id':str})
                else:
                    set_train=set(persons.iloc[p_train].participant_id)
                    set_test=set(persons.iloc[p_test].participant_id)
                    assert not (set_train&set_test)
                    mask_tr=meta.participant_id.isin(set_train).to_numpy()
                    mask_te=meta.participant_id.isin(set_test).to_numpy()
                    ytr=meta.loc[mask_tr,'y'].to_numpy(dtype=int)
                    xt,xv=scaled_features(X[mask_tr],X[mask_te],ytr,128)
                    clf=SVC(kernel=kernel,C=1.0,gamma='scale',class_weight='balanced',cache_size=512)
                    clf.fit(xt,ytr)
                    frame=meta.loc[mask_te,['participant_id','y']].copy().reset_index(drop=True)
                    frame['score']=clf.decision_function(xv)
                    fold_pred=frame.groupby('participant_id',as_index=False).agg(y=('y','first'),score=('score','mean'))
                    if len(fold_pred)!=len(set_test):raise ValueError('Numero de pessoas incongruente no teste')
                    fold_pred['fold']=f
                    atomic_csv(fold_pred,output)
                if not {'participant_id','y','score','fold'}.issubset(fold_pred.columns):raise ValueError(f'Checkpoint antigo/invalido {output}')
                if set(fold_pred.participant_id)!=set(persons.iloc[p_test].participant_id):
                    raise ValueError(f'Checkpoint nao corresponde a particao seed {seed} fold {f}')
                score=summarize(fold_pred.y,fold_pred.score)
                fold_rows.append({'seed':seed,'modelo':mod,'fold':f,**score})
                frames.append(fold_pred)
                print(f'seed {seed}, {mod}, fold {f}/5, BA={score["ba"]:.3f}',flush=True)
            joined=pd.concat(frames,ignore_index=True)
            if len(joined)!=len(persons) or joined.participant_id.duplicated().any():
                raise ValueError('OOF incompleto ou repetido')
            total=summarize(joined.y,joined.score)
            seed_results.append({'seed':seed,'modelo':mod,**total})
            joined['seed']=seed; joined['modelo']=mod
            all_predictions.append(joined)
            print(f'  => seed {seed} {mod}: BA={total["ba"]:.4f} AUC={total["auc"]:.4f} Sens={total["sens"]:.3f} Esp={total["esp"]:.3f}',flush=True)
    result=pd.DataFrame(seed_results);pred=pd.concat(all_predictions,ignore_index=True)
    atomic_csv(result,dest/'resultados_por_seed.csv')
    atomic_csv(pd.DataFrame(fold_rows),dest/'resultados_por_fold.csv')
    atomic_csv(pred,dest/'predicoes_oof_por_seed.csv')
    lines=['# Experimento Scattering AVFAD — avaliacao por participante','',
           'AVISO: seeds repetem os MESMOS 411 individuos. Nao ha teste externo independente.',
           'Modelo primario definido ANTES: Kymatio S1+S2, SVC linear, k=128, C=1, limiar=0.',
           'Baselines fixos: WPT sym5 linear e MFCC RBF; mesma divisao por participante.',
           'Nao selecionar a melhor seed nem o melhor modelo exploratorio como resultado final.','',
           '| Representacao | BA media | min..max BA | AUC media | sens media | esp media |',
           '|---|---:|---:|---:|---:|---:|']
    for mod,g in result.groupby('modelo',sort=False):
        lines.append(f'| {mod} | {g.ba.mean():.4f} | {g.ba.min():.4f}..{g.ba.max():.4f} | {g.auc.mean():.4f} | {g.sens.mean():.4f} | {g.esp.mean():.4f} |')
    wide=result.pivot(index='seed',columns='modelo',values='ba')
    if 'SCAT_S1_S2_SVC_linear' in wide and 'WPT_sym5_SVC_linear' in wide:
        delta=wide.SCAT_S1_S2_SVC_linear-wide.WPT_sym5_SVC_linear
        lines.extend(['',f'Diferenca pareada SCAT-WPT por seed: media={delta.mean():+.4f}, seeds positivas={int((delta>0).sum())}/{len(delta)}.',
                      'Essa comparacao utiliza as mesmas pessoas reiteradamente; nao equivale a teste de generalizacao clinica.'])
    report='\n'.join(lines)+'\n'
    (dest/'RELATORIO_SCATTERING.md').write_text(report,encoding='utf-8')
    print('\n'+report+'\nRelatorio:',dest/'RELATORIO_SCATTERING.md')

if __name__=='__main__':main()
