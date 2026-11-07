# -*- coding: utf-8 -*-
"""Funcoes comuns: somente matrizes ja extraidas, SEM leitura de WAV."""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold

class AVFADError(RuntimeError):
    pass


def load_metadata(path):
    m = pd.read_csv(path)
    required = {'sample_id', 'participant_id', 'task', 'y'}
    if not required.issubset(m.columns):
        raise AVFADError(f'{path.name}: colunas ausentes {required-set(m.columns)}')
    m = m.copy()
    m['sample_id'] = m.sample_id.astype(str).str.strip().str.upper()
    m['participant_id'] = m.participant_id.astype(str).str.strip().str.upper()
    m['task'] = m.task.astype(str).str.strip().str.zfill(3)
    m['y'] = pd.to_numeric(m.y, errors='raise').astype(int)
    if set(m.y.unique()) != {0, 1}:
        raise AVFADError('Esperava rotulos y=0 (normal) e y=1 (patologico).')
    if m.sample_id.duplicated().any():
        raise AVFADError('Ha sample_id duplicado.')
    if (m.groupby('participant_id').y.nunique() != 1).any():
        raise AVFADError('Rotulos divergentes entre frases da mesma pessoa.')
    return m


def input_signature(paths):
    raw = []
    for p in paths:
        s = p.stat()
        raw.append([str(p.resolve()), s.st_size, s.st_mtime_ns])
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()


def load_data(root, require_baselines=True):
    root = Path(root)
    wp = asset(root, 'features_wavelet_temporal_avfad_reduzido.npz')
    wm = asset(root, 'metadata_wavelet_temporal_avfad_reduzido.csv')
    if not wp.exists() or not wm.exists():
        raise AVFADError('Nao encontrei features_wavelet_temporal_avfad_reduzido.npz e metadata_wavelet_temporal_avfad_reduzido.csv.')
    meta = load_metadata(wm)
    with np.load(wp, allow_pickle=False) as data:
        if not {'wavelet','feature_names'}.issubset(data.files):
            raise AVFADError('NPZ wavelet sem chaves wavelet e feature_names.')
        wave = np.asarray(data['wavelet'], dtype=np.float32)
        wn = np.asarray(data['feature_names']).astype(str)
    if wave.ndim != 2 or wave.shape != (len(meta),len(wn)):
        raise AVFADError(f'Wavelet matriz {wave.shape}, metadata={len(meta)}, nomes={len(wn)}.')
    if wave.shape[1] != 4160:
        raise AVFADError(f'Esperava 4160 atributos wavelet, achei {wave.shape[1]}.')
    paths = [wp,wm]
    matrices = {'WAVELET': (wave, wn)}
    bp = asset(root, 'features_capev_compare_avfad.npz')
    bm = next((p for p in [asset(root, 'metadata_capev_avfad.csv'),asset(root, 'metadata_capev_compare_avfad.csv')] if p.exists()),None)
    if require_baselines:
        if not bp.exists() or bm is None:
            raise AVFADError('Baselines solicitados, mas falta features_capev_compare_avfad.npz ou metadata_capev_avfad.csv.')
    if bp.exists() and bm is not None:
        baseline_meta = load_metadata(bm)
        if set(meta.sample_id) != set(baseline_meta.sample_id):
            raise AVFADError('IDs de gravacao diferem entre NPZ Wavelet e NPZ baseline.')
        aligned = baseline_meta.set_index('sample_id').loc[meta.sample_id]
        if not np.array_equal(aligned.y.to_numpy(),meta.y.to_numpy()):
            raise AVFADError('Rotulos inconsistentes entre Wavelet e baseline.')
        if not np.array_equal(aligned.participant_id.to_numpy(),meta.participant_id.to_numpy()):
            raise AVFADError('Identidades inconsistentes entre Wavelet e baseline.')
        ix = pd.Series(np.arange(len(baseline_meta)),index=baseline_meta.sample_id).loc[meta.sample_id].to_numpy(dtype=int)
        with np.load(bp, allow_pickle=False) as data:
            an = np.asarray(data['acoustic_names']).astype(str)
            a = np.asarray(data['acoustic'],dtype=np.float32)
            cm = np.asarray(data['compare'],dtype=np.float32) if 'compare' in data.files else None
            cn = np.asarray(data['compare_names']).astype(str) if 'compare_names' in data.files else None
        if a.shape != (len(baseline_meta),len(an)):
            raise AVFADError('Dimensoes acoustic incongruentes.')
        mi = np.array([i for i,n in enumerate(an) if n.startswith(('mfcc_','delta_','delta2_'))],dtype=int)
        if len(mi) != 150:
            raise AVFADError(f'Esperava 150 atributos MFCC, achei {len(mi)}.')
        matrices['MFCC']=(a[ix][:,mi],an[mi])
        if cm is not None and cn is not None:
            if cm.shape != (len(baseline_meta),len(cn)):
                raise AVFADError('Dimensoes ComParE incongruentes.')
            matrices['COMPARE']=(cm[ix],cn)
        paths += [bp,bm]
    for name,(x,names) in matrices.items():
        if not np.isfinite(x).all():
            x=np.array(x,copy=True);x[~np.isfinite(x)]=np.nan
            matrices[name]=(x,names)
    persons=meta[['participant_id','y']].drop_duplicates()
    if len(meta)!=2465 or len(persons)!=411 or persons.y.value_counts().to_dict()!={1:329,0:82}:
        raise AVFADError('Entrada nao e a coorte HSA auditada (2465 gravacoes, 411 participantes, 82/329).')
    return meta,matrices,paths



def aggregate_participants(meta,matrices,mode):
    """Agregacao NAO supervisionada das frases da propria pessoa (sem cruzar individuos).

    O modo media+desvio usa somente as 5 ou 6 frases da pessoa. E uma
    representacao exploratoria alternativa ao treinamento por gravacao.
    """
    if mode=='frases':
        return meta,matrices
    ids=np.array(sorted(meta.participant_id.unique()))
    positions={pid:np.flatnonzero(meta.participant_id.to_numpy()==pid) for pid in ids}
    new_meta=meta.drop_duplicates('participant_id').set_index('participant_id').loc[ids].reset_index()
    new_meta['sample_id']=new_meta['participant_id']+'__AGG'
    new_meta['task']='TODAS'
    new_meta['n_frases_aggregadas']=[len(positions[pid]) for pid in ids]
    transformed={}
    for arm,(x,names) in matrices.items():
        means=[];stds=[]
        for pid in ids:
            group=x[positions[pid]]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',category=RuntimeWarning)
                means.append(np.nanmean(group,axis=0))
                if mode=='media_desvio_pessoa':
                    stds.append(np.nanstd(group,axis=0))
        mean_x=np.asarray(means,dtype=np.float32)
        if mode=='media_desvio_pessoa':
            xnew=np.hstack([mean_x,np.asarray(stds,dtype=np.float32)])
            new_names=np.asarray([f'{n}__media_frasal' for n in names]+[f'{n}__desvio_frasal' for n in names])
        else:
            xnew=mean_x
            new_names=np.asarray([f'{n}__media_frasal' for n in names])
        xnew[~np.isfinite(xnew)]=np.nan
        transformed[arm]=(xnew,new_names)
    return new_meta,transformed


def representations(wave_names,preset):
    n=np.asarray(wave_names).astype(str)
    patterns={
        'WPT_sym5': lambda s:s.startswith('WPT__sym5__'),
        'WPT_sym5_L5':lambda s:s.startswith('WPT__sym5__') and '__L5__' in s,
        'WPT_sym5_L6':lambda s:s.startswith('WPT__sym5__') and '__L6__' in s,
        'DWT_sym5':lambda s:s.startswith('DWT__sym5__'),
        'WPT_db4':lambda s:s.startswith('WPT__db4__'),
        'DWT_db4':lambda s:s.startswith('DWT__db4__'),
        'sym5_DWT_WPT':lambda s:s.startswith(('WPT__sym5__','DWT__sym5__')),
        'WPT_duas_familias':lambda s:s.startswith(('WPT__sym5__','WPT__db4__')),
        'WPT_sym5_energia_entropia':lambda s:s.startswith('WPT__sym5__') and any(f'__{v}__' in s for v in ('rel_energy','log_energy','entropy')),
        'WPT_sym5_kurtose_zcr':lambda s:s.startswith('WPT__sym5__') and any(f'__{v}__' in s for v in ('kurtosis','zcr')),
    }
    requested={
        'teste':['WPT_sym5'],
        'rapido':['WPT_sym5','WPT_sym5_L5','WPT_sym5_L6','DWT_sym5'],
        'completo':list(patterns),
    }[preset]
    out={}
    for key in requested:
        idx=np.flatnonzero(np.fromiter((patterns[key](s) for s in n),dtype=bool,count=len(n)))
        if len(idx)==0:
            raise AVFADError(f'Representacao {key} sem caracteristicas: confira padrao dos nomes do NPZ.')
        out[key]=idx
    return out


def participant_table(meta, coorte_file=None):
    persons=meta[['participant_id','y']].drop_duplicates().sort_values('participant_id').reset_index(drop=True)
    if coorte_file:
        c=pd.read_csv(coorte_file,dtype={'participant_id':str,'pair_id':str})
        if 'participant_id' not in c.columns:
            raise AVFADError('Coorte precisa conter participant_id e opcionalmente pair_id.')
        c['participant_id']=c.participant_id.astype(str).str.strip().str.upper()
        if c.participant_id.duplicated().any():
            raise AVFADError('Coorte com participante duplicado.')
        if not set(c.participant_id).issubset(set(persons.participant_id)):
            raise AVFADError('Coorte contem IDs nao presentes na AVFAD HSA.')
        if 'y' in c.columns:
            labels=persons.set_index('participant_id')['y']
            if not np.array_equal(c['y'].to_numpy(dtype=int), labels.loc[c.participant_id].to_numpy(dtype=int)):
                raise AVFADError('Coorte com rotulos divergentes do manifesto original.')
            c=c.drop(columns=['y'])
        persons=persons.merge(c,on='participant_id',how='inner',validate='one_to_one')
        if 'pair_id' in persons:
            persons['pair_id']=persons.pair_id.astype(str)
            valid=persons.groupby('pair_id').agg(n=('y','size'),n_classes=('y','nunique'))
            if not ((valid.n==2)&(valid.n_classes==2)).all():
                raise AVFADError('Cada pair_id deve conter exatamente dois participantes, um por classe.')
    if persons.y.value_counts().min() < 6:
        raise AVFADError('Coorte muito pequena para 5-fold externo / 3-fold interno.')
    return persons


def make_folds(persons,n_splits,seed):
    if 'pair_id' in persons.columns:
        pairs=np.array(sorted(persons.pair_id.unique()))
        if len(pairs)<n_splits:
            raise AVFADError('Numero de pares menor que numero de folds.')
        splitter=KFold(n_splits=n_splits,shuffle=True,random_state=seed)
        for tr,te in splitter.split(pairs):
            tr_p=persons[persons.pair_id.isin(pairs[tr])].participant_id.to_numpy()
            te_p=persons[persons.pair_id.isin(pairs[te])].participant_id.to_numpy()
            yield tr_p,te_p
    else:
        splitter=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
        for tr,te in splitter.split(persons.participant_id,persons.y):
            yield persons.participant_id.iloc[tr].to_numpy(),persons.participant_id.iloc[te].to_numpy()


def transform(train,test,y,k):
    a=np.asarray(train,dtype=np.float32)
    b=np.asarray(test,dtype=np.float32)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True)
    a=imputer.fit_transform(a);b=imputer.transform(b)
    var=VarianceThreshold(0.0)
    a=var.fit_transform(a);b=var.transform(b)
    kval=min(k,a.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sel=SelectKBest(f_classif,k=kval)
        a=sel.fit_transform(a,y);b=sel.transform(b)
    scale=StandardScaler()
    return (scale.fit_transform(a).astype(np.float32),
            scale.transform(b).astype(np.float32),int(kval))


def row_weights(train_meta):
    """Peso balanceado por classe e por participante, nao por numero de frases."""
    counts=train_meta.groupby('participant_id').size()
    persons=train_meta[['participant_id','y']].drop_duplicates()
    classes=persons.y.value_counts().to_dict()
    n=len(persons)
    w=np.array([n/(2*classes[y]*counts[pid]) for pid,y in zip(train_meta.participant_id,train_meta.y)],dtype=float)
    return w*(len(w)/w.sum())


def model_spec(preset):
    if preset=='teste':
        return [('linear_C1','linear',1.0),]
    if preset=='rapido':
        return [('linear_C1','linear',1.0),('rbf_C1','rbf',1.0),('rbf_C10','rbf',10.0),('logistic_C1','logistic',1.0)]
    return [(f'linear_C{c:g}','linear',c) for c in (0.1,1.0,10.0)] + [(f'rbf_C{c:g}','rbf',c) for c in (0.1,1.0,10.0,100.0)] + [(f'logistic_C{c:g}','logistic',c) for c in (0.1,1.0,10.0)]


def build_model(kind,c):
    if kind in ('linear','rbf'):
        return SVC(kernel=kind,C=float(c),gamma='scale',cache_size=768)
    return LogisticRegression(C=float(c),solver='liblinear',max_iter=2500,random_state=42)


def scores(model, X):
    s=np.asarray(model.decision_function(X),dtype=float).reshape(-1)
    if not np.isfinite(s).all():
        raise AVFADError('Modelo produziu scores nao finitos.')
    return s


def collapse(meta,s):
    frame=meta[['participant_id','y']].copy()
    frame['score']=np.asarray(s,dtype=float)
    out=frame.groupby('participant_id',sort=True).agg(y=('y','first'),score=('score','mean')).reset_index()
    return out


def threshold_from_inner(y,s,mode):
    if mode=='zero':
        return 0.0
    q=np.unique(np.r_[0.0,np.quantile(s,np.linspace(0.05,0.95,37))])
    candidates=[(balanced_accuracy_score(y,(s>=v).astype(int)),float(v)) for v in q]
    return sorted(candidates,key=lambda p:(-p[0],abs(p[1]),p[1]))[0][1]


def metrics(y,s,t=0.0):
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    pred=(s>=t).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'ba':float(balanced_accuracy_score(y,pred)),
            'auc':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else float('nan'),
            'accuracy':float(accuracy_score(y,pred)),
            'f1':float(f1_score(y,pred,zero_division=0)),
            'sensitivity':float(tp/(tp+fn)) if tp+fn else float('nan'),
            'specificity':float(tn/(tn+fp)) if tn+fp else float('nan'),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def atomic_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
