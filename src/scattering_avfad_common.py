# -*- coding: utf-8 -*-
"""Funcoes independentes para o experimento scattering AVFAD HSA.
Nao altera arquivos ja existentes. Python >=3.10.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir
import hashlib, json, re
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

TASKS = {'004','005','006','007','008','009'}
VERSION = 'scattering-avfad-1.0'


def load_manifest(root: Path) -> pd.DataFrame:
    p = asset(root, 'metadata_wavelet_temporal_avfad_reduzido.csv')
    if not p.exists(): raise FileNotFoundError(f'Falta {p}')
    df = pd.read_csv(p, dtype={'sample_id':str,'participant_id':str,'task':str})
    for c in ('sample_id','participant_id','task','y'):
        if c not in df: raise ValueError(f'Metadata sem coluna {c}')
    df = df.copy()
    df['sample_id'] = df.sample_id.astype(str).str.strip().str.upper()
    df['participant_id'] = df.participant_id.astype(str).str.strip().str.upper()
    df['task'] = df.task.astype(str).str.zfill(3)
    df['y'] = df['y'].astype(int)
    if df.sample_id.duplicated().any(): raise ValueError('sample_id duplicado')
    if set(df.task) != TASKS: raise ValueError(f'Tarefas inesperadas: {set(df.task)}')
    if (df.groupby('participant_id').y.nunique() != 1).any(): raise ValueError('Participante com rotulos diferentes')
    if len(df) != 2465 or df.participant_id.nunique() != 411 or df.drop_duplicates('participant_id').y.value_counts().to_dict() != {1:329,0:82}:
        raise ValueError('Manifesto diferente da coorte AVFAD HSA auditada 411/2465. Pare e confira.')
    return df


def zip_index(root: Path, needed_ids: set[str]) -> tuple[dict, list]:
    zips = sorted(raw_dir(root).glob('AVFAD_01_00_00_*.zip'))
    if len(zips) < 4: raise FileNotFoundError('Nao encontrei os ZIPs originais AVFAD_01_00_00_*.zip')
    idx = {}
    rx = re.compile(r'^([A-Z]{3})(00[4-9])\.wav$', re.IGNORECASE)
    for z in zips:
        with ZipFile(z, 'r') as handle:
            for info in handle.infolist():
                if info.is_dir(): continue
                basename=info.filename.replace('\\','/').rsplit('/',1)[-1].upper()
                if not rx.fullmatch(basename) or basename[:-4] not in needed_ids: continue
                key=basename[:-4]
                if key in idx:
                    raise RuntimeError(f'Mais de um WAV para {key}: {idx[key]} e {(str(z),info.filename)}')
                idx[key]=(z,info.filename)
    missing=sorted(needed_ids-set(idx))
    if missing: raise FileNotFoundError(f'Faltam {len(missing)} WAVs nos ZIPs: {missing[:12]}')
    return idx,zips


def signature(root:Path, paths:list[Path], params:dict)->str:
    sig=[]
    for p in paths:
        stat=p.stat()
        sig.append([p.name,int(stat.st_size),int(stat.st_mtime_ns)])
    blob=json.dumps({'version':VERSION,'inputs':sig,'params':params},sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def feature_layout(scattering):
    meta=scattering.meta()
    order=np.asarray(meta['order']).reshape(-1).astype(int)
    selected=np.flatnonzero(np.isin(order,[1,2]))
    if not np.any(order==1) or not np.any(order==2):
        raise RuntimeError('Kymatio nao retornou coeficientes de primeira e segunda ordem.')
    stats=('mean','std','q10','q90')
    names=[f'S{order[i]}_p{i:04d}__{stat}' for i in selected for stat in stats]
    table=[]
    for i in selected:
        row={'path_idx':int(i),'order':int(order[i])}
        for key in ('n','j','xi','sigma'):
            if key not in meta:continue
            v=np.asarray(meta[key][i]).ravel()
            row[key]=' '.join('nan' if not np.isfinite(float(x)) else f'{float(x):.8g}' for x in v)
        table.append(row)
    return selected, np.asarray(names, dtype=str), pd.DataFrame(table)


def load_audio(zip_entry):
    import io
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    zp,member=zip_entry
    with ZipFile(zp,'r') as archive: blob=archive.read(member)
    data,sr=sf.read(io.BytesIO(blob),dtype='float32',always_2d=True)
    if data.shape[0]<100: raise ValueError('Audio vazio ou excessivamente curto')
    mono=data.mean(axis=1,dtype=np.float64).astype(np.float32)
    if not np.isfinite(mono).all(): raise ValueError('Audio com valores nao finitos')
    target_sr=16000
    if sr!=target_sr:
        d=gcd(sr,target_sr)
        mono=resample_poly(mono,target_sr//d,sr//d).astype(np.float32)
    return mono,int(sr)


def active_audio(x, sr=16000, top_db=40, gap_s=.25, min_s=.25):
    import librosa
    # 2048 amostras a 48kHz ~= 42.7 ms, aqui ~= 43ms a 16kHz.
    intervals=np.asarray(librosa.effects.split(x,top_db=top_db,frame_length=688,hop_length=160),dtype=np.int64).reshape(-1,2)
    if not len(intervals): raise ValueError('VAD nao detectou fala')
    merged=[]
    for a,b in intervals:
        if merged and a-merged[-1][1]<=round(gap_s*sr): merged[-1][1]=int(b)
        else: merged.append([int(a),int(b)])
    keep=[(a,b) for a,b in merged if b-a >= round(min_s*sr)]
    if not keep: raise ValueError('Nao ha regioes de fala >= 0,25s')
    speech=np.concatenate([x[a:b] for a,b in keep])
    return speech,len(keep),float(len(speech)/len(x))


def window_positions(n:int,win:int,hop:int,max_windows:int):
    if n<=win: starts=np.array([0],dtype=np.int64)
    else:
        starts=np.arange(0,n-win+1,hop,dtype=np.int64)
        last=n-win
        if starts[-1] < last: starts=np.r_[starts,last]
    if len(starts)>max_windows:
        select=np.rint(np.linspace(0,len(starts)-1,max_windows)).astype(int)
        starts=starts[select]
    return starts


def extract_one(x,scattering,paths,max_windows=64,sr=16000):
    active,regions,ratio=active_audio(x,sr=sr)
    rms=float(np.sqrt(np.mean(np.asarray(active,dtype=np.float64)**2)))
    if rms < 1e-7: raise ValueError('Voz de amplitude insuficiente')
    active=(active*(0.1/rms)).astype(np.float32)
    win=int(.4*sr);hop=int(.2*sr)
    starts=window_positions(len(active),win,hop,max_windows)
    blocks=[]
    # Kymatio aceita batch (..., comprimento), economizando chamadas de FFT.
    for chunk_start in range(0,len(starts),8):
        chunk=[]
        for st in starts[chunk_start:chunk_start+8]:
            v=active[st:st+win]
            if len(v)<win:v=np.pad(v,(0,win-len(v)))
            chunk.append(v)
        Y=np.asarray(scattering(np.stack(chunk).astype(np.float32)))
        if Y.ndim!=3 or Y.shape[0]!=len(chunk) or Y.shape[1]<=max(paths):
            raise RuntimeError(f'Kymatio retornou dimensoes inesperadas: {Y.shape}')
        avg=Y[:,paths,:].mean(axis=-1)
        if np.any(avg < -1e-5):raise RuntimeError('Coeficientes negativos fora da tolerancia')
        blocks.extend(np.log1p(np.maximum(avg,0.0)*100.0))
    samples=np.asarray(blocks,dtype=np.float32)
    vec=np.stack([samples.mean(axis=0),samples.std(axis=0),
                  np.quantile(samples,.1,axis=0),np.quantile(samples,.9,axis=0)],axis=1).ravel().astype(np.float32)
    if not np.isfinite(vec).all():raise ValueError('Scattering gerou valor NaN/Inf')
    return vec, {'n_windows':len(starts),'n_regions':regions,'ratio_active':ratio,
                 'duration_active_s':len(active)/sr, 'n_windows_possible':len(window_positions(len(active),win,hop,10**9))}


def make_scattering(J=9,Q1=8,Q2=1, sr=16000):
    from kymatio.numpy import Scattering1D
    win=round(.4*sr)
    transform=Scattering1D(J=J,shape=win,Q=(Q1,Q2),max_order=2)
    ix,names,meta=feature_layout(transform)
    t=np.arange(win,dtype=np.float32)/sr
    synthetic=(.1*np.sin(2*np.pi*220*t)).astype('float32')
    test=np.asarray(transform(np.stack([synthetic,synthetic])))
    if test.ndim!=3 or test.shape[0]!=2 or test.shape[1]!=len(transform.meta()['order']):
        raise RuntimeError(f'Scattering output shape inesperado {test.shape}')
    return transform,ix,names,meta
