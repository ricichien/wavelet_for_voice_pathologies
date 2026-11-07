# -*- coding: utf-8 -*-
"""Pareamento deterministico para analises SECUNDARIAS de controle demografico.

Cria 2 CSVs com pair_id, preservando ambos os membros de cada par nos mesmos
folds em 01_otimizar_wavelet_nested.py --coorte <arquivo>. Pareamento por
sexo + faixa de 5 anos (+ mesmo ano, quando solicitado).
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from avfad_core import load_metadata,AVFADError


def choose_col(m,aliases):
    names={str(k).casefold().strip().replace(' ','_'):k for k in m.columns}
    for alias in aliases:
        a=alias.casefold().replace(' ','_')
        if a in names:return names[a]
    return None


def meta_demografica(path):
    m=pd.read_csv(path)
    keys={
        'participant_id':['participant_id','file_id','file id','pid','id'],
        'age':['age','idade'],
        'sex':['sex','sexo','gender'],
        'year':['visit_year','year','ano','ano_visita'],
        'visit_date':['visit_date','visit date','data_visita','date'],
        'visit_place':['visit_place','visit place','centro','site'],
        'y':['y','label','class','classe','diagnostic_binary'],
    }
    cols={k:choose_col(m,a) for k,a in keys.items()}
    for col in ('participant_id','age','sex'):
        if cols[col] is None:raise AVFADError(f'{path.name}: nao encontrei {col}; colunas: {list(m.columns)}')
    o=pd.DataFrame()
    o['participant_id']=m[cols['participant_id']].astype(str).str.strip().str.upper()
    o['age']=pd.to_numeric(m[cols['age']],errors='coerce')
    o['sex']=m[cols['sex']].astype(str).str.strip().str.upper()
    if cols['year']:
        o['year']=pd.to_numeric(m[cols['year']],errors='coerce')
    elif cols['visit_date']:
        o['year']=pd.to_datetime(m[cols['visit_date']],errors='coerce',dayfirst=True).dt.year
    else:
        o['year']=np.nan
    if cols['visit_place']:
        center=m[cols['visit_place']].astype(str).str.strip().str.upper()
        o=o[center.eq('HSA')].copy()
    o=o.drop_duplicates('participant_id')
    return o


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--dados',default='.')
    a.add_argument('--metadata-demografia',default=None,help='CSV com participant_id/age/sex/year; pode ser validacao_tarefa_002.csv')
    a.add_argument('--saida',default='coortes_wavelet_pareadas')
    a.add_argument('--seed',type=int,default=42)
    o=a.parse_args()
    root=Path(o.dados).resolve()
    wave=load_metadata(asset(root, 'metadata_wavelet_temporal_avfad_reduzido.csv'))
    people=wave[['participant_id','y']].drop_duplicates()
    if o.metadata_demografia:
        demo=meta_demografica(Path(o.metadata_demografia))
    else:
        demo=meta_demografica(asset(root, 'metadata_wavelet_temporal_avfad_reduzido.csv'))
        if demo.year.isna().all():
            extra=asset(root, 'validacao_tarefa_002.csv')
            if extra.exists():
                fallback=meta_demografica(extra)
                demo=demo.drop(columns=['year']).merge(fallback[['participant_id','year']],on='participant_id',how='left')
    m=people.merge(demo,on='participant_id',validate='one_to_one',how='left')
    if m.age.isna().any() or m.sex.isna().any():
        raise AVFADError('Idade ou sexo ausentes. Forneca --metadata-demografia com colunas adequadas.')
    m['agebin']=(m.age//5).astype(int)
    out=Path(o.saida)
    if not out.is_absolute():out=asset(root, out)
    out.mkdir(parents=True,exist_ok=True)
    for mode in ['sexo_idade','sexo_idade_ano']:
        if mode.endswith('ano') and m.year.isna().any():
            print('Nao foi possivel criar coorte por ano: coluna year ausente para alguns IDs.')
            continue
        cols=['sex','agebin']+(['year'] if mode.endswith('ano') else [])
        rows=[];rng=np.random.default_rng(o.seed)
        for _,chunk in m.groupby(cols,sort=True):
            n=chunk[chunk.y==0].sort_values('participant_id')
            p=chunk[chunk.y==1].sort_values('participant_id')
            if not len(n) or not len(p):continue
            n=n.iloc[rng.permutation(len(n))]
            p=p.iloc[rng.permutation(len(p))]
            for i in range(min(len(n),len(p))):
                pair=f'P{len(rows)//2+1:04d}'
                for item in [n.iloc[i],p.iloc[i]]:
                    rows.append({'participant_id':item.participant_id,'pair_id':pair,'y':int(item.y),
                                 'age':int(item.age),'sex':item.sex,'year':item.year})
        if not rows:raise AVFADError('Nao foi possivel formar nenhum par.')
        out_file=out/f'coorte_{mode}.csv'
        pd.DataFrame(rows).to_csv(out_file,index=False,encoding='utf-8-sig')
        print(f'{mode}: {len(rows)//2} pares / {len(rows)} participantes -> {out_file}')
    print('Pares formados APENAS por demografia. Nao usar desempenho externo para escolher coorte vencedora.')


if __name__=='__main__':
    try:main()
    except (AVFADError,FileNotFoundError,ValueError) as e:raise SystemExit(f'ERRO: {e}')
