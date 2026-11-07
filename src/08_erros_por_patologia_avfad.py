# -*- coding: utf-8 -*-
"""Analisa sensibilidadeda WPT por codigo de patologia (apenas POS-HOC).

SEM treinar modelo. Usa predicoes OOF do experimento 06 e metadados clinicos
AVFAD. Subgrupos pequenos tem grande incerteza; nao escolher o subgrupo
mais facil para reivindicar desempenho de generalizacao.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def normalize(t):
    return re.sub(r'[^a-z0-9]+','_',str(t).strip().lower()).strip('_')


def choose_column(df, explicit, candidates, label):
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f'Coluna {label} "{explicit}" nao encontrada. Disponiveis: {list(df.columns)}')
        return explicit
    normalized={normalize(c):c for c in df.columns}
    for c in candidates:
        if c in normalized:return normalized[c]
    raise ValueError(f'Nao identifiquei automaticamente coluna {label}.\n'
                     f'Colunas: {list(df.columns)}\n'
                     f'Execute de novo com --{"id-col" if label=="ID" else "codigo-col"} "NOME_DA_COLUNA"')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadados',default='validacao_tarefa_002.csv')
    p.add_argument('--predicoes',default='resultados_robustez_folds_wavelet/predicoes_por_participante_todas_seeds.csv')
    p.add_argument('--saida',default='resultados_08_subtipos')
    p.add_argument('--id-col',help='Nome EXATO da coluna de ID em metadados')
    p.add_argument('--codigo-col',help='Nome EXATO da coluna com codigo diagnostico')
    p.add_argument('--min-pessoas',type=int,default=8)
    a=p.parse_args()
    f=asset(project_root(), a.metadados);predpath=asset(project_root(), a.predicoes)
    if not f.exists():raise SystemExit(f'Metadados ausentes: {f}\nEste script precisa da tabela de diagnosticos original.')
    if not predpath.exists():raise SystemExit(f'Predicoes ausentes: {predpath}\nExecute primeiro o experimento 06.')
    src=pd.read_csv(f,dtype=str,low_memory=False)
    try:
        idcol=choose_column(src,a.id_col,['participant_id','file_id','fileid','pid','speaker_id','speaker','id','codigo_participante','participante','arquivo'], 'ID')
        codecol=choose_column(src,a.codigo_col,[
            'cmvd_i_dimension_1_numeric_system',
            'cmvd_i_dimension_1_numeric','cmvd_i_code','cmvdi','cmvd_code',
            'clinical_code','clinical_class','pathology_code','diagnosis_code','diagnostic_code',
            'codigo_patologia','codigo_clinico','codigo_diagnostico','diagnostico_codigo',
            'diagnosis','diagnostico','codigo','code'], 'codigo')
    except ValueError as e:
        raise SystemExit(str(e))
    dat=src[[idcol,codecol]].rename(columns={idcol:'participant_id',codecol:'codigo'}).copy()
    dat['participant_id']=dat.participant_id.astype(str).str.strip().str.upper()
    dat['codigo']=dat.codigo.astype(str).str.strip()
    dat=dat.dropna(subset=['participant_id'])
    conflicts=dat.groupby('participant_id').codigo.nunique()
    if (conflicts>1).any():raise RuntimeError('Metadados tem IDs com codigos clinicos contraditorios')
    dat=dat.drop_duplicates('participant_id')
    preds=pd.read_csv(predpath,dtype={'participant_id':str},low_memory=False)
    required={'participant_id','seed','y','score','modelo','esquema'}
    if not required.issubset(preds.columns):
        raise SystemExit(f'Predicoes incompletas: faltam {required-set(preds.columns)}')
    preds['participant_id']=preds.participant_id.str.strip().str.upper()
    preds=preds[(preds.modelo=='WPT_sym5_SVC_linear') & (preds.esquema=='SKF')].copy()
    if preds.empty:raise RuntimeError('Nao achei WPT_sym5_SVC_linear e esquema SKF no experimento 06')
    preds['y']=pd.to_numeric(preds.y,errors='raise').astype(int)
    preds['score']=pd.to_numeric(preds.score,errors='raise')
    preds['acertou']=((preds.score>=0).astype(int)==preds.y.astype(int))
    merged=preds.merge(dat,on='participant_id',how='left',validate='many_to_one')
    missing=merged.loc[merged.codigo.isna(),'participant_id'].unique().tolist()
    if missing:
        raise RuntimeError(f'Faltam codigos de diagnostico para {len(missing)} pessoas. Exemplos: {missing[:10]}')
    pathology=merged[merged.y==1].copy()
    if pathology.empty:raise RuntimeError('Nenhuma pessoa patologica nas predicoes')
    per_seed=(pathology.groupby(['codigo','seed'],as_index=False)
              .agg(n=('participant_id','nunique'),sensibilidade=('acertou','mean')))
    counts=pathology.groupby('codigo').participant_id.nunique().rename('pessoas')
    grouped=(per_seed.groupby('codigo').agg(sens_media=('sensibilidade','mean'),
                    sens_min=('sensibilidade','min'),sens_max=('sensibilidade','max'),
                    sementes=('seed','nunique')).join(counts).reset_index())
    grouped=grouped.sort_values(['pessoas','codigo'],ascending=[False,True])
    dest=asset(project_root(), a.saida);dest.mkdir(parents=True,exist_ok=True)
    per_seed.to_csv(dest/'sensibilidade_diagnostico_por_seed.csv',index=False,encoding='utf-8-sig')
    grouped.to_csv(dest/'resumo_por_codigo_diagnostico.csv',index=False,encoding='utf-8-sig')
    pathology[['participant_id','seed','codigo','y','score','acertou']].to_csv(
        dest/'predicoes_patologicos_com_codigo.csv',index=False,encoding='utf-8-sig')
    print('Colunas identificadas:',idcol,'/',codecol)
    print('Pessoas patologicas:',pathology.participant_id.nunique(),
          '| sementes:',preds.seed.nunique(), '| codigos distintos:',pathology.codigo.nunique())
    print('Especificidade nao pode ser calculada por subtipo apenas com casos patologicos.')
    print(f'\nCodigos com ao menos {a.min_pessoas} participantes:')
    show=grouped[grouped.pessoas>=a.min_pessoas].copy()
    if show.empty:print('Nenhum codigo tem pessoas suficientes para mostrar. Consulte CSV.')
    for z in show.itertuples(index=False):
        print(f'  codigo={z.codigo:>8s} | n={z.pessoas:>3} | sensibilidade media={z.sens_media:.3f} (seeds {z.sens_min:.3f}..{z.sens_max:.3f})')
    lines=['# AVFAD — Erros da WPT por grupo diagnostico (exploratorio)', '',
           f'- CSV clinico: {f}; coluna de pessoa: {idcol}; codigo diagnostico: {codecol}.',
           f'- Predicoes: {predpath}; modelo: WPT_sym5_SVC_linear; esquema SKF.',
           f'- Pessoas patologicas: {pathology.participant_id.nunique()}; codigos: {pathology.codigo.nunique()}.',
           '- Sensibilidade por codigo = taxa de patologia detectada naquele codigo; sem normais por codigo, especificidade por codigo nao se aplica.',
           '- Cada seed reutiliza as mesmas pessoas: intervalos entre seeds NAO sao IC95%.',
           '- Os subgrupos nao sao alvos de classificacao treinados separadamente.',
           '- NAO escolher o codigo mais facil para alegar BA de 80%; isto e apenas diagnostico de erros.',
           '- Para um detector especifico de um diagnostico, a pergunta clinica e a coorte precisam ser definidas antes de uma nova avaliacao.',
           '', '| Codigo | Pessoas | Sens media | Min | Max |', '|---|---:|---:|---:|---:|']
    for z in show.itertuples(index=False):
        lines.append(f'| {z.codigo} | {z.pessoas} | {z.sens_media:.3f} | {z.sens_min:.3f} | {z.sens_max:.3f} |')
    (dest/'RELATORIO_08.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\nRelatorio:',dest/'RELATORIO_08.md')

if __name__=='__main__':main()
