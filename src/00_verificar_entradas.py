# -*- coding: utf-8 -*-
"""Valida dimensoes, rotulos e identidade entre os NPZs da AVFAD."""
import argparse
from pathlib import Path
import pandas as pd
from avfad_core import load_data,representations,AVFADError


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados',default='.')
    o=p.parse_args()
    root=Path(o.dados)
    meta,matrices,paths=load_data(root,require_baselines=True)
    print('=== ENTRADAS CONFIRMADAS ===')
    print('Gravacoes:',len(meta),'| participantes:',meta.participant_id.nunique())
    print('Participantes por classe:',meta[['participant_id','y']].drop_duplicates().y.value_counts().to_dict())
    print('Tarefas:',meta.task.value_counts().sort_index().to_dict())
    print('Frases por participante:',meta.groupby('participant_id').size().value_counts().sort_index().to_dict())
    for name,(x,names) in matrices.items():
        print(name,'matriz:',x.shape,'| NaN:',int(pd.isna(x).sum()),'| nomes:',len(names))
    print('Representacoes completas:',{k:len(v) for k,v in representations(matrices['WAVELET'][1],'completo').items()})
    print('\nTudo alinhado por sample_id. Nenhum WAV foi lido e nenhum treino foi realizado.')


if __name__=='__main__':
    try:main()
    except (AVFADError,ValueError,FileNotFoundError) as e:raise SystemExit(f'ERRO: {e}')
