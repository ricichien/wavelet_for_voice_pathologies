# -*- coding: utf-8 -*-
"""Preflight do Kymatio e dos WAVs originais AVFAD dentro dos ZIPs."""
import argparse, platform
from pathlib import Path
import numpy as np
from scattering_avfad_common import load_manifest,zip_index,make_scattering

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dados',default='.')
    a=p.parse_args()
    root=Path(a.dados).resolve()
    print('Python:',platform.python_version())
    print('NumPy:',np.__version__)
    try:
        import kymatio
        print('Kymatio:',getattr(kymatio,'__version__','instalado'))
        scattering,ix,names,meta=make_scattering()
        print(f'Scattering J=9 Q=(8,1): {len(ix)} caminhos de ordens 1/2, {len(names)} atributos')
        print('Ordens:',meta.order.value_counts().to_dict())
    except Exception as exc:
        raise RuntimeError('Nao consegui inicializar Kymatio: use python -m pip install kymatio==0.3.0 e revise README') from exc
    manifest=load_manifest(root)
    idx,zips=zip_index(root,set(manifest.sample_id))
    print(f'AVFAD HSA: {len(manifest)} gravacoes, {manifest.participant_id.nunique()} pessoas; arquivos WAV localizados: {len(idx)}')
    for f in zips:print('  ZIP:',f.name)
    print('PRE-FLIGHT OK. Nenhuma caracteristica foi gravada.')

if __name__=='__main__': main()
