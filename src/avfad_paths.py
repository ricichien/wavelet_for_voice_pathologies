# -*- coding: utf-8 -*-
"""Caminhos unificados para AVFAD.

Toda a pesquisa reside na mesma pasta raiz do repositorio, sem replicar o corpus.
Os scripts ficam em src/; os arquivos grandes permanecem na unidade D:.
"""
from __future__ import annotations
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def raw_dir(root: Path | None = None) -> Path:
    # Os cinco ZIPs originais ficam permanentemente na raiz do projeto.
    return Path(root or project_root())


def asset(root: Path, name: str | Path) -> Path:
    """Roteia referências históricas para os novos locais, sem cópias nem atalhos.

    Caminhos absolutos e caminhos que já apontam para uma subpasta nova
    não são modificados. Diretórios de experimento ficam em results/experiments.
    """
    root = Path(root)
    item = Path(name)
    if item.is_absolute():
        return item
    parts = item.parts
    if not parts:
        return root
    first = parts[0]
    if first in {'data','results','checkpoints','docs','src','backups','arquivos_auxiliares'}:
        return root / item
    if first == 'metadata':
        return root / 'data' / 'metadata' / Path(*parts[1:])
    if first == 'saida':
        return root / 'results' / 'legacy' / item
    if first.startswith('resultados_'):
        return root / 'results' / 'experiments' / item
    if first.startswith('checkpoint_') or first.startswith('cache_'):
        return root / 'checkpoints' / item
    if first.startswith('AVFAD_01_00_00_') and item.suffix.lower() == '.zip':
        return root / item
    if item.suffix.lower() == '.npz':
        return root / 'data' / 'processed' / item
    if item.suffix.lower() == '.sqlite':
        return root / 'checkpoints' / item
    if item.suffix.lower() == '.csv' and first.startswith(('metadata_','validacao_', 'features_')):
        return root / 'data' / 'processed' / item
    if item.suffix.lower() in {'.csv','.txt','.md','.png'}:
        return root / 'results' / 'legacy' / item
    return root / item
