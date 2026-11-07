# -*- coding: utf-8 -*-
"""Guarda copias pequenas do codigo, verificadas pela assinatura da execucao."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experimento", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    experiment = args.experimento.resolve()
    if not experiment.is_relative_to(root / "results/experiments"):
        parser.error("Usar pasta dentro de results/experiments")
    signature = json.loads((experiment / "assinatura_execucao.json").read_text(encoding="utf-8"))
    records = {}
    for field in ("code", "files"):
        for source, value in signature.get(field, {}).items():
            if Path(source).suffix == ".py":
                records[source] = value["sha256"] if isinstance(value, dict) else value
    if not records:
        raise ValueError("Assinatura nao apresenta hashes de fontes compativeis")
    destination = experiment / "codigo_congelado"
    destination.mkdir(exist_ok=True)
    lines = ["CODIGO CONGELADO DA EXECUCAO", "",
             "Copias geradas depois do inicio da execucao e verificadas contra",
             "os hashes de codigo registrados antes dos respectivos ajustes.",
             "Nao inclui dados, matrizes, audios ou dependencias instaladas.", ""]
    for source_name, expected in records.items():
        source = Path(source_name).resolve()
        if not source.is_relative_to(root / "src") or digest(source) != expected:
            raise ValueError(f"Fonte fora do escopo ou alterado: {source}")
        target = destination / source.name
        if target.exists():
            if digest(target) != expected:
                raise ValueError(f"Copia existente difere: {target}")
        else:
            target.write_bytes(source.read_bytes())
        if digest(target) != expected:
            raise ValueError(f"Falha ao verificar copia: {target}")
        lines.append(f"{source} -> {target.name} | SHA256={expected}")
    (destination / "LEIA.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Fontes verificadas e preservadas: {len(records)} em {destination}")


if __name__ == "__main__":
    main()
