# -*- coding: utf-8 -*-
"""Inspeciona alguns PRT da AVFAD sem abrir a interface do Praat.

Lê os arquivos .prt diretamente dos ZIPs, usando Parselmouth (motor do Praat
acessado pelo Python). Não modifica nem extrai os ZIPs originais.

Saída: diagnostico_prt_avfad.txt
"""

from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import tempfile
import zipfile
from pathlib import Path

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError as exc:
    raise SystemExit(
        "Biblioteca ausente. Com o .venv ativado, execute:\n"
        "python -m pip install praat-parselmouth"
    ) from exc

ROOT = project_root()
OUTPUT = asset(ROOT, "diagnostico_prt_avfad.txt")
MAX_FILES = 5  # poucos exemplos, apenas para descobrir a estrutura
MAX_INTERVALS_PER_TIER = 30


def get_objects(prt_bytes: bytes, file_name: str):
    """Praat lê a Collection binária; Parselmouth retorna seus objetos."""
    with tempfile.TemporaryDirectory(prefix="avfad_prt_") as tmp:
        path = Path(tmp) / file_name
        path.write_bytes(prt_bytes)
        objects = call("Read from file", str(path))
        if not isinstance(objects, list):
            objects = [objects]
        # Os objetos já foram carregados em memória pelo Parselmouth.
        return objects


def describe_object(report, obj, object_index: int):
    class_name = getattr(obj, "class_name", type(obj).__name__)
    object_name = getattr(obj, "name", "")
    report.write(f"  Objeto {object_index}: classe={class_name}; nome={object_name}\n")

    if not isinstance(obj, parselmouth.TextGrid):
        if isinstance(obj, parselmouth.Sound):
            report.write(
                f"    Sound: duração={obj.get_total_duration():.4f}s; "
                f"amostragem={obj.sampling_frequency:.0f} Hz; "
                f"canais={obj.n_channels}\n"
            )
        return

    n_tiers = int(call(obj, "Get number of tiers"))
    report.write(f"    TextGrid: {n_tiers} tiers; intervalo global="
                 f"{obj.xmin:.4f}–{obj.xmax:.4f}s\n")

    for tier_no in range(1, n_tiers + 1):
        tier_name = call(obj, "Get tier name", tier_no)
        report.write(f"    Tier {tier_no}: {tier_name!r}\n")

        try:
            n_intervals = int(call(obj, "Get number of intervals", tier_no))
        except Exception:
            n_intervals = None

        if n_intervals is not None:
            report.write(f"      Intervalos: {n_intervals}\n")
            for interval_no in range(1, min(n_intervals, MAX_INTERVALS_PER_TIER) + 1):
                start = float(call(obj, "Get start time of interval", tier_no, interval_no))
                end = float(call(obj, "Get end time of interval", tier_no, interval_no))
                label = call(obj, "Get label of interval", tier_no, interval_no)
                report.write(
                    f"        {interval_no}: {start:.4f}–{end:.4f}s "
                    f"({end-start:.4f}s), rótulo={label!r}\n"
                )
            if n_intervals > MAX_INTERVALS_PER_TIER:
                report.write("        [demais intervalos omitidos]\n")
            continue

        try:
            n_points = int(call(obj, "Get number of points", tier_no))
        except Exception:
            report.write("      Tipo de tier não identificado como intervalo/ponto.\n")
            continue

        report.write(f"      Pontos: {n_points}\n")
        for point_no in range(1, min(n_points, MAX_INTERVALS_PER_TIER) + 1):
            time = float(call(obj, "Get time of point", tier_no, point_no))
            label = call(obj, "Get label of point", tier_no, point_no)
            report.write(f"        {point_no}: {time:.4f}s, rótulo={label!r}\n")


def main():
    candidates = []

    for zip_path in sorted(raw_dir(ROOT).glob("AVFAD_01_00_00_*.zip")):
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                base = info.filename.replace("\\", "/").split("/")[-1]
                upper = base.upper()
                if len(upper) == 10 and upper.endswith("002.PRT"):
                    candidates.append((zip_path, info.filename, base))

    # Um exemplo por arquivo ZIP, quando disponível.
    chosen = []
    seen_archives = set()
    for item in candidates:
        zip_path = item[0]
        if zip_path.name not in seen_archives:
            chosen.append(item)
            seen_archives.add(zip_path.name)
        if len(chosen) >= MAX_FILES:
            break

    if not chosen:
        raise SystemExit(f"Nenhum arquivo *002.prt encontrado nos ZIPs de {ROOT}")

    with OUTPUT.open("w", encoding="utf-8") as report:
        report.write("DIAGNÓSTICO DOS ARQUIVOS PRT — AVFAD\n")
        report.write("=" * 60 + "\n")
        report.write(f"Exemplos analisados: {len(chosen)}\n")
        report.write("Os ZIPs originais não são modificados.\n\n")

        for zip_path, internal_name, base in chosen:
            report.write(f"ARQUIVO: {base}\nZIP: {zip_path.name}\n")
            try:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    data = archive.read(internal_name)
                objects = get_objects(data, base)
                report.write(f"Objetos carregados: {len(objects)}\n")
                for index, obj in enumerate(objects, start=1):
                    describe_object(report, obj, index)
            except Exception as exc:
                report.write(f"ERRO: {type(exc).__name__}: {exc}\n")
            report.write("\n" + "-" * 60 + "\n\n")

    print("Diagnóstico concluído.")
    print(f"Arquivo: {OUTPUT}")
    print("Envie o conteúdo desse TXT; ele deve ser pequeno.")


if __name__ == "__main__":
    main()
