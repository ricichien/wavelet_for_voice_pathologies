# -*- coding: utf-8 -*-
"""
AVFAD - extração inicial de características da tarefa 002 (/a/).

Entrada:
    raiz do repositorio\\validacao_tarefa_002.csv
    ZIPs originais da AVFAD no mesmo diretório.

Saída:
    features_avfad_hsa_002.csv
    resumo_extracao_avfad.txt
    erros_extracao_avfad.csv (somente se houver erros)

O script não altera nem extrai os ZIPs originais.
Cada participante gera uma única linha. As três repetições da vogal são
segmentadas por energia e suas características são agregadas por média.
"""

from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import csv
import io
import math
import random
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import pywt
import soundfile as sf
from openpyxl import load_workbook


ROOT = project_root()
INPUT_CSV = asset(ROOT, "validacao_tarefa_002.csv")
OUTPUT_CSV = asset(ROOT, "features_avfad_hsa_002.csv")
REPORT_TXT = asset(ROOT, "resumo_extracao_avfad.txt")
ERROR_CSV = asset(ROOT, "erros_extracao_avfad.csv")

SAMPLE_RATE_EXPECTED = 48000
WAVELET = "db4"
WAVELET_LEVEL = 6
RANDOM_SEED = 42

# Configuração MFCC baseada em janelas de 25 ms, avanço de 10 ms,
# 120 filtros Mel e 25 coeficientes, mantendo toda a banda até 24 kHz.
FRAME_LENGTH = int(0.025 * SAMPLE_RATE_EXPECTED)  # 1200 amostras
HOP_LENGTH = int(0.010 * SAMPLE_RATE_EXPECTED)    # 480 amostras
N_FFT = 2048
N_MELS = 120
N_MFCC = 25
FMIN = 50.0
FMAX = SAMPLE_RATE_EXPECTED / 2

# VAD simples por energia: limiar relativo ao pico do próprio arquivo.
VAD_TOP_DB = 35
VAD_FRAME_LENGTH = 2048
VAD_HOP_LENGTH = 480
VAD_MERGE_GAP_SECONDS = 0.20
VAD_MIN_SEGMENT_SECONDS = 0.75
LONG_SEGMENT_SECONDS = 6.0
LONG_SEGMENT_CHUNK_SECONDS = 4.5


class DatasetError(RuntimeError):
    pass


def load_validated_rows():
    if not INPUT_CSV.exists():
        raise DatasetError(
            f"Não encontrei {INPUT_CSV}. Execute primeiro validar_dataset.py."
        )

    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    required = {
        "participant_id", "label", "age", "sex", "visit_place",
        "audio_archive", "audio_path", "duration_s"
    }
    if not rows:
        raise DatasetError("O CSV de validação está vazio.")

    missing = required - set(rows[0].keys())
    if missing:
        raise DatasetError(f"Colunas ausentes no CSV: {sorted(missing)}")

    # Um único áudio por participante; somente HSA e rótulos binários válidos.
    hsa = []
    seen = set()
    for row in rows:
        pid = row["participant_id"].strip().upper()
        if row["visit_place"].strip().upper() != "HSA":
            continue
        if row["label"] not in {"Normal", "Pathological"}:
            continue
        if pid in seen:
            raise DatasetError(f"ID duplicado no CSV: {pid}")
        seen.add(pid)
        row["participant_id"] = pid
        row["age"] = int(float(row["age"]))
        row["sex"] = row["sex"].strip().upper()
        row["duration_s"] = float(row["duration_s"])
        hsa.append(row)

    if not hsa:
        raise DatasetError("Nenhum participante HSA válido encontrado.")

    return hsa


def load_visit_years():
    """Lê somente File ID e Visit date da planilha dentro dos ZIPs."""
    matches = []
    for zip_path in sorted(raw_dir(ROOT).glob("AVFAD_01_00_00_*.zip")):
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                base = info.filename.replace("\\", "/").split("/")[-1]
                if base.lower() == "avfad_01_00_00.xlsx":
                    matches.append((zip_path, info.filename))

    if len(matches) != 1:
        raise DatasetError(
            "Esperava encontrar exatamente uma AVFAD_01_00_00.xlsx "
            f"nos ZIPs, encontrei {len(matches)}."
        )

    zip_path, internal_path = matches[0]
    with zipfile.ZipFile(zip_path, "r") as archive:
        data = io.BytesIO(archive.read(internal_path))

    wb = load_workbook(data, read_only=True, data_only=True)
    if "AVFAD" not in wb.sheetnames:
        wb.close()
        raise DatasetError("A aba AVFAD não foi encontrada na planilha.")

    ws = wb["AVFAD"]
    iterator = ws.iter_rows(values_only=True)
    headers = next(iterator)
    index = {name: headers.index(name) for name in ("File ID", "Visit date")}

    years = {}
    for row in iterator:
        pid_value = row[index["File ID"]]
        visit_date = row[index["Visit date"]]
        if pid_value is None:
            continue
        pid = str(pid_value).strip().upper()
        years[pid] = visit_date.year if visit_date is not None else None

    wb.close()
    return years


def age_band(age: int) -> int:
    return 18 + ((age - 18) // 5) * 5


def matched_ids(rows, years, include_year=False):
    """Pareamento 1:1 sem reposição dentro dos estratos definidos."""
    buckets = defaultdict(lambda: {"Normal": [], "Pathological": []})

    for row in rows:
        pid = row["participant_id"]
        age = row.get("age")
        sex = row.get("sex")
        year = years.get(pid)
        if age is None or sex not in {"F", "M"}:
            continue
        if include_year and year is None:
            continue

        key = (sex, age_band(int(age)))
        if include_year:
            key = key + (year,)
        buckets[key][row["label"]].append(pid)

    rng = random.Random(RANDOM_SEED)
    selected = set()
    pairs = 0

    for key in sorted(buckets, key=lambda x: tuple(str(v) for v in x)):
        normals = sorted(buckets[key]["Normal"])
        patients = sorted(buckets[key]["Pathological"])
        n = min(len(normals), len(patients))
        if n == 0:
            continue
        selected.update(rng.sample(normals, n))
        selected.update(rng.sample(patients, n))
        pairs += n

    return selected, pairs


def merge_intervals(intervals, max_gap_samples):
    if len(intervals) == 0:
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= max_gap_samples:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(s), int(e)) for s, e in merged]


def split_long_segment(segment, sr):
    """Evita que uma região muito longa seja tratada como uma única emissão."""
    max_len = int(LONG_SEGMENT_SECONDS * sr)
    chunk_len = int(LONG_SEGMENT_CHUNK_SECONDS * sr)
    if len(segment) <= max_len:
        return [segment]

    chunks = []
    start = 0
    while start < len(segment):
        end = min(start + chunk_len, len(segment))
        chunk = segment[start:end]
        if len(chunk) >= int(VAD_MIN_SEGMENT_SECONDS * sr):
            chunks.append(chunk)
        start = end
    return chunks


def segment_vowel_audio(y, sr):
    """Detecta regiões sonoras e retorna trechos com pelo menos 0,75 s."""
    y = np.asarray(y, dtype=np.float64).flatten()
    if y.size == 0:
        raise DatasetError("Áudio vazio.")

    # Remoção de offset DC; não normaliza o volume nem altera a taxa de amostragem.
    y = y - np.mean(y)

    intervals = librosa.effects.split(
        y,
        top_db=VAD_TOP_DB,
        ref=np.max,
        frame_length=VAD_FRAME_LENGTH,
        hop_length=VAD_HOP_LENGTH
    )

    intervals = merge_intervals(
        intervals,
        int(VAD_MERGE_GAP_SECONDS * sr)
    )

    min_len = int(VAD_MIN_SEGMENT_SECONDS * sr)
    segments = []

    for start, end in intervals:
        segment = y[max(0, start):min(len(y), end)]
        if len(segment) >= min_len:
            segments.extend(split_long_segment(segment, sr))

    # Fallback conservador: se o VAD não detectou voz, usa o arquivo todo.
    if not segments:
        if len(y) < min_len:
            raise DatasetError("Áudio curto demais para extrair características.")
        segments = split_long_segment(y, sr)

    return segments


def normalized_entropy(values):
    values = np.asarray(values, dtype=np.float64)
    power = values * values
    total = float(np.sum(power))
    if total <= 1e-20 or power.size <= 1:
        return 0.0
    probabilities = power / total
    probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    denominator = math.log2(power.size)
    return entropy / denominator if denominator > 0 else 0.0


def mfcc_features(segment, sr):
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=N_FFT,
        win_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
        window="hamming",
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        norm="slaney",
        htk=False
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)

    # Derivadas temporais calculadas dentro de cada emissão.
    delta = librosa.feature.delta(mfcc, order=1, width=9, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=9, mode="nearest")

    result = {}
    for label, matrix in (("mfcc", mfcc), ("delta", delta), ("delta2", delta2)):
        means = np.mean(matrix, axis=1)
        stds = np.std(matrix, axis=1)
        for i, value in enumerate(means):
            result[f"{label}_{i:02d}_mean"] = float(value)
        for i, value in enumerate(stds):
            result[f"{label}_{i:02d}_std"] = float(value)
    return result


def dwt_features(segment):
    max_level = pywt.dwt_max_level(len(segment), pywt.Wavelet(WAVELET).dec_len)
    level = min(WAVELET_LEVEL, max_level)
    if level < 1:
        raise DatasetError("Segmento curto demais para DWT.")

    coeffs = pywt.wavedec(
        segment,
        wavelet=WAVELET,
        mode="symmetric",
        level=level
    )
    energies = [float(np.sum(c * c)) for c in coeffs]
    total_energy = sum(energies) + 1e-20

    # cA_level, cD_level, ..., cD1
    labels = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    result = {}
    for band, coeff, energy in zip(labels, coeffs, energies):
        result[f"dwt_{band}_rel_energy"] = energy / total_energy
        result[f"dwt_{band}_entropy"] = normalized_entropy(coeff)
    return result


def wpt_features(segment):
    max_level = pywt.dwt_max_level(len(segment), pywt.Wavelet(WAVELET).dec_len)
    level = min(WAVELET_LEVEL, max_level)
    if level < 1:
        raise DatasetError("Segmento curto demais para WPT.")

    packet = pywt.WaveletPacket(
        data=segment,
        wavelet=WAVELET,
        mode="symmetric",
        maxlevel=level
    )
    nodes = packet.get_level(level, order="freq")
    energies = [float(np.sum(node.data * node.data)) for node in nodes]
    total_energy = sum(energies) + 1e-20

    result = {}
    for i, (node, energy) in enumerate(zip(nodes, energies)):
        result[f"wpt_{i:02d}_rel_energy"] = energy / total_energy
        result[f"wpt_{i:02d}_entropy"] = normalized_entropy(node.data)
    return result


def extract_segment_features(segment, sr):
    features = {}
    features.update(mfcc_features(segment, sr))
    features.update(dwt_features(segment))
    features.update(wpt_features(segment))

    rms = float(np.sqrt(np.mean(segment * segment) + 1e-20))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(
        segment,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )))
    features["signal_rms"] = rms
    features["signal_rms_db"] = 20.0 * math.log10(rms + 1e-12)
    features["signal_zcr"] = zcr

    return features


def aggregate_recording(segments, sr):
    segment_features = [extract_segment_features(seg, sr) for seg in segments]
    keys = segment_features[0].keys()

    # Uma linha por participante: média das três repetições/segmentos detectados.
    aggregated = {}
    for key in keys:
        values = [row[key] for row in segment_features]
        aggregated[key] = float(np.mean(values))

    return aggregated


def main():
    print("Carregando CSV validado e metadados...")
    rows = load_validated_rows()
    years = load_visit_years()

    for row in rows:
        row["visit_year"] = years.get(row["participant_id"])

    age_sex_ids, age_sex_pairs = matched_ids(rows, years, include_year=False)
    age_sex_year_ids, age_sex_year_pairs = matched_ids(rows, years, include_year=True)

    print(f"Participantes HSA: {len(rows)}")
    print(f"Amostra pareada por sexo + faixa etária: {len(age_sex_ids)} ({age_sex_pairs} pares)")
    print(f"Amostra pareada por sexo + faixa etária + ano: {len(age_sex_year_ids)} ({age_sex_year_pairs} pares)")
    print("Extraindo MFCC, DWT e WPT em 48 kHz...")

    archives = {}
    output_rows = []
    errors = []
    segment_counts = []
    segment_durations = []

    try:
        for index, row in enumerate(rows, start=1):
            pid = row["participant_id"]
            archive_name = row["audio_archive"]
            internal_path = row["audio_path"]

            try:
                if archive_name not in archives:
                    zip_file = asset(ROOT, archive_name)
                    if not zip_file.exists():
                        raise DatasetError(f"ZIP não encontrado: {zip_file}")
                    archives[archive_name] = zipfile.ZipFile(zip_file, "r")

                archive = archives[archive_name]
                audio_bytes = archive.read(internal_path)
                y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float64", always_2d=False)

                if y.ndim == 2:
                    y = np.mean(y, axis=1)
                if sr != SAMPLE_RATE_EXPECTED:
                    raise DatasetError(
                        f"Taxa inesperada: {sr} Hz (esperado {SAMPLE_RATE_EXPECTED} Hz)."
                    )
                if not np.all(np.isfinite(y)):
                    raise DatasetError("Áudio contém NaN ou infinito.")

                segments = segment_vowel_audio(y, sr)
                features = aggregate_recording(segments, sr)

                segment_counts.append(len(segments))
                segment_durations.extend([len(s) / sr for s in segments])

                output = {
                    "participant_id": pid,
                    "label": row["label"],
                    "age": row["age"],
                    "sex": row["sex"],
                    "visit_place": row["visit_place"],
                    "visit_year": row["visit_year"],
                    "duration_s": row["duration_s"],
                    "n_segments": len(segments),
                    "matched_age_sex": int(pid in age_sex_ids),
                    "matched_age_sex_year": int(pid in age_sex_year_ids),
                }
                output.update(features)
                output_rows.append(output)

            except Exception as exc:
                errors.append({"participant_id": pid, "error": repr(exc)})

            if index % 25 == 0 or index == len(rows):
                print(f"  Processados: {index}/{len(rows)} | válidos: {len(output_rows)} | erros: {len(errors)}")

    finally:
        for archive in archives.values():
            archive.close()

    if output_rows:
        fieldnames = list(output_rows[0].keys())
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    if errors:
        with ERROR_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["participant_id", "error"])
            writer.writeheader()
            writer.writerows(errors)

    feature_names = [
        key for key in output_rows[0].keys()
        if key not in {
            "participant_id", "label", "age", "sex", "visit_place",
            "visit_year", "duration_s", "n_segments",
            "matched_age_sex", "matched_age_sex_year"
        }
    ] if output_rows else []

    with REPORT_TXT.open("w", encoding="utf-8") as report:
        report.write("EXTRAÇÃO DE CARACTERÍSTICAS — AVFAD TAREFA 002\n")
        report.write("=" * 58 + "\n")
        report.write(f"Participantes HSA de entrada: {len(rows)}\n")
        report.write(f"Características extraídas: {len(output_rows)}\n")
        report.write(f"Erros: {len(errors)}\n")
        report.write(f"Dimensão do vetor acústico: {len(feature_names)}\n")
        report.write(f"Amostra pareada sexo + idade: {len(age_sex_ids)} ({age_sex_pairs} pares)\n")
        report.write(f"Amostra pareada sexo + idade + ano: {len(age_sex_year_ids)} ({age_sex_year_pairs} pares)\n")
        report.write(f"Taxa de amostragem preservada: {SAMPLE_RATE_EXPECTED} Hz\n")
        report.write(f"Wavelet: {WAVELET}; nível máximo: {WAVELET_LEVEL}\n")
        report.write("MFCC: 25 coeficientes; 120 filtros Mel; janela 25 ms; avanço 10 ms; fmax 24 kHz\n")

        if segment_counts:
            report.write("\nSegmentação por participante:\n")
            report.write(f"  Mínimo de segmentos: {min(segment_counts)}\n")
            report.write(f"  Mediana de segmentos: {statistics.median(segment_counts):.1f}\n")
            report.write(f"  Máximo de segmentos: {max(segment_counts)}\n")
        if segment_durations:
            report.write("\nDuração dos segmentos detectados:\n")
            report.write(f"  Mínima: {min(segment_durations):.2f} s\n")
            report.write(f"  Mediana: {statistics.median(segment_durations):.2f} s\n")
            report.write(f"  Máxima: {max(segment_durations):.2f} s\n")

        if errors:
            report.write("\nPrimeiros erros:\n")
            for item in errors[:20]:
                report.write(f"  {item['participant_id']}: {item['error']}\n")

        report.write("\nArquivos gerados:\n")
        report.write(f"  {OUTPUT_CSV.name}\n")
        report.write(f"  {REPORT_TXT.name}\n")
        if errors:
            report.write(f"  {ERROR_CSV.name}\n")

    print("\nExtração concluída.")
    print(f"Características extraídas: {len(output_rows)}/{len(rows)}")
    print(f"Dimensão do vetor acústico: {len(feature_names)}")
    print(f"Relatório: {REPORT_TXT}")
    print(f"CSV de características: {OUTPUT_CSV}")
    if errors:
        print(f"Há erros; veja: {ERROR_CSV}")

    if len(output_rows) != len(rows):
        print("ATENÇÃO: não use os resultados para treinamento enquanto os erros não forem avaliados.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERRO:", exc)
        raise
