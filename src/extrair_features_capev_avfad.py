# -*- coding: utf-8 -*-
"""
AVFAD - extração acústica das seis frases CAPE-V (tarefas 004 a 009).

Entrada:
  raiz do repositorio\\validacao_tarefa_002.csv
  ZIPs originais da AVFAD no mesmo diretório.

Saídas:
  features_avfad_capev_hsa_004_009.csv
  resumo_extracao_capev_avfad.txt
  erros_extracao_capev_avfad.csv (somente se houver erros)

O script não extrai nem modifica os ZIPs originais.
Uma linha por participante. Para cada tarefa (004-009), as características
são calculadas em segmentos de fala detectados por energia e agregadas por
média. Também são calculadas média e variabilidade entre as seis tarefas,
dando peso igual a cada frase.

Nota metodológica: a detecção por energia serve para separar repetições/trechos
com pausas longas, não para segmentar fonemas. O relatório registra quantos
segmentos foram detectados em cada tarefa para permitir auditoria.
"""

from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import csv
import io
import math
import random
import statistics
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
import re

import librosa
import numpy as np
import pywt
import soundfile as sf
from scipy.signal import resample_poly
from openpyxl import load_workbook


ROOT = project_root()
INPUT_CSV = asset(ROOT, "validacao_tarefa_002.csv")
OUTPUT_CSV = asset(ROOT, "features_avfad_capev_hsa_004_009.csv")
REPORT_TXT = asset(ROOT, "resumo_extracao_capev_avfad.txt")
ERROR_CSV = asset(ROOT, "erros_extracao_capev_avfad.csv")

TASKS = ("004", "005", "006", "007", "008", "009")
SAMPLE_RATE_TARGET = 48000
WAVELET = "db4"
WAVELET_LEVEL = 6
RANDOM_SEED = 42

# MFCC: 25 ms / 10 ms; 25 coeficientes e 120 filtros Mel.
FRAME_LENGTH = int(0.025 * SAMPLE_RATE_TARGET)
HOP_LENGTH = int(0.010 * SAMPLE_RATE_TARGET)
N_FFT = 2048
N_MELS = 120
N_MFCC = 25
FMIN = 50.0
FMAX = SAMPLE_RATE_TARGET / 2

# VAD conservador para frases: preserva transições curtas e une pausas
# intrafrase de até 300 ms. Não é uma segmentação fonética.
VAD_TOP_DB = 40
VAD_FRAME_LENGTH = 2048
VAD_HOP_LENGTH = 480
VAD_MERGE_GAP_SECONDS = 0.30
VAD_MIN_SEGMENT_SECONDS = 0.35


class DatasetError(RuntimeError):
    pass


def load_validated_rows():
    if not INPUT_CSV.exists():
        raise DatasetError(
            f"Não encontrei {INPUT_CSV}. O arquivo validacao_tarefa_002.csv "
            "gerado na validação inicial é necessário."
        )

    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise DatasetError("O CSV de validação está vazio.")

    required = {
        "participant_id", "label", "age", "sex", "visit_place"
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise DatasetError(f"Colunas ausentes no CSV: {sorted(missing)}")

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
        try:
            row["age"] = int(float(row["age"]))
        except (TypeError, ValueError):
            row["age"] = None
        row["sex"] = row["sex"].strip().upper()
        hsa.append(row)

    if not hsa:
        raise DatasetError("Nenhum participante HSA válido encontrado.")
    return sorted(hsa, key=lambda r: r["participant_id"])


def load_visit_years():
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
            "Esperava encontrar exatamente uma AVFAD_01_00_00.xlsx nos ZIPs; "
            f"encontrei {len(matches)}."
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
    idx = {name: headers.index(name) for name in ("File ID", "Visit date")}

    years = {}
    for row in iterator:
        value = row[idx["File ID"]]
        visit_date = row[idx["Visit date"]]
        if value is None:
            continue
        pid = str(value).strip().upper()
        if isinstance(visit_date, (datetime, date)):
            years[pid] = visit_date.year
        else:
            years[pid] = None

    wb.close()
    return years


def age_band(age: int) -> int:
    return 18 + ((age - 18) // 5) * 5


def matched_ids(rows, years, include_year=False):
    """Reproduz o pareamento 1:1 anterior, sem reposição e com seed fixo."""
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
            key += (year,)
        buckets[key][row["label"]].append(pid)

    rng = random.Random(RANDOM_SEED)
    selected = set()
    pairs = 0
    for key in sorted(buckets, key=lambda x: tuple(str(v) for v in x)):
        normals = sorted(buckets[key]["Normal"])
        patients = sorted(buckets[key]["Pathological"])
        n = min(len(normals), len(patients))
        if n:
            selected.update(rng.sample(normals, n))
            selected.update(rng.sample(patients, n))
            pairs += n
    return selected, pairs


def build_audio_index(zip_paths):
    """Indexa basename ZZZ004.wav ... ZZZ009.wav sem extrair os ZIPs."""
    index = defaultdict(list)
    malformed = []
    file_count = 0
    pattern = re.compile(r"^([A-Z]{3})(00[4-9])\.WAV$", re.IGNORECASE)

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                basename = info.filename.replace("\\", "/").split("/")[-1]
                match = pattern.fullmatch(basename.upper())
                if match:
                    pid, task = match.group(1).upper(), match.group(2)
                    index[(pid, task)].append((zip_path.name, info.filename))
                    file_count += 1
                elif basename.upper().endswith("00") and basename.lower().endswith(".wav"):
                    malformed.append(info.filename)
    return index, malformed, file_count


def merge_intervals(intervals, max_gap_samples):
    if not len(intervals):
        return []
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start - merged[-1][1] <= max_gap_samples:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(int(s), int(e)) for s, e in merged]


def segment_sentence(y, sr):
    """Detecta blocos de fala separados por pausas; preserva a frase inteira."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise DatasetError("Sinal vazio ou com valores não finitos.")

    y = y - np.mean(y)
    if float(np.max(np.abs(y))) < 1e-8:
        raise DatasetError("Gravação silenciosa ou com amplitude praticamente nula.")

    intervals = librosa.effects.split(
        y,
        top_db=VAD_TOP_DB,
        ref=np.max,
        frame_length=VAD_FRAME_LENGTH,
        hop_length=VAD_HOP_LENGTH,
    )
    intervals = merge_intervals(
        intervals, int(VAD_MERGE_GAP_SECONDS * sr)
    )

    min_len = int(VAD_MIN_SEGMENT_SECONDS * sr)
    segments = [
        y[max(0, start):min(len(y), end)]
        for start, end in intervals
        if end - start >= min_len
    ]

    if not segments:
        # Não descarta uma gravação válida só porque o VAD não separou pausas.
        if len(y) < int(0.20 * sr):
            raise DatasetError("Gravação curta demais para análise acústica.")
        segments = [y]

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
        htk=False,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)

    # A duração mínima dos segmentos é suficiente para width=9 em condições normais.
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
    if width < 3:
        raise DatasetError("Poucos frames para calcular deltas MFCC.")
    delta = librosa.feature.delta(mfcc, order=1, width=width, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=width, mode="nearest")

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
    coeffs = pywt.wavedec(segment, wavelet=WAVELET, mode="symmetric", level=level)
    energies = [float(np.sum(c * c)) for c in coeffs]
    total_energy = sum(energies) + 1e-20
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
        data=segment, wavelet=WAVELET, mode="symmetric", maxlevel=level
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
        segment, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )))
    features["signal_rms"] = rms
    features["signal_rms_db"] = 20.0 * math.log10(rms + 1e-12)
    features["signal_zcr"] = zcr
    return features


def aggregate_segments(segments, sr):
    per_segment = [extract_segment_features(seg, sr) for seg in segments]
    keys = per_segment[0].keys()
    return {
        key: float(np.mean([item[key] for item in per_segment]))
        for key in keys
    }


def read_audio(archive, internal_path):
    raw = archive.read(internal_path)
    audio, sr = sf.read(
        io.BytesIO(raw), dtype="float64", always_2d=True
    )
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise DatasetError("WAV vazio ou com estrutura inesperada.")
    if not np.all(np.isfinite(audio)):
        raise DatasetError("Áudio contém NaN ou infinito.")

    channels = audio.shape[1]
    # Downmix simples para mono; registra o número original de canais no relatório.
    mono = np.mean(audio, axis=1)
    original_sr = int(sr)
    if original_sr != SAMPLE_RATE_TARGET:
        from math import gcd
        common = gcd(original_sr, SAMPLE_RATE_TARGET)
        up = SAMPLE_RATE_TARGET // common
        down = original_sr // common
        mono = resample_poly(mono, up, down).astype(np.float64, copy=False)
        sr = SAMPLE_RATE_TARGET

    mono = mono - np.mean(mono)
    return mono, int(sr), original_sr, channels, len(raw)


def main():
    print("Carregando participantes HSA e anos de visita...")
    rows = load_validated_rows()
    years = load_visit_years()
    for row in rows:
        row["visit_year"] = years.get(row["participant_id"])

    matched_age_sex, pairs_age_sex = matched_ids(rows, years, include_year=False)
    matched_age_sex_year, pairs_age_sex_year = matched_ids(rows, years, include_year=True)

    zip_paths = sorted(raw_dir(ROOT).glob("AVFAD_01_00_00_*.zip"))
    if not zip_paths:
        raise DatasetError(f"Nenhum ZIP encontrado em {ROOT}.")
    print("Indexando tarefas 004-009 nos ZIPs (sem extrair arquivos)...")
    audio_index, malformed, indexed_count = build_audio_index(zip_paths)

    missing = []
    duplicated = []
    for row in rows:
        pid = row["participant_id"]
        for task in TASKS:
            items = audio_index.get((pid, task), [])
            if len(items) == 0:
                missing.append((pid, task))
            elif len(items) > 1:
                duplicated.append((pid, task, items))

    if missing or duplicated:
        detail = [
            f"Faltando: {len(missing)}; duplicados: {len(duplicated)}."
        ]
        detail.extend([f"Ausente {pid}{task}." for pid, task in missing[:20]])
        detail.extend([f"Duplicado {pid}{task}: {items}" for pid, task, items in duplicated[:10]])
        raise DatasetError("\n".join(detail))

    print(f"Participantes HSA: {len(rows)} | gravações esperadas: {len(rows) * len(TASKS)}")
    print("Extraindo MFCC, DWT e WPT das frases CAPE-V...")

    archives = {}
    output_rows = []
    errors = []
    task_segment_counts = defaultdict(list)
    task_segment_durations = defaultdict(list)
    formats = Counter()
    task_source_counts = Counter()
    total_segments = 0

    try:
        for participant_index, row in enumerate(rows, start=1):
            pid = row["participant_id"]
            output = {
                "participant_id": pid,
                "label": row["label"],
                "age": row["age"],
                "sex": row["sex"],
                "visit_place": row["visit_place"],
                "visit_year": row["visit_year"],
                "matched_age_sex": int(pid in matched_age_sex),
                "matched_age_sex_year": int(pid in matched_age_sex_year),
            }
            task_feature_vectors = []
            participant_ok = True

            for task in TASKS:
                archive_name, internal_path = audio_index[(pid, task)][0]
                try:
                    if archive_name not in archives:
                        archives[archive_name] = zipfile.ZipFile(asset(ROOT, archive_name), "r")
                    y, sr, original_sr, channels, byte_count = read_audio(
                        archives[archive_name], internal_path
                    )
                    formats[(task, original_sr, channels)] += 1
                    segments = segment_sentence(y, sr)
                    features = aggregate_segments(segments, sr)

                    # Prefix por frase permite analisar cada tarefa isoladamente depois.
                    for feature_name, value in features.items():
                        output[f"t{task}_{feature_name}"] = value

                    task_feature_vectors.append(features)
                    task_segment_counts[task].append(len(segments))
                    task_segment_durations[task].extend(
                        len(segment) / sr for segment in segments
                    )
                    total_segments += len(segments)
                    task_source_counts[task] += 1
                    output[f"n_segments_t{task}"] = len(segments)
                    output[f"voiced_duration_t{task}_s"] = round(
                        sum(len(segment) for segment in segments) / sr, 4
                    )
                    output[f"source_t{task}"] = internal_path

                except Exception as exc:
                    participant_ok = False
                    errors.append({
                        "participant_id": pid,
                        "task": task,
                        "file": internal_path,
                        "error": repr(exc),
                    })
                    break

            if participant_ok and len(task_feature_vectors) == len(TASKS):
                common_features = list(task_feature_vectors[0].keys())
                # Média e desvio entre frases: cada uma das seis tarefas tem o mesmo peso.
                for feature_name in common_features:
                    values = [v[feature_name] for v in task_feature_vectors]
                    output[f"all_mean_{feature_name}"] = float(np.mean(values))
                    output[f"all_std_{feature_name}"] = float(np.std(values))
                output_rows.append(output)

            if participant_index % 10 == 0 or participant_index == len(rows):
                print(
                    f"  Participantes: {participant_index}/{len(rows)} | "
                    f"completos: {len(output_rows)} | erros: {len(errors)}"
                )

    finally:
        for archive in archives.values():
            archive.close()

    if not output_rows:
        raise DatasetError("Nenhum participante foi extraído com sucesso.")

    # Colunas estáveis, mesmo quando um ou mais participantes falharem.
    fieldnames = list(output_rows[0].keys())
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    if errors:
        with ERROR_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=["participant_id", "task", "file", "error"]
            )
            writer.writeheader()
            writer.writerows(errors)

    labels = Counter(r["label"] for r in output_rows)
    report_lines = [
        "EXTRAÇÃO CAPE-V — AVFAD TAREFAS 004-009",
        "=" * 62,
        f"Participantes HSA de entrada: {len(rows)}",
        f"Participantes com as seis tarefas extraídas: {len(output_rows)}",
        f"Participantes com erro: {len(set(e['participant_id'] for e in errors))}",
        f"Gravações esperadas: {len(rows) * len(TASKS)}",
        f"Arquivos CAPE-V indexados nos ZIPs: {indexed_count}",
        f"Tarefas: {', '.join(TASKS)}",
        f"Amostra pareada sexo + idade: {len(matched_age_sex)} ({pairs_age_sex} pares)",
        f"Amostra pareada sexo + idade + ano: {len(matched_age_sex_year)} ({pairs_age_sex_year} pares)",
        f"Classes extraídas: Normal={labels['Normal']}; Pathological={labels['Pathological']}",
        f"Taxa-alvo: {SAMPLE_RATE_TARGET} Hz; downmix para mono; wavelet={WAVELET}; nível máximo={WAVELET_LEVEL}",
        "MFCC: 25 coeficientes; 120 filtros Mel; janela 25 ms; avanço 10 ms; fmax 24 kHz",
        f"VAD: top_db={VAD_TOP_DB}; junção de pausas até {VAD_MERGE_GAP_SECONDS:.2f}s; segmento mínimo={VAD_MIN_SEGMENT_SECONDS:.2f}s",
        "Agregação: média dos segmentos dentro de cada frase; média e DP entre as seis frases (peso igual por frase).",
        "Cada participante gera uma única linha; metadados não devem entrar como preditores acústicos.",
        "",
        "DIMENSÃO DAS CARACTERÍSTICAS",
        f"Características por frase: {len([c for c in fieldnames if c.startswith('t004_')])}",
        f"Frases individuais: {sum(c.startswith(f't{task}_') for c in fieldnames for task in TASKS)}",
        f"Características globais (média + DP entre frases): {sum(c.startswith('all_mean_') or c.startswith('all_std_') for c in fieldnames)}",
        f"Total de segmentos detectados: {total_segments}",
        "",
        "SEGMENTAÇÃO POR TAREFA (número de blocos por gravação)",
    ]

    for task in TASKS:
        counts = task_segment_counts[task]
        durations = task_segment_durations[task]
        if counts:
            report_lines.append(
                f"Tarefa {task}: gravações={task_source_counts[task]}; "
                f"segmentos/arquivo min={min(counts)}, mediana={statistics.median(counts):.1f}, max={max(counts)}"
            )
            if durations:
                report_lines.append(
                    f"  duração dos segmentos: min={min(durations):.2f}s; "
                    f"mediana={statistics.median(durations):.2f}s; max={max(durations):.2f}s"
                )
        else:
            report_lines.append(f"Tarefa {task}: nenhuma gravação processada.")

    report_lines.extend(["", "FORMATOS ORIGINAIS (tarefa, Hz, canais)"])
    for key, count in sorted(formats.items()):
        task, rate, channels = key
        report_lines.append(f"  {task}: {rate} Hz, {channels} canal(is): {count} arquivos")

    if errors:
        report_lines.extend(["", "PRIMEIROS ERROS"])
        for item in errors[:30]:
            report_lines.append(
                f"  {item['participant_id']} tarefa {item['task']} "
                f"({item['file']}): {item['error']}"
            )

    report_lines.extend(["", "ARQUIVOS GERADOS", f"  {OUTPUT_CSV.name}", f"  {REPORT_TXT.name}"])
    if errors:
        report_lines.append(f"  {ERROR_CSV.name}")

    REPORT_TXT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\nExtração CAPE-V concluída.")
    print(f"Participantes completos: {len(output_rows)}/{len(rows)}")
    print(f"Erros de tarefa/áudio: {len(errors)}")
    print(f"Relatório: {REPORT_TXT}")
    print(f"CSV de características: {OUTPUT_CSV}")
    if len(output_rows) != len(rows):
        print("ATENÇÃO: não treine até revisar os participantes com tarefas ausentes/erro.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERRO:", exc)
        raise
