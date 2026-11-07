# -*- coding: utf-8 -*-
"""Extracao wavelet temporal AVFAD CAPE-V (tarefas 004-009), HSA.

Objetivo:
  Criar descritores wavelet mais ricos do que o baseline db4 global anterior.

Entrada:
  metadata_capev_avfad.csv (gerado pelo extrator CAPE-V + ComParE)
  ZIPs originais da AVFAD no diretorio raiz do repositorio

Saidas:
  features_wavelet_temporal_avfad.npz
  metadata_wavelet_temporal_avfad.csv
  resumo_wavelet_temporal_avfad.txt
  erros_wavelet_temporal_avfad.csv (se houver)
  cache_wavelet_temporal_avfad.sqlite

Metodologia:
  - Le cada WAV completo dos ZIPs, converte para mono e mantem 48 kHz.
  - Detecta regioes de fala com librosa.effects.split; une pausas curtas.
  - Cria janelas sobrepostas de 400 ms com passo de 200 ms dentro da fala.
  - Extrai DWT (nivel 7) e WPT (niveis 5, 6 e 7) com 5 familias wavelet.
  - Por banda/no, calcula energia relativa, log-energia, entropia, curtose e ZCR.
  - Resume a distribuicao temporal das janelas por media, DP, q10 e q90.
  - Uma linha por gravacao; a avaliacao posterior deve agrupar por participante.

Importante: o codigo nao treina modelos nem usa os rotulos para selecionar features.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import csv
import io
import json
import math
import sqlite3
import time
import zipfile
from collections import Counter
from pathlib import Path
from statistics import median

import librosa
import numpy as np
import pywt
import soundfile as sf
from scipy.signal import resample_poly
from scipy.stats import kurtosis
from math import gcd

ROOT = project_root()
INPUT_META = asset(ROOT, "metadata_capev_avfad.csv")
OUTPUT_NPZ = asset(ROOT, "features_wavelet_temporal_avfad.npz")
OUTPUT_META = asset(ROOT, "metadata_wavelet_temporal_avfad.csv")
OUTPUT_REPORT = asset(ROOT, "resumo_wavelet_temporal_avfad.txt")
OUTPUT_ERRORS = asset(ROOT, "erros_wavelet_temporal_avfad.csv")
CACHE_DB = asset(ROOT, "cache_wavelet_temporal_avfad.sqlite")

VERSION = "wavelet_temporal_v1"
TARGET_SR = 48000
TASKS = {"004", "005", "006", "007", "008", "009"}
WAVELETS = ("haar", "db4", "db8", "sym5", "coif3")
DWT_LEVEL = 7
WPT_LEVELS = (5, 6, 7)
DESCRIPTORS = ("rel_energy", "log_energy", "entropy", "kurtosis", "zcr")
SUMMARY_STATS = ("mean", "std", "q10", "q90")

# VAD: parametros conservadores para fala, nao para segmentacao de fonemas.
VAD_TOP_DB = 40
VAD_FRAME_LENGTH = 2048     # 42,7 ms a 48 kHz
VAD_HOP_LENGTH = 480        # 10 ms
VAD_MERGE_GAP_S = 0.25
VAD_MIN_REGION_S = 0.25

# Janelamento wavelet temporal.
WINDOW_S = 0.40
WINDOW_HOP_S = 0.20
EPS = 1e-12


class PipelineError(RuntimeError):
    pass


def load_metadata():
    if not INPUT_META.exists():
        raise PipelineError(
            f"Nao encontrei {INPUT_META.name}. Execute/guarde primeiro "
            "a extracao CAPE-V + ComParE que gerou metadata_capev_avfad.csv."
        )
    with INPUT_META.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise PipelineError("metadata_capev_avfad.csv esta vazio.")
    required = {
        "row_index", "sample_id", "participant_id", "task", "label", "y",
        "age", "sex", "visit_year", "visit_place", "zip_name", "audio_path_in_zip"
    }
    missing = required - set(rows[0])
    if missing:
        raise PipelineError(f"Colunas ausentes nos metadados: {sorted(missing)}")
    rows.sort(key=lambda r: int(r["row_index"]))
    seen = set()
    for row in rows:
        row["participant_id"] = row["participant_id"].strip().upper()
        row["task"] = row["task"].strip().zfill(3)
        row["sample_id"] = row["sample_id"].strip().upper()
        row["zip_name"] = Path(row["zip_name"]).name
        if row["task"] not in TASKS:
            raise PipelineError(f"Tarefa inesperada em {row['sample_id']}: {row['task']}")
        key = (row["participant_id"], row["task"])
        if key in seen:
            raise PipelineError(f"Participante/tarefa duplicado: {key}")
        seen.add(key)
        if row["label"] not in ("Normal", "Pathological"):
            raise PipelineError(f"Rotulo inesperado em {row['sample_id']}: {row['label']}")
    return rows


def build_feature_names():
    names = []
    base_names = []
    for family in WAVELETS:
        # DWT nivel 7: aproximacao A7 e detalhes D7 ate D1.
        dwt_nodes = ["A7"] + [f"D{i}" for i in range(7, 0, -1)]
        for node in dwt_nodes:
            for desc in DESCRIPTORS:
                base_names.append(f"DWT__{family}__L7__{node}__{desc}")
        # WPT: tres resolucoes; nodes estao ordenados por frequencia.
        for level in WPT_LEVELS:
            for node in range(2 ** level):
                for desc in DESCRIPTORS:
                    base_names.append(
                        f"WPT__{family}__L{level}__N{node:03d}__{desc}"
                    )
    for stat in SUMMARY_STATS:
        names.extend(f"{base}__{stat}" for base in base_names)
    return names, base_names


def merge_intervals(intervals, max_gap_samples):
    if len(intervals) == 0:
        return []
    merged = []
    start, end = int(intervals[0][0]), int(intervals[0][1])
    for next_start, next_end in intervals[1:]:
        next_start, next_end = int(next_start), int(next_end)
        if next_start - end <= max_gap_samples:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return merged


def detect_speech_regions(signal, sr):
    intervals = librosa.effects.split(
        signal.astype(np.float32, copy=False),
        top_db=VAD_TOP_DB,
        ref=np.max,
        frame_length=VAD_FRAME_LENGTH,
        hop_length=VAD_HOP_LENGTH,
    )
    intervals = merge_intervals(intervals, int(round(VAD_MERGE_GAP_S * sr)))
    min_samples = int(round(VAD_MIN_REGION_S * sr))
    regions = [
        signal[max(0, start):min(len(signal), end)]
        for start, end in intervals
        if min_samples <= end - start
    ]
    fallback = False
    if not regions:
        if len(signal) < int(0.20 * sr):
            raise PipelineError("Audio curto demais, sem regiao de fala utilizavel.")
        # Nao perde uma gravacao valida quando o VAD falha.
        regions = [signal]
        fallback = True
    return regions, fallback


def make_windows(regions, sr):
    win = int(round(WINDOW_S * sr))
    hop = int(round(WINDOW_HOP_S * sr))
    windows = []
    for region in regions:
        x = np.asarray(region, dtype=np.float32)
        if len(x) == 0:
            continue
        if len(x) <= win:
            if len(x) < win:
                pad = win - len(x)
                mode = "reflect" if len(x) > 1 else "edge"
                x = np.pad(x, (0, pad), mode=mode)
            windows.append(x[:win])
            continue
        starts = list(range(0, len(x) - win + 1, hop))
        last = len(x) - win
        if not starts or starts[-1] != last:
            starts.append(last)
        for start in starts:
            windows.append(x[start:start + win])
    if not windows:
        raise PipelineError("Nenhuma janela wavelet foi criada.")
    return windows


def normalized_entropy(coeff):
    c = np.asarray(coeff, dtype=np.float64)
    power = c * c
    total = float(power.sum())
    if not math.isfinite(total) or total <= EPS or power.size <= 1:
        return 0.0
    p = power / total
    p = p[p > 0]
    h = -float(np.sum(p * np.log2(p)))
    denom = math.log2(power.size)
    return float(h / denom) if denom > 0 else 0.0


def coefficient_descriptors(coeff, rel_energy):
    c = np.asarray(coeff, dtype=np.float64)
    energy = float(np.dot(c, c))
    log_energy = math.log10(energy / max(1, c.size) + EPS)
    ent = normalized_entropy(c)
    if c.size > 3 and float(np.std(c)) > 1e-14:
        k = float(kurtosis(c, fisher=True, bias=False, nan_policy="omit"))
        if not math.isfinite(k):
            k = 0.0
        k = float(np.clip(k, -50.0, 50.0))
    else:
        k = 0.0
    zcr = float(np.mean(np.signbit(c[1:]) != np.signbit(c[:-1]))) if c.size > 1 else 0.0
    return [float(rel_energy), log_energy, ent, k, zcr]


def wavelet_features_one_window(window):
    values = []
    x = np.asarray(window, dtype=np.float64)
    x = x - np.mean(x)
    for family in WAVELETS:
        wavelet = pywt.Wavelet(family)
        max_allowed = pywt.dwt_max_level(len(x), wavelet.dec_len)
        if max_allowed < DWT_LEVEL:
            raise PipelineError(
                f"Janela insuficiente para DWT nivel {DWT_LEVEL} com {family}."
            )

        # DWT multiescala: todos os detalhes D1..D7 e aproximacao A7.
        dwt_coeffs = pywt.wavedec(
            x, wavelet=wavelet, mode="symmetric", level=DWT_LEVEL
        )
        dwt_energies = np.asarray(
            [float(np.dot(c, c)) for c in dwt_coeffs], dtype=np.float64
        )
        dwt_total = float(dwt_energies.sum()) + EPS
        for coeff, energy in zip(dwt_coeffs, dwt_energies):
            values.extend(coefficient_descriptors(coeff, float(energy / dwt_total)))

        # Uma arvore WPT por familia; reutiliza a decomposicao para niveis 5, 6 e 7.
        packet = pywt.WaveletPacket(
            data=x, wavelet=wavelet, mode="symmetric", maxlevel=max(WPT_LEVELS)
        )
        for level in WPT_LEVELS:
            nodes = packet.get_level(level, order="freq")
            energies = np.asarray(
                [float(np.dot(node.data, node.data)) for node in nodes],
                dtype=np.float64,
            )
            total = float(energies.sum()) + EPS
            for node, energy in zip(nodes, energies):
                values.extend(
                    coefficient_descriptors(node.data, float(energy / total))
                )

    result = np.asarray(values, dtype=np.float32)
    if result.size != len(build_feature_names()[1]):
        raise PipelineError(
            f"Dimensao por janela inesperada: {result.size}; "
            f"esperada {len(build_feature_names()[1])}."
        )
    if not np.isfinite(result).all():
        raise PipelineError("Caracteristicas wavelet contem NaN ou infinito.")
    return result


def extract_recording(signal, sr):
    regions, fallback = detect_speech_regions(signal, sr)
    windows = make_windows(regions, sr)
    matrix = np.vstack([wavelet_features_one_window(w) for w in windows])
    summaries = [
        np.mean(matrix, axis=0),
        np.std(matrix, axis=0),
        np.quantile(matrix, 0.10, axis=0),
        np.quantile(matrix, 0.90, axis=0),
    ]
    vector = np.concatenate(summaries).astype(np.float32, copy=False)
    active_samples = sum(len(r) for r in regions)
    original_duration = len(signal) / float(sr)
    active_duration = active_samples / float(sr)
    diagnostics = {
        "speech_regions": len(regions),
        "windows": len(windows),
        "active_duration_s": active_duration,
        "recording_duration_s": original_duration,
        "speech_ratio": active_duration / max(original_duration, EPS),
        "vad_fallback": int(fallback),
    }
    return vector, diagnostics


def init_cache(path, signature):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM config WHERE key='signature'").fetchone()
    if row is None:
        conn.execute("INSERT INTO config(key,value) VALUES('signature',?)", (signature,))
        conn.commit()
    elif row[0] != signature:
        conn.close()
        raise PipelineError(
            "O cache existente foi criado com outra configuracao. "
            "Nao o apague: renomeie cache_wavelet_temporal_avfad.sqlite "
            "para cache_wavelet_temporal_avfad_antigo.sqlite e execute novamente."
        )
    conn.execute("""CREATE TABLE IF NOT EXISTS features (
        sample_id TEXT PRIMARY KEY,
        participant_id TEXT NOT NULL,
        task TEXT NOT NULL,
        vector_blob BLOB NOT NULL,
        speech_regions INTEGER NOT NULL,
        windows INTEGER NOT NULL,
        active_duration_s REAL NOT NULL,
        recording_duration_s REAL NOT NULL,
        speech_ratio REAL NOT NULL,
        vad_fallback INTEGER NOT NULL,
        sample_rate_hz INTEGER NOT NULL,
        channels_original INTEGER NOT NULL
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS errors (
        sample_id TEXT PRIMARY KEY,
        participant_id TEXT NOT NULL,
        task TEXT NOT NULL,
        error TEXT NOT NULL
    )""")
    conn.commit()
    return conn


def main():
    started = time.time()
    rows = load_metadata()
    if len(rows) != 2465:
        print(f"ATENCAO: metadata contem {len(rows)} gravacoes, nao 2465.")
    people = {r["participant_id"] for r in rows}
    labels = {}
    for r in rows:
        previous = labels.setdefault(r["participant_id"], r["label"])
        if previous != r["label"]:
            raise PipelineError(f"Rotulos inconsistentes para {r['participant_id']}.")

    feature_names, base_names = build_feature_names()
    expected_dim = len(feature_names)
    # 5 families x (DWT 8 bands + WPT levels 5/6/7: 224 nodes) x 5 descriptors x 4 summaries.
    if expected_dim != 23200:
        raise PipelineError(f"Dimensao configurada inesperada: {expected_dim}, esperado 23200.")

    signature = json.dumps({
        "version": VERSION,
        "wavelets": WAVELETS,
        "dwt_level": DWT_LEVEL,
        "wpt_levels": WPT_LEVELS,
        "descriptors": DESCRIPTORS,
        "summaries": SUMMARY_STATS,
        "vad": [VAD_TOP_DB, VAD_FRAME_LENGTH, VAD_HOP_LENGTH, VAD_MERGE_GAP_S, VAD_MIN_REGION_S],
        "window": [WINDOW_S, WINDOW_HOP_S],
        "target_sr": TARGET_SR,
    }, sort_keys=True)
    conn = init_cache(CACHE_DB, signature)

    completed = {row[0] for row in conn.execute("SELECT sample_id FROM features")}
    print(f"Participantes: {len(people)} | gravacoes no manifesto: {len(rows)}")
    print(f"Cache: {len(completed)} gravacoes ja processadas.")
    print(f"Wavelets: {', '.join(WAVELETS)} | DWT L{DWT_LEVEL} | WPT L{WPT_LEVELS}")
    print(f"Janelas: {WINDOW_S:.2f}s / passo {WINDOW_HOP_S:.2f}s | VAD top_db={VAD_TOP_DB}")

    archives = {}
    run_errors = []
    processed_now = 0
    try:
        for i, row in enumerate(rows, start=1):
            sample_id = row["sample_id"]
            if sample_id in completed:
                continue
            zip_name = row["zip_name"]
            internal_path = row["audio_path_in_zip"]
            try:
                if zip_name not in archives:
                    archive_path = asset(ROOT, zip_name)
                    if not archive_path.exists():
                        raise PipelineError(f"ZIP nao encontrado: {archive_path.name}")
                    archives[zip_name] = zipfile.ZipFile(archive_path, "r")
                raw = archives[zip_name].read(internal_path)
                audio, sr = sf.read(
                    io.BytesIO(raw), dtype="float32", always_2d=True
                )
                if audio.ndim != 2 or audio.shape[0] < 1:
                    raise PipelineError("WAV vazio ou com formato invalido.")
                if not np.isfinite(audio).all():
                    raise PipelineError("WAV contem NaN ou infinito.")
                channels = int(audio.shape[1])
                signal = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
                sr = int(sr)
                if sr != TARGET_SR:
                    common = gcd(sr, TARGET_SR)
                    signal = resample_poly(
                        signal, TARGET_SR // common, sr // common
                    ).astype(np.float32)
                    sr = TARGET_SR
                signal = signal - np.mean(signal, dtype=np.float64)
                peak = float(np.max(np.abs(signal))) if signal.size else 0.0
                if not math.isfinite(peak) or peak < 1e-8:
                    raise PipelineError("WAV silencioso ou praticamente vazio.")
                vector, diag = extract_recording(signal, sr)
                if vector.size != expected_dim:
                    raise PipelineError(
                        f"Vetor tem {vector.size} atributos; esperado {expected_dim}."
                    )
                conn.execute(
                    """INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        sample_id, row["participant_id"], row["task"],
                        vector.tobytes(), diag["speech_regions"], diag["windows"],
                        diag["active_duration_s"], diag["recording_duration_s"],
                        diag["speech_ratio"], diag["vad_fallback"], sr, channels,
                    ),
                )
                conn.execute("DELETE FROM errors WHERE sample_id=?", (sample_id,))
                conn.commit()
                processed_now += 1
            except Exception as exc:
                message = repr(exc)
                run_errors.append((sample_id, message))
                conn.execute(
                    "INSERT OR REPLACE INTO errors VALUES (?,?,?,?)",
                    (sample_id, row["participant_id"], row["task"], message),
                )
                conn.commit()
            if i % 10 == 0 or i == len(rows):
                done = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
                errn = conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
                print(
                    f"  {i}/{len(rows)} | concluidas={done} | erros={errn} "
                    f"| novas nesta execucao={processed_now}"
                )
    finally:
        for archive in archives.values():
            archive.close()

    cached_rows = conn.execute(
        """SELECT sample_id, participant_id, task, vector_blob, speech_regions,
                  windows, active_duration_s, recording_duration_s, speech_ratio,
                  vad_fallback, sample_rate_hz, channels_original
           FROM features"""
    ).fetchall()
    error_rows = conn.execute(
        "SELECT sample_id, participant_id, task, error FROM errors ORDER BY sample_id"
    ).fetchall()
    conn.close()

    by_id = {r[0]: r for r in cached_rows}
    ordered = [by_id[r["sample_id"]] for r in rows if r["sample_id"] in by_id]
    if not ordered:
        raise PipelineError("Nenhuma gravacao possui caracteristicas no cache.")

    matrix = np.vstack([
        np.frombuffer(r[3], dtype=np.float32) for r in ordered
    ]).astype(np.float32, copy=False)
    if matrix.shape[1] != expected_dim:
        raise PipelineError(f"Matriz final tem {matrix.shape[1]} colunas; esperado {expected_dim}.")
    if not np.isfinite(matrix).all():
        raise PipelineError("Matriz final contem valores nao finitos.")

    np.savez_compressed(
        OUTPUT_NPZ,
        wavelet=matrix,
        feature_names=np.asarray(feature_names, dtype=str),
        base_feature_names=np.asarray(base_names, dtype=str),
    )

    metadata_out = []
    diag_by_id = {r[0]: r for r in ordered}
    for row in rows:
        cached = diag_by_id.get(row["sample_id"])
        if cached is None:
            continue
        metadata_out.append({
            **row,
            "speech_regions": cached[4],
            "window_count": cached[5],
            "active_duration_s": round(float(cached[6]), 4),
            "recording_duration_s": round(float(cached[7]), 4),
            "speech_ratio": round(float(cached[8]), 5),
            "vad_fallback": cached[9],
            "sample_rate_hz": cached[10],
            "channels_original": cached[11],
        })
    with OUTPUT_META.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_out[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_out)

    if error_rows:
        with OUTPUT_ERRORS.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "participant_id", "task", "error"])
            writer.writerows(error_rows)
    elif OUTPUT_ERRORS.exists():
        OUTPUT_ERRORS.unlink()

    per_person = Counter(r[1] for r in ordered)
    windows = [int(r[5]) for r in ordered]
    regions = [int(r[4]) for r in ordered]
    active_durations = [float(r[6]) for r in ordered]
    speech_ratios = [float(r[8]) for r in ordered]
    format_counts = Counter((int(r[10]), int(r[11])) for r in ordered)
    task_counts = Counter(r[2] for r in ordered)
    label_counts = Counter(labels.values())
    all_missing_people = sorted(pid for pid in people if per_person[pid] < 6)
    fallback_count = sum(int(r[9]) for r in ordered)
    report = [
        "EXTRACAO WAVELET TEMPORAL AVFAD — CAPE-V 004-009",
        "=" * 72,
        f"Participantes HSA no manifesto: {len(people)}",
        f"Classes: Normal={label_counts['Normal']}; Pathological={label_counts['Pathological']}",
        f"Gravacoes no manifesto: {len(rows)}",
        f"Gravacoes processadas: {len(ordered)}",
        f"Participantes com as seis frases: {sum(per_person[p] == 6 for p in people)}/{len(people)}",
        f"Participantes com pelo menos uma frase: {len(per_person)}",
        f"Participantes com frase ausente: {len(all_missing_people)} ({', '.join(all_missing_people) if all_missing_people else 'nenhum'})",
        f"Erros registrados: {len(error_rows)}",
        "",
        "CONFIGURACAO WAVELET",
        f"Familias: {', '.join(WAVELETS)}",
        f"DWT: nivel {DWT_LEVEL}, A{DWT_LEVEL} + D{DWT_LEVEL}...D1",
        f"WPT: niveis {', '.join(map(str, WPT_LEVELS))}",
        f"Descritores por banda/no: {', '.join(DESCRIPTORS)}",
        f"Resumo temporal: {', '.join(SUMMARY_STATS)} das janelas",
        f"Dimensao por gravacao: {matrix.shape[1]} caracteristicas",
        "",
        "SEGMENTACAO DE FALA E JANELAMENTO",
        f"VAD: librosa.effects.split, top_db={VAD_TOP_DB}, frame={VAD_FRAME_LENGTH}, hop={VAD_HOP_LENGTH}",
        f"Pausas unidas ate: {VAD_MERGE_GAP_S:.2f}s; regiao minima: {VAD_MIN_REGION_S:.2f}s",
        f"Janelas: {WINDOW_S:.2f}s; passo: {WINDOW_HOP_S:.2f}s",
        f"Gravacoes com fallback do VAD: {fallback_count}",
        f"Regioes de fala por gravacao: min={min(regions)}; mediana={median(regions):.1f}; max={max(regions)}",
        f"Janelas por gravacao: min={min(windows)}; mediana={median(windows):.1f}; max={max(windows)}",
        f"Duracao ativa por gravacao (s): min={min(active_durations):.2f}; mediana={median(active_durations):.2f}; max={max(active_durations):.2f}",
        f"Proporcao de fala (ativa/duracao total): min={min(speech_ratios):.3f}; mediana={median(speech_ratios):.3f}; max={max(speech_ratios):.3f}",
        "",
        "ARQUIVOS POR TAREFA",
    ]
    for task in sorted(TASKS):
        report.append(f"  {task}: {task_counts[task]}")
    report.extend(["", "FORMATOS PROCESSADOS"])
    for (sr, channels), count in sorted(format_counts.items()):
        report.append(f"  {sr} Hz, {channels} canal(is): {count}")
    if error_rows:
        report.extend(["", "PRIMEIROS ERROS"])
        for sample_id, pid, task, error in error_rows[:25]:
            report.append(f"  {sample_id} ({pid}, tarefa {task}): {error}")
    report.extend([
        "",
        "ARQUIVOS GERADOS",
        f"  {OUTPUT_NPZ.name}",
        f"  {OUTPUT_META.name}",
        f"  {OUTPUT_REPORT.name}",
        f"  {CACHE_DB.name} (checkpoint retomavel)",
    ])
    if error_rows:
        report.append(f"  {OUTPUT_ERRORS.name}")
    report.append(f"Tempo desta execucao: {(time.time() - started) / 60:.1f} minutos")
    OUTPUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nExtracao wavelet temporal concluida.")
    print(f"Matriz: {matrix.shape} | gravacoes processadas: {len(ordered)}")
    print(f"Relatorio: {OUTPUT_REPORT}")
    print(f"Checkpoint: {CACHE_DB}")


if __name__ == "__main__":
    main()
