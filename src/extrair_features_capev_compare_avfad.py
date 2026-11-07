# -*- coding: utf-8 -*-
"""AVFAD CAPE-V (004-009): ComParE 2016 + MFCC/DWT/WPT.

Processa cada WAV completo, sem VAD/recorte fonetico, para manter o mesmo
escopo temporal da extracao por arquivo usada pelo baseline openSMILE dos autores.
Cada gravacao e uma amostra, mas participant_id e preservado: na modelagem,
todas as gravacoes de uma pessoa DEVEM ficar no mesmo fold.

Entrada:
  raiz do repositorio\\validacao_tarefa_002.csv
  ZIPs originais AVFAD em raiz do repositorio\\*.zip

Saidas finais:
  features_capev_compare_avfad.npz
  metadata_capev_avfad.csv
  resumo_capev_compare_avfad.txt
  erros_capev_compare_avfad.csv (se houver erros)

Checkpoint:
  cache_capev_compare.sqlite (retomavel; nao apagar durante a extracao)

Requer openSMILE Python 2.6.0 (64-bit), librosa, soundfile, PyWavelets,
scipy, numpy, openpyxl. Os ZIPs originais sao somente lidos.
"""

from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import csv
import io
import json
import math
from importlib.metadata import version as package_version
import platform
import random
import sqlite3
import statistics
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
import re

import librosa
import numpy as np
import pywt
import soundfile as sf
import opensmile
from openpyxl import load_workbook


ROOT = project_root()
INPUT_CSV = asset(ROOT, "validacao_tarefa_002.csv")
CACHE_DB = asset(ROOT, "cache_capev_compare.sqlite")
OUTPUT_NPZ = asset(ROOT, "features_capev_compare_avfad.npz")
OUTPUT_META = asset(ROOT, "metadata_capev_avfad.csv")
OUTPUT_REPORT = asset(ROOT, "resumo_capev_compare_avfad.txt")
OUTPUT_ERRORS = asset(ROOT, "erros_capev_compare_avfad.csv")

TASKS = ("004", "005", "006", "007", "008", "009")
EXPECTED_COMPARE_FEATURES = 6373
EXPECTED_SAMPLE_RATE = 48000
WAVELET = "db4"
WAVELET_LEVEL = 6
RANDOM_SEED = 42

# MFCC implementados sobre a gravacao completa, na mesma taxa nativa (normalmente 48 kHz).
MFCC_N = 25
MEL_N = 120
MFCC_WIN_MS = 25
MFCC_HOP_MS = 10
MFCC_NFFT = 2048
MFCC_FMIN = 50.0


class DatasetError(RuntimeError):
    pass


def load_metadata_csv():
    if not INPUT_CSV.exists():
        raise DatasetError(
            f"Nao encontrei {INPUT_CSV}. O arquivo validacao_tarefa_002.csv "
            "gerado na validacao inicial precisa estar na pasta AVFAD."
        )

    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        source_rows = list(csv.DictReader(f))

    if not source_rows:
        raise DatasetError("validacao_tarefa_002.csv esta vazio.")

    required = {"participant_id", "label", "age", "sex", "visit_place"}
    missing = required - set(source_rows[0].keys())
    if missing:
        raise DatasetError(f"Colunas ausentes no CSV de validacao: {sorted(missing)}")

    rows = {}
    for row in source_rows:
        pid = (row.get("participant_id") or "").strip().upper()
        if not pid:
            continue
        if (row.get("visit_place") or "").strip().upper() != "HSA":
            continue
        label = (row.get("label") or "").strip()
        if label not in {"Normal", "Pathological"}:
            continue
        if pid in rows:
            raise DatasetError(f"ID duplicado no CSV de validacao: {pid}")
        try:
            age = int(float(row["age"]))
        except (TypeError, ValueError):
            age = None
        sex = (row.get("sex") or "").strip().upper()
        rows[pid] = {
            "participant_id": pid,
            "label": label,
            "y": 0 if label == "Normal" else 1,
            "age": age,
            "sex": sex,
            "visit_place": "HSA",
        }

    if len(rows) != 411:
        raise DatasetError(
            f"Esperava 411 participantes HSA validados; encontrei {len(rows)}. "
            "Confira o CSV antes de continuar."
        )
    return rows


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
            "Esperava uma AVFAD_01_00_00.xlsx nos ZIPs; "
            f"encontrei {len(matches)}."
        )

    zip_path, internal_path = matches[0]
    with zipfile.ZipFile(zip_path, "r") as archive:
        workbook_data = io.BytesIO(archive.read(internal_path))

    wb = load_workbook(workbook_data, read_only=True, data_only=True)
    if "AVFAD" not in wb.sheetnames:
        wb.close()
        raise DatasetError("A aba AVFAD nao foi encontrada na planilha.")

    ws = wb["AVFAD"]
    iterator = ws.iter_rows(values_only=True)
    headers = next(iterator)
    id_col = headers.index("File ID")
    date_col = headers.index("Visit date")

    years = {}
    for row in iterator:
        raw_id = row[id_col]
        if raw_id is None:
            continue
        pid = str(raw_id).strip().upper()
        visit_date = row[date_col]
        if isinstance(visit_date, (datetime, date)):
            years[pid] = int(visit_date.year)
        else:
            years[pid] = None

    wb.close()
    return years


def age_band(age):
    return 18 + ((int(age) - 18) // 5) * 5


def make_matched_sets(metadata, years):
    """Reproduz os pareamentos anteriores: seed 42, sem reposicao."""
    buckets_age = defaultdict(lambda: {"Normal": [], "Pathological": []})
    buckets_year = defaultdict(lambda: {"Normal": [], "Pathological": []})

    for pid, row in metadata.items():
        age = row["age"]
        sex = row["sex"]
        year = years.get(pid)
        if age is None or sex not in {"F", "M"}:
            continue
        key_age = (sex, age_band(age))
        buckets_age[key_age][row["label"]].append(pid)
        if year is not None:
            key_year = (sex, age_band(age), year)
            buckets_year[key_year][row["label"]].append(pid)

    rng = random.Random(RANDOM_SEED)
    selected_age = set()
    selected_year = set()
    pairs_age = 0
    pairs_year = 0

    for key in sorted(buckets_age, key=lambda x: tuple(str(v) for v in x)):
        normals = sorted(buckets_age[key]["Normal"])
        patients = sorted(buckets_age[key]["Pathological"])
        n = min(len(normals), len(patients))
        if n:
            selected_age.update(rng.sample(normals, n))
            selected_age.update(rng.sample(patients, n))
            pairs_age += n

    # Reinicia RNG para que o pareamento com ano seja reproduzivel e independente.
    rng = random.Random(RANDOM_SEED)
    for key in sorted(buckets_year, key=lambda x: tuple(str(v) for v in x)):
        normals = sorted(buckets_year[key]["Normal"])
        patients = sorted(buckets_year[key]["Pathological"])
        n = min(len(normals), len(patients))
        if n:
            selected_year.update(rng.sample(normals, n))
            selected_year.update(rng.sample(patients, n))
            pairs_year += n

    return selected_age, pairs_age, selected_year, pairs_year


def index_audio_files(zip_paths, participant_ids):
    pattern = re.compile(r"^([A-Z]{3})(00[4-9])\.WAV$", re.IGNORECASE)
    index = defaultdict(list)
    indexed_total = 0

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                basename = info.filename.replace("\\", "/").split("/")[-1]
                match = pattern.fullmatch(basename.upper())
                if not match:
                    continue
                pid = match.group(1).upper()
                task = match.group(2)
                if pid not in participant_ids:
                    continue
                index[(pid, task)].append((zip_path.name, info.filename))
                indexed_total += 1

    duplicates = [(key, paths) for key, paths in index.items() if len(paths) > 1]
    if duplicates:
        examples = "; ".join(f"{k}: {v}" for k, v in duplicates[:10])
        raise DatasetError(f"Arquivos duplicados para participante/tarefa: {examples}")

    return index, indexed_total


def read_wav_from_zip(archive, internal_path):
    raw = archive.read(internal_path)
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if audio.ndim != 2 or audio.shape[0] < 1:
        raise DatasetError("WAV vazio ou com formato inesperado.")
    if not np.all(np.isfinite(audio)):
        raise DatasetError("Audio contem NaN ou infinito.")

    channels = int(audio.shape[1])
    if channels == 1:
        mono = audio[:, 0]
    else:
        # Downmix deterministico para mono. Os arquivos comuns ja sao mono.
        mono = np.mean(audio, axis=1, dtype=np.float64).astype(np.float32)

    peak = float(np.max(np.abs(mono)))
    if not math.isfinite(peak) or peak < 1e-8:
        raise DatasetError("Audio vazio ou praticamente silencioso.")
    duration = len(mono) / float(sr)
    if duration < 0.20:
        raise DatasetError(f"Audio curto demais para ComParE ({duration:.3f}s).")

    return mono, int(sr), channels, len(audio), duration, len(raw)


def compare_features(smile, signal, sr):
    frame = smile.process_signal(signal, sr)
    if frame is None or frame.empty:
        raise DatasetError("openSMILE nao retornou caracteristicas.")
    values = frame.iloc[0].to_numpy(dtype=np.float32, copy=True)
    names = list(frame.columns)
    if values.size != EXPECTED_COMPARE_FEATURES:
        raise DatasetError(
            f"ComParE retornou {values.size} caracteristicas; esperadas "
            f"{EXPECTED_COMPARE_FEATURES}. Confira a versao do openSMILE."
        )
    if not np.isfinite(values).any():
        raise DatasetError("Vetor ComParE sem nenhum valor finito.")
    return values, names


def normalized_entropy(coefficients):
    values = np.asarray(coefficients, dtype=np.float64)
    power = values * values
    total = float(np.sum(power))
    if not math.isfinite(total) or total <= 1e-20 or power.size <= 1:
        return 0.0
    probabilities = power / total
    probabilities = probabilities[probabilities > 0]
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    denom = math.log2(power.size)
    return entropy / denom if denom > 0 else 0.0


def own_acoustic_features(signal, sr):
    """MFCC (150) + DWT (14) + WPT (128) + sinal (3) sobre WAV completo."""
    y = np.asarray(signal, dtype=np.float32).reshape(-1)
    y = y - np.mean(y, dtype=np.float64)
    win_length = max(64, int(round(MFCC_WIN_MS / 1000 * sr)))
    hop_length = max(32, int(round(MFCC_HOP_MS / 1000 * sr)))
    n_fft = max(MFCC_NFFT, 2 ** int(math.ceil(math.log2(win_length))))
    n_mels = min(MEL_N, max(20, n_fft // 4))
    fmax = sr / 2.0

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hamming",
        n_mels=n_mels,
        fmin=min(MFCC_FMIN, fmax * 0.5),
        fmax=fmax,
        power=2.0,
        norm="slaney",
        htk=False,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=MFCC_N)

    n_frames = mfcc.shape[1]
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    if width < 3:
        raise DatasetError(f"Poucos frames ({n_frames}) para deltas MFCC.")
    delta = librosa.feature.delta(mfcc, order=1, width=width, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=width, mode="nearest")

    result = {}
    for prefix, matrix in (("mfcc", mfcc), ("delta", delta), ("delta2", delta2)):
        means = np.mean(matrix, axis=1)
        stds = np.std(matrix, axis=1)
        for i, value in enumerate(means):
            result[f"{prefix}_{i:02d}_mean"] = float(value)
        for i, value in enumerate(stds):
            result[f"{prefix}_{i:02d}_std"] = float(value)

    wavelet = pywt.Wavelet(WAVELET)
    max_level = pywt.dwt_max_level(len(y), wavelet.dec_len)
    level = min(WAVELET_LEVEL, max_level)
    if level < 1:
        raise DatasetError("Audio curto demais para DWT/WPT.")

    coeffs = pywt.wavedec(y, wavelet=wavelet, mode="symmetric", level=level)
    energies = [float(np.sum(c.astype(np.float64) ** 2)) for c in coeffs]
    total_energy = sum(energies) + 1e-20
    bands = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    for band, coeff, energy in zip(bands, coeffs, energies):
        result[f"dwt_{band}_rel_energy"] = energy / total_energy
        result[f"dwt_{band}_entropy"] = normalized_entropy(coeff)

    packet = pywt.WaveletPacket(
        data=y, wavelet=WAVELET, mode="symmetric", maxlevel=WAVELET_LEVEL
    )
    nodes = packet.get_level(WAVELET_LEVEL, order="freq")
    wpt_energies = [float(np.sum(node.data.astype(np.float64) ** 2)) for node in nodes]
    wpt_total = sum(wpt_energies) + 1e-20
    for i, (node, energy) in enumerate(zip(nodes, wpt_energies)):
        result[f"wpt_{i:02d}_rel_energy"] = energy / wpt_total
        result[f"wpt_{i:02d}_entropy"] = normalized_entropy(node.data)

    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-20))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(
        y, frame_length=win_length, hop_length=hop_length
    )))
    result["signal_rms"] = rms
    result["signal_rms_db"] = 20.0 * math.log10(rms + 1e-12)
    result["signal_zcr"] = zcr

    expected = 150 + 14 + 128 + 3
    if len(result) != expected:
        raise DatasetError(f"Vetor MFCC/DWT/WPT tem {len(result)} valores; esperado {expected}.")
    names = list(result.keys())
    values = np.asarray([result[name] for name in names], dtype=np.float32)
    return values, names


def init_cache(db_path, signature):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS config (
               key TEXT PRIMARY KEY,
               value TEXT NOT NULL
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS features (
               participant_id TEXT NOT NULL,
               task TEXT NOT NULL,
               label TEXT NOT NULL,
               y INTEGER NOT NULL,
               age INTEGER,
               sex TEXT,
               visit_year INTEGER,
               visit_place TEXT,
               zip_name TEXT NOT NULL,
               internal_path TEXT NOT NULL,
               duration_s REAL NOT NULL,
               sample_rate_hz INTEGER NOT NULL,
               channels INTEGER NOT NULL,
               samples INTEGER NOT NULL,
               compare_blob BLOB NOT NULL,
               acoustic_blob BLOB NOT NULL,
               PRIMARY KEY (participant_id, task)
           )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS errors (
               participant_id TEXT NOT NULL,
               task TEXT NOT NULL,
               file_path TEXT,
               error TEXT NOT NULL,
               PRIMARY KEY (participant_id, task)
           )"""
    )
    existing = dict(conn.execute("SELECT key, value FROM config").fetchall())
    if existing and existing.get("signature") != signature:
        conn.close()
        raise DatasetError(
            "O cache existe com parametros/versao diferentes. Para evitar misturar "
            "extracoes, renomeie ou remova cache_capev_compare.sqlite e seus arquivos "
            "-wal/-shm somente se tiver certeza de que deseja recomecar."
        )
    if not existing:
        conn.execute(
            "INSERT INTO config(key, value) VALUES('signature', ?)",
            (signature,)
        )
        conn.commit()
    return conn


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    if platform.architecture()[0] != "64bit":
        raise DatasetError("openSMILE Python requer Python de 64 bits.")

    try:
        version = package_version("opensmile")
    except Exception:
        version = "desconhecida"
    if version != "2.6.0":
        raise DatasetError(
            f"Versao openSMILE detectada: {version}. Para reproduzir a configuracao, "
            "instale opensmile==2.6.0 no .venv."
        )

    zip_paths = sorted(raw_dir(ROOT).glob("AVFAD_01_00_00_*.zip"))
    if not zip_paths:
        raise DatasetError(f"Nenhum ZIP encontrado em {ROOT}.")

    print("Carregando metadados e pareamentos...")
    metadata = load_metadata_csv()
    years = load_visit_years()
    matched_age, pairs_age, matched_year, pairs_year = make_matched_sets(metadata, years)

    print("Indexando gravacoes CAPE-V 004-009 nos ZIPs...")
    audio_index, indexed_total = index_audio_files(zip_paths, set(metadata))
    expected_keys = [(pid, task) for pid in sorted(metadata) for task in TASKS]
    missing_keys = [key for key in expected_keys if key not in audio_index]

    if missing_keys:
        print(f"ATENCAO: {len(missing_keys)} gravacoes nao encontradas.")

    expected_extracted = len(expected_keys) - len(missing_keys)
    print(f"Participantes HSA: {len(metadata)} | WAVs esperados: {len(expected_keys)}")
    print(f"WAVs encontrados: {expected_extracted} | ausentes: {len(missing_keys)}")
    print("Configurando ComParE 2016 Functionals (6373 features)...")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
        loglevel=0,
    )
    if len(smile.feature_names) != EXPECTED_COMPARE_FEATURES:
        raise DatasetError(
            f"openSMILE configurou {len(smile.feature_names)} features; "
            f"esperadas {EXPECTED_COMPARE_FEATURES}."
        )

    signature = json.dumps({
        "opensmile_version": version,
        "feature_set": "ComParE_2016",
        "feature_level": "Functionals",
        "expected_compare_features": EXPECTED_COMPARE_FEATURES,
        "tasks": TASKS,
        "sampling_policy": "original_rate; downmix_to_mono_if_multichannel",
        "own_features": "MFCC25+delta+delta2;DWT-db4-L6;WPT-db4-L6;signal3;whole_recording",
        "seed": RANDOM_SEED,
    }, sort_keys=True)

    conn = init_cache(CACHE_DB, signature)
    # Registra WAVs ausentes no ZIP para que aparecam explicitamente no relatorio.
    for pid, task in missing_keys:
        conn.execute(
            "INSERT OR REPLACE INTO errors(participant_id, task, file_path, error) VALUES (?, ?, ?, ?)",
            (pid, task, "AUSENTE_NO_ZIP", "Arquivo WAV nao encontrado nos ZIPs.")
        )
    conn.commit()
    archives = {}
    errors = []
    formats = Counter()
    downmixed = 0
    started = time.time()
    compare_names = list(smile.feature_names)
    acoustic_names = None

    try:
        completed = {
            (pid, task)
            for pid, task in conn.execute("SELECT participant_id, task FROM features")
        }
        print(f"Checkpoint encontrado: {len(completed)} gravacoes ja processadas.")

        for idx, key in enumerate(expected_keys, start=1):
            pid, task = key
            if key not in audio_index:
                errors.append({
                    "participant_id": pid,
                    "task": task,
                    "file_path": "AUSENTE_NO_ZIP",
                    "error": "Arquivo WAV nao encontrado nos ZIPs."
                })
                continue
            if key in completed:
                continue

            zip_name, internal_path = audio_index[key][0]
            try:
                if zip_name not in archives:
                    archives[zip_name] = zipfile.ZipFile(asset(ROOT, zip_name), "r")
                signal, sr, channels, samples, duration, byte_count = read_wav_from_zip(
                    archives[zip_name], internal_path
                )
                formats[(sr, channels)] += 1
                if channels > 1:
                    downmixed += 1

                compare_vector, current_compare_names = compare_features(smile, signal, sr)
                if current_compare_names != compare_names:
                    raise DatasetError("Nomes/ordem das features ComParE variaram entre arquivos.")
                acoustic_vector, current_acoustic_names = own_acoustic_features(signal, sr)
                if acoustic_names is None:
                    acoustic_names = current_acoustic_names
                elif current_acoustic_names != acoustic_names:
                    raise DatasetError("Nomes/ordem das features MFCC/DWT/WPT variaram.")

                row = metadata[pid]
                conn.execute(
                    """INSERT OR REPLACE INTO features (
                        participant_id, task, label, y, age, sex, visit_year,
                        visit_place, zip_name, internal_path, duration_s,
                        sample_rate_hz, channels, samples, compare_blob, acoustic_blob
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pid, task, row["label"], row["y"], row["age"], row["sex"],
                        years.get(pid), row["visit_place"], zip_name, internal_path,
                        duration, sr, channels, samples,
                        compare_vector.astype(np.float32, copy=False).tobytes(),
                        acoustic_vector.astype(np.float32, copy=False).tobytes(),
                    )
                )
                conn.execute("DELETE FROM errors WHERE participant_id=? AND task=?", (pid, task))
                conn.commit()  # checkpoint por gravacao: pode retomar se interromper.

            except Exception as exc:
                message = repr(exc)
                errors.append({
                    "participant_id": pid,
                    "task": task,
                    "file_path": internal_path,
                    "error": message,
                })
                conn.execute(
                    "INSERT OR REPLACE INTO errors(participant_id, task, file_path, error) VALUES (?, ?, ?, ?)",
                    (pid, task, internal_path, message)
                )
                conn.commit()

            if idx % 30 == 0 or idx == len(expected_keys):
                done_count = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
                print(f"  {idx}/{len(expected_keys)} | gravacoes concluidas: {done_count} | erros nesta execucao: {len(errors)}")

    finally:
        for archive in archives.values():
            archive.close()
        conn.commit()

    rows_db = conn.execute(
        """SELECT participant_id, task, label, y, age, sex, visit_year,
                  visit_place, zip_name, internal_path, duration_s,
                  sample_rate_hz, channels, samples, compare_blob, acoustic_blob
           FROM features ORDER BY participant_id, task"""
    ).fetchall()
    db_errors = conn.execute(
        "SELECT participant_id, task, file_path, error FROM errors ORDER BY participant_id, task"
    ).fetchall()
    conn.close()

    if not rows_db:
        raise DatasetError("Nenhuma gravacao foi processada com sucesso.")

    compare_matrix = np.vstack([
        np.frombuffer(row[14], dtype=np.float32) for row in rows_db
    ])
    acoustic_matrix = np.vstack([
        np.frombuffer(row[15], dtype=np.float32) for row in rows_db
    ])

    meta_rows = []
    for i, row in enumerate(rows_db):
        (
            pid, task, label, y, age, sex, year, place, zip_name,
            internal_path, duration, sr, channels, samples, _, _
        ) = row
        meta_rows.append({
            "row_index": i,
            "sample_id": f"{pid}{task}",
            "participant_id": pid,
            "task": task,
            "label": label,
            "y": y,
            "age": age,
            "sex": sex,
            "visit_year": year,
            "visit_place": place,
            "matched_age_sex": int(pid in matched_age),
            "matched_age_sex_year": int(pid in matched_year),
            "duration_s": round(float(duration), 6),
            "sample_rate_hz": int(sr),
            "channels_original": int(channels),
            "samples": int(samples),
            "zip_name": zip_name,
            "audio_path_in_zip": internal_path,
        })

    # Saida atomica: evita deixar NPZ final parcialmente escrito.
    tmp_npz = OUTPUT_NPZ.with_name(OUTPUT_NPZ.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_npz,
        compare=compare_matrix,
        acoustic=acoustic_matrix,
        compare_names=np.asarray(compare_names, dtype=str),
        acoustic_names=np.asarray(acoustic_names or [], dtype=str),
    )
    tmp_npz.replace(OUTPUT_NPZ)

    meta_fields = list(meta_rows[0].keys())
    write_csv(OUTPUT_META, meta_rows, meta_fields)

    if db_errors:
        error_rows = [
            {"participant_id": r[0], "task": r[1], "file_path": r[2], "error": r[3]}
            for r in db_errors
        ]
        write_csv(
            OUTPUT_ERRORS,
            error_rows,
            ["participant_id", "task", "file_path", "error"]
        )
    elif OUTPUT_ERRORS.exists():
        OUTPUT_ERRORS.unlink()

    per_participant = Counter(row[0] for row in rows_db)
    complete_participants = sum(per_participant[pid] == len(TASKS) for pid in metadata)
    labels_by_person = Counter(r["label"] for r in metadata.values())
    task_counts = Counter(row[1] for row in rows_db)
    duration_values = [float(row[10]) for row in rows_db]
    compare_nan = int(np.isnan(compare_matrix).sum())
    compare_inf = int(np.isinf(compare_matrix).sum())
    acoustic_nan = int(np.isnan(acoustic_matrix).sum())
    acoustic_inf = int(np.isinf(acoustic_matrix).sum())
    elapsed = time.time() - started

    report = [
        "EXTRACAO AVFAD CAPE-V 004-009 — ComParE 2016 + MFCC/DWT/WPT",
        "=" * 76,
        f"Participantes HSA de entrada: {len(metadata)}",
        f"Classes por participante: Normal={labels_by_person['Normal']}; Pathological={labels_by_person['Pathological']}",
        f"Gravacoes esperadas: {len(metadata) * len(TASKS)}",
        f"Arquivos localizados nos ZIPs: {indexed_total}",
        f"Gravacoes processadas com sucesso: {len(rows_db)}",
        f"Participantes com as seis frases processadas: {complete_participants}/{len(metadata)}",
        f"Participantes com pelo menos uma frase processada: {len(per_participant)}",
        f"Erros/ausencias registrados: {len(db_errors)}",
        f"Coorte pareada sexo+idade: {len(matched_age)} participantes ({pairs_age} pares)",
        f"Coorte pareada sexo+idade+ano: {len(matched_year)} participantes ({pairs_year} pares)",
        "",
        "CONFIGURACAO ComParE (baseline dos autores)",
        f"openSMILE Python: {version}",
        "Feature set: ComParE_2016",
        "Feature level: Functionals",
        f"Dimensao ComParE: {compare_matrix.shape[1]} (esperado {EXPECTED_COMPARE_FEATURES})",
        "Escopo temporal: WAV completo; sem VAD, sem corte de silencio e sem recorte de fonema.",
        "Canais: mono preservado; arquivos multicanal convertidos para mono pela media dos canais.",
        "Taxa de amostragem: taxa nativa preservada; nao foi feito resampling.",
        f"Arquivos multicanal convertidos para mono: {sum(1 for r in rows_db if int(r[12]) > 1)}",
        "",
        "CARACTERISTICAS PROPRIAS (mesmo WAV completo)",
        "MFCC: 25 coeficientes + delta + delta-delta; media e desvio por gravacao (150).",
        "DWT: db4 nivel 6 quando permitido pelo comprimento (energia relativa + entropia por banda; 14).",
        "WPT: db4 nivel 6 (energia relativa + entropia por sub-banda; 128).",
        "Sinal: RMS, RMS em dB e ZCR (3).",
        f"Dimensao MFCC+DWT+WPT+sinal: {acoustic_matrix.shape[1]} (esperado 295)",
        "",
        "DURACAO DAS GRAVACOES PROCESSADAS",
        f"Minima={min(duration_values):.3f}s | mediana={statistics.median(duration_values):.3f}s | maxima={max(duration_values):.3f}s",
        "",
        "ARQUIVOS PROCESSADOS POR TAREFA",
    ]
    for task in TASKS:
        report.append(f"  {task}: {task_counts[task]}")

    report.extend([
        "",
        "QUALIDADE NUMERICA (valores nao finitos sao mantidos para imputacao dentro dos folds)",
        f"ComParE: NaN={compare_nan}; infinito={compare_inf}",
        f"MFCC/DWT/WPT: NaN={acoustic_nan}; infinito={acoustic_inf}",
        "",
        "FORMATOS ORIGINAIS",
    ])
    format_counts = Counter((int(r[11]), int(r[12])) for r in rows_db)
    for (sr, channels), count in sorted(format_counts.items()):
        report.append(f"  {sr} Hz, {channels} canal(is): {count} gravacoes")

    if db_errors:
        report.extend(["", "PRIMEIROS ERROS/AUSENCIAS"])
        for row in db_errors[:30]:
            report.append(f"  {row[0]} tarefa {row[1]}: {row[3]}")

    report.extend([
        "",
        "ARQUIVOS GERADOS",
        f"  {OUTPUT_NPZ.name}",
        f"  {OUTPUT_META.name}",
        f"  {OUTPUT_REPORT.name}",
        f"  {CACHE_DB.name} (checkpoint para retomada)",
    ])
    if db_errors:
        report.append(f"  {OUTPUT_ERRORS.name}")
    report.append(f"Tempo desta execucao: {elapsed / 60:.1f} minutos")
    OUTPUT_REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nExtracao concluida.")
    print(f"Gravacoes: {len(rows_db)} | participantes completos: {complete_participants}/{len(metadata)}")
    print(f"ComParE: {compare_matrix.shape} | acusticas: {acoustic_matrix.shape}")
    print(f"Relatorio: {OUTPUT_REPORT}")
    print(f"Checkpoint preservado: {CACHE_DB}")
    if db_errors:
        print(f"ATENCAO: existem {len(db_errors)} erros/ausencias. Consulte {OUTPUT_ERRORS.name}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuario. O checkpoint SQLite foi preservado; execute novamente para retomar.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nERRO: {exc}")
        sys.exit(1)
