# -*- coding: utf-8 -*-
"""AVFAD /a/: extrai características do intervalo /a/ marcado no PRT.

Usa o Sound e o TextGrid armazenados juntos no arquivo PRT. Portanto, as
características são calculadas exatamente no intervalo anotado, sem VAD por
energia. Compara também o Sound embutido com o WAV original como controle.

Entrada: raiz do repositorio\\validacao_tarefa_002.csv e ZIPs AVFAD.
Saídas:
  features_avfad_prt_hsa_002.csv
  resumo_extracao_prt_avfad.txt
  erros_extracao_prt_avfad.csv (somente se houver erros)

Não modifica nem extrai os ZIPs originais.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import csv
import io
import math
import random
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import numpy as np
import pywt
import soundfile as sf
from openpyxl import load_workbook

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError as exc:
    raise SystemExit(
        "Biblioteca ausente. Ative o .venv e execute:\n"
        "python -m pip install praat-parselmouth"
    ) from exc

ROOT = project_root()
INPUT_CSV = asset(ROOT, "validacao_tarefa_002.csv")
OUTPUT_CSV = asset(ROOT, "features_avfad_prt_hsa_002.csv")
REPORT_TXT = asset(ROOT, "resumo_extracao_prt_avfad.txt")
ERROR_CSV = asset(ROOT, "erros_extracao_prt_avfad.csv")

EXPECTED_SR = 48000
WAVELET = "db4"
WAVELET_LEVEL = 6
RANDOM_SEED = 42

FRAME_LENGTH = int(0.025 * EXPECTED_SR)
HOP_LENGTH = int(0.010 * EXPECTED_SR)
N_FFT = 2048
N_MELS = 120
N_MFCC = 25
FMIN = 50.0
FMAX = EXPECTED_SR / 2


class DatasetError(RuntimeError):
    pass


def load_validated_rows():
    if not INPUT_CSV.exists():
        raise DatasetError(f"Não encontrei {INPUT_CSV}.")
    with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {
        "participant_id", "label", "age", "sex", "visit_place",
        "audio_archive", "audio_path", "duration_s"
    }
    if not rows:
        raise DatasetError("O CSV de validação está vazio.")
    missing = required - set(rows[0])
    if missing:
        raise DatasetError(f"Colunas ausentes no CSV: {sorted(missing)}")

    hsa = []
    seen = set()
    for row in rows:
        if row["visit_place"].strip().upper() != "HSA":
            continue
        if row["label"] not in {"Normal", "Pathological"}:
            continue
        pid = row["participant_id"].strip().upper()
        if pid in seen:
            raise DatasetError(f"ID duplicado: {pid}")
        seen.add(pid)
        row["participant_id"] = pid
        row["age"] = int(float(row["age"]))
        row["sex"] = row["sex"].strip().upper()
        row["duration_s"] = float(row["duration_s"])
        hsa.append(row)
    return hsa


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
        data = io.BytesIO(archive.read(internal_path))
    wb = load_workbook(data, read_only=True, data_only=True)
    ws = wb["AVFAD"]
    it = ws.iter_rows(values_only=True)
    headers = next(it)
    idx_id = headers.index("File ID")
    idx_date = headers.index("Visit date")
    years = {}
    for row in it:
        if row[idx_id] is None:
            continue
        pid = str(row[idx_id]).strip().upper()
        years[pid] = row[idx_date].year if row[idx_date] is not None else None
    wb.close()
    return years


def age_band(age: int) -> int:
    return 18 + ((age - 18) // 5) * 5


def matched_ids(rows, years, include_year=False):
    buckets = defaultdict(lambda: {"Normal": [], "Pathological": []})
    for row in rows:
        pid = row["participant_id"]
        year = years.get(pid)
        if row["sex"] not in {"F", "M"} or row["age"] is None:
            continue
        if include_year and year is None:
            continue
        key = (row["sex"], age_band(int(row["age"])))
        if include_year:
            key += (year,)
        buckets[key][row["label"]].append(pid)

    rng = random.Random(RANDOM_SEED)
    selected = set()
    pair_count = 0
    for key in sorted(buckets, key=lambda x: tuple(str(v) for v in x)):
        normals = sorted(buckets[key]["Normal"])
        patients = sorted(buckets[key]["Pathological"])
        n = min(len(normals), len(patients))
        if n:
            selected.update(rng.sample(normals, n))
            selected.update(rng.sample(patients, n))
            pair_count += n
    return selected, pair_count


def read_prt_objects(prt_bytes: bytes, filename: str):
    with tempfile.TemporaryDirectory(prefix="avfad_prt_feature_") as folder:
        path = Path(folder) / filename
        path.write_bytes(prt_bytes)
        objects = call("Read from file", str(path))
        return objects if isinstance(objects, list) else [objects]


def get_annotated_segment(objects, participant_id):
    sounds = [obj for obj in objects if isinstance(obj, parselmouth.Sound)]
    grids = [obj for obj in objects if isinstance(obj, parselmouth.TextGrid)]
    if len(sounds) != 1 or len(grids) != 1:
        raise DatasetError(
            f"{participant_id}: esperado 1 Sound + 1 TextGrid; "
            f"encontrados {len(sounds)} + {len(grids)}"
        )

    sound, grid = sounds[0], grids[0]
    sr = int(round(float(sound.sampling_frequency)))
    if sr != EXPECTED_SR:
        raise DatasetError(f"{participant_id}: PRT em {sr} Hz, esperado {EXPECTED_SR} Hz")
    if sound.n_channels < 1 or sound.values.size == 0:
        raise DatasetError(f"{participant_id}: Sound vazio ou inválido")

    annotations = []
    n_tiers = int(call(grid, "Get number of tiers"))
    for tier_no in range(1, n_tiers + 1):
        try:
            n_intervals = int(call(grid, "Get number of intervals", tier_no))
        except Exception:
            continue
        for interval_no in range(1, n_intervals + 1):
            label = str(call(grid, "Get label of interval", tier_no, interval_no)).strip()
            if label.casefold() in {"a", "/a/", "á"}:
                start = float(call(grid, "Get start time of interval", tier_no, interval_no))
                end = float(call(grid, "Get end time of interval", tier_no, interval_no))
                annotations.append((start, end, label))

    if len(annotations) != 1:
        raise DatasetError(
            f"{participant_id}: esperado exatamente um intervalo rotulado /a/; "
            f"encontrados {len(annotations)}"
        )

    start, end, label = annotations[0]
    sound_xmin = float(sound.xmin)
    sound_xmax = float(sound.xmax)
    if end <= start or start < sound_xmin - 1e-4 or end > sound_xmax + 1e-4:
        raise DatasetError(
            f"{participant_id}: intervalo /a/ fora dos limites do Sound "
            f"({start:.4f}-{end:.4f}s; Sound {sound_xmin:.4f}-{sound_xmax:.4f}s)"
        )

    # Em Praat, xmin corresponde ao limite externo do primeiro intervalo de amostras.
    values = np.asarray(sound.values, dtype=np.float64)
    mono = np.mean(values, axis=0) if values.shape[0] > 1 else values[0]
    start_idx = int(round((start - sound_xmin) * sr))
    end_idx = int(round((end - sound_xmin) * sr))
    start_idx = max(0, min(start_idx, len(mono)))
    end_idx = max(0, min(end_idx, len(mono)))
    segment = np.asarray(mono[start_idx:end_idx], dtype=np.float64).copy()
    if segment.size < int(0.20 * sr):
        raise DatasetError(
            f"{participant_id}: intervalo /a/ curto demais ({len(segment)/sr:.4f}s)"
        )
    if not np.all(np.isfinite(segment)):
        raise DatasetError(f"{participant_id}: segmento /a/ contém NaN ou infinito")

    # Remove somente o offset DC; não normaliza o volume.
    segment -= np.mean(segment)
    return segment, sr, start, end, label, sound


def compare_embedded_sound(sound, wav_bytes):
    wav, wav_sr = sf.read(io.BytesIO(wav_bytes), dtype="float64", always_2d=True)
    p = np.asarray(sound.values, dtype=np.float64).T
    prt_sr = int(round(float(sound.sampling_frequency)))
    if wav_sr != prt_sr or wav.shape != p.shape:
        return {
            "same_shape_rate": False,
            "waveform_match": False,
            "max_abs_diff": None,
        }
    max_diff = float(np.max(np.abs(wav - p))) if wav.size else 0.0
    return {
        "same_shape_rate": True,
        "waveform_match": bool(np.allclose(wav, p, rtol=0.0, atol=1e-5)),
        "max_abs_diff": max_diff,
    }


def normalized_entropy(values):
    values = np.asarray(values, dtype=np.float64)
    power = values * values
    total = float(np.sum(power))
    if total <= 1e-20 or power.size <= 1:
        return 0.0
    p = power / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log2(p)))
    return entropy / math.log2(power.size)


def mfcc_features(segment, sr):
    mel = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_fft=N_FFT, win_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH, window="hamming", n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, power=2.0, norm="slaney", htk=False,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0, top_db=None)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc, order=1, width=9, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=9, mode="nearest")
    result = {}
    for prefix, matrix in (("mfcc", mfcc), ("delta", delta), ("delta2", delta2)):
        means = np.mean(matrix, axis=1)
        stds = np.std(matrix, axis=1)
        for i, value in enumerate(means):
            result[f"{prefix}_{i:02d}_mean"] = float(value)
        for i, value in enumerate(stds):
            result[f"{prefix}_{i:02d}_std"] = float(value)
    return result


def dwt_features(segment):
    wavelet = pywt.Wavelet(WAVELET)
    level = min(WAVELET_LEVEL, pywt.dwt_max_level(len(segment), wavelet.dec_len))
    if level < 1:
        raise DatasetError("Segmento curto demais para DWT")
    coeffs = pywt.wavedec(segment, wavelet=WAVELET, mode="symmetric", level=level)
    energies = [float(np.sum(c * c)) for c in coeffs]
    total = sum(energies) + 1e-20
    labels = [f"A{level}"] + [f"D{i}" for i in range(level, 0, -1)]
    result = {}
    for band, coeff, energy in zip(labels, coeffs, energies):
        result[f"dwt_{band}_rel_energy"] = energy / total
        result[f"dwt_{band}_entropy"] = normalized_entropy(coeff)
    return result


def wpt_features(segment):
    wavelet = pywt.Wavelet(WAVELET)
    level = min(WAVELET_LEVEL, pywt.dwt_max_level(len(segment), wavelet.dec_len))
    if level < 1:
        raise DatasetError("Segmento curto demais para WPT")
    packet = pywt.WaveletPacket(
        data=segment, wavelet=WAVELET, mode="symmetric", maxlevel=level
    )
    nodes = packet.get_level(level, order="freq")
    energies = [float(np.sum(node.data * node.data)) for node in nodes]
    total = sum(energies) + 1e-20
    result = {}
    for i, (node, energy) in enumerate(zip(nodes, energies)):
        result[f"wpt_{i:02d}_rel_energy"] = energy / total
        result[f"wpt_{i:02d}_entropy"] = normalized_entropy(node.data)
    return result


def extract_features(segment, sr):
    result = {}
    result.update(mfcc_features(segment, sr))
    result.update(dwt_features(segment))
    result.update(wpt_features(segment))
    rms = float(np.sqrt(np.mean(segment * segment) + 1e-20))
    result["signal_rms"] = rms
    result["signal_rms_db"] = 20.0 * math.log10(rms + 1e-12)
    result["signal_zcr"] = float(np.mean(librosa.feature.zero_crossing_rate(
        segment, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )))
    if not all(np.isfinite(v) for v in result.values()):
        raise DatasetError("Características extraídas contêm NaN ou infinito")
    return result


def main():
    rows = load_validated_rows()
    years = load_visit_years()
    age_sex_ids, age_sex_pairs = matched_ids(rows, years, include_year=False)
    age_sex_year_ids, age_sex_year_pairs = matched_ids(rows, years, include_year=True)

    archives = {}
    member_indexes = {}
    output_rows = []
    errors = []
    durations = []
    audio_shape_match = Counter()
    waveform_match = Counter()
    expected_samples = 0

    try:
        for index, row in enumerate(rows, start=1):
            pid = row["participant_id"]
            archive_name = row["audio_archive"]
            try:
                if archive_name not in archives:
                    zip_path = asset(ROOT, archive_name)
                    if not zip_path.exists():
                        raise DatasetError(f"ZIP não encontrado: {zip_path}")
                    archives[archive_name] = zipfile.ZipFile(zip_path, "r")
                    member_indexes[archive_name] = {
                        info.filename.replace("\\", "/").split("/")[-1].upper(): info.filename
                        for info in archives[archive_name].infolist()
                        if not info.is_dir()
                    }

                archive = archives[archive_name]
                members = member_indexes[archive_name]
                prt_path = members.get(f"{pid}002.PRT")
                if not prt_path:
                    raise DatasetError("PRT 002 não encontrado no mesmo ZIP")
                wav_path = row["audio_path"]
                if wav_path not in archive.NameToInfo:
                    # Fallback pelo nome-base, mantendo a busca dentro do ZIP correto.
                    wav_path = members.get(f"{pid}002.WAV")
                if not wav_path:
                    raise DatasetError("WAV 002 não encontrado no ZIP")

                prt_bytes = archive.read(prt_path)
                wav_bytes = archive.read(wav_path)
                objects = read_prt_objects(prt_bytes, f"{pid}002.PRT")
                segment, sr, ann_start, ann_end, ann_label, sound = get_annotated_segment(objects, pid)
                match = compare_embedded_sound(sound, wav_bytes)
                audio_shape_match[str(match["same_shape_rate"])] += 1
                waveform_match[str(match["waveform_match"])] += 1

                features = extract_features(segment, sr)
                segment_duration = len(segment) / sr
                durations.append(segment_duration)
                expected_samples += 1

                output = {
                    "participant_id": pid,
                    "label": row["label"],
                    "age": row["age"],
                    "sex": row["sex"],
                    "visit_place": row["visit_place"],
                    "visit_year": years.get(pid),
                    "duration_s": row["duration_s"],
                    "n_segments": 1,
                    "annotation_start_s": ann_start,
                    "annotation_end_s": ann_end,
                    "annotation_duration_s": segment_duration,
                    "annotation_label": ann_label,
                    "prt_same_shape_rate_as_wav": int(match["same_shape_rate"]),
                    "prt_waveform_matches_wav": int(match["waveform_match"]),
                    "matched_age_sex": int(pid in age_sex_ids),
                    "matched_age_sex_year": int(pid in age_sex_year_ids),
                }
                output.update(features)
                output_rows.append(output)

            except Exception as exc:
                errors.append({"participant_id": pid, "error": f"{type(exc).__name__}: {exc}"})

            if index % 25 == 0 or index == len(rows):
                print(
                    f"Processados: {index}/{len(rows)} | "
                    f"válidos: {len(output_rows)} | erros: {len(errors)}",
                    flush=True,
                )

    finally:
        for archive in archives.values():
            archive.close()

    if output_rows:
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)

    if errors:
        with ERROR_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["participant_id", "error"])
            writer.writeheader()
            writer.writerows(errors)

    feature_names = [
        c for c in output_rows[0]
        if c.startswith(("mfcc_", "delta_", "delta2_", "dwt_", "wpt_", "signal_"))
    ] if output_rows else []

    with REPORT_TXT.open("w", encoding="utf-8") as report:
        report.write("EXTRAÇÃO AVFAD 002 — INTERVALO /a/ ANOTADO NO PRT\n")
        report.write("=" * 64 + "\n")
        report.write(f"Participantes HSA de entrada: {len(rows)}\n")
        report.write(f"Características extraídas: {len(output_rows)}\n")
        report.write(f"Erros: {len(errors)}\n")
        report.write(f"Dimensão do vetor acústico: {len(feature_names)}\n")
        report.write(f"Amostra pareada sexo + idade: {len(age_sex_ids)} ({age_sex_pairs} pares)\n")
        report.write(f"Amostra pareada sexo + idade + ano: {len(age_sex_year_ids)} ({age_sex_year_pairs} pares)\n")
        report.write(f"Fonte temporal das características: intervalo /a/ anotado no TextGrid do PRT\n")
        report.write(f"Taxa de amostragem exigida: {EXPECTED_SR} Hz\n")
        report.write(f"Wavelet: {WAVELET}; nível máximo: {WAVELET_LEVEL}\n")
        report.write("MFCC: 25 coeficientes; 120 filtros Mel; janela 25 ms; avanço 10 ms; fmax 24 kHz\n")
        report.write("Segmentos por participante: exatamente 1 intervalo /a/ anotado\n")
        if durations:
            report.write("\nDuração dos intervalos /a/ efetivamente extraídos:\n")
            report.write(f"  Mínima: {min(durations):.4f}s\n")
            report.write(f"  Mediana: {float(np.median(durations)):.4f}s\n")
            report.write(f"  Máxima: {max(durations):.4f}s\n")
            report.write(f"  Abaixo de 0,75s: {sum(d < 0.75 for d in durations)}\n")
            report.write(f"  Abaixo de 1,00s: {sum(d < 1.00 for d in durations)}\n")
        report.write("\nComparação do Sound embutido no PRT com o WAV original:\n")
        report.write(f"  Mesma forma e taxa de amostragem: {audio_shape_match['True']} de {len(output_rows)}\n")
        report.write(f"  Mesma forma de onda (tolerância 1e-5): {waveform_match['True']} de {len(output_rows)}\n")
        report.write("\nObservação: a extração usa o Sound que está dentro do próprio PRT, pois é esse áudio que o TextGrid anota.\n")
        if errors:
            report.write("\nPrimeiros erros (máximo 20):\n")
            for item in errors[:20]:
                report.write(f"  {item['participant_id']}: {item['error']}\n")
        report.write("\nArquivos gerados:\n")
        report.write(f"  {OUTPUT_CSV.name}\n")
        if errors:
            report.write(f"  {ERROR_CSV.name}\n")
        report.write(f"  {REPORT_TXT.name}\n")

    print("\nExtração concluída.")
    print(f"Resumo: {REPORT_TXT}")
    print(f"CSV: {OUTPUT_CSV}")
    if errors:
        print(f"Erros: {ERROR_CSV}")
    print("Envie somente o conteúdo do resumo TXT.")


if __name__ == "__main__":
    main()
