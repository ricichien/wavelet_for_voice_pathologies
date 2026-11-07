# -*- coding: utf-8 -*-
"""Exp23: qualidade do sinal, WPT48k com RMS global e sequencias temporais.

Nao treina classificadores. Preserva todos os casos e os caches anteriores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from math import gcd
from pathlib import Path
import platform
import sqlite3
import sys
import time
import zipfile
import zlib

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp18_for_exp23", HERE / "18_extrair_wpt_16khz_avfad.py")
EXP18 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP18)
LEGACY = EXP18.LEGACY
VERSION = "23.1"
SAMPLE_RATE, TARGET_RMS, WIDTH, DIM = 48000, .1, 480, 1920
CLIP_THRESHOLD = 1.0 - 1e-6
OUTPUT_NAME = "resultados_23_wpt_rms_sequencias"
EXPORTS = ("features_wpt_rms.npz", "sequencias_wpt.npz", "metadata_wpt_rms.csv",
           "QUALIDADE_AUDIO.csv", "RESUMO_EXTRACAO.txt", "assinatura_execucao.json",
           "CONTROLE_REFERENCIA_48KHZ.csv")


def sha(path):
    return EXP18.digest_file(Path(path))


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(text.encode("utf8"))
    temporary.replace(path)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def log(out, value):
    EXP18.log(out, value)


def audio_quality(audio, sr):
    """Medidas do WAV decodificado, antes de downmix/DC/ganho; sem exclusoes."""
    x = np.asarray(audio)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("WAV vazio, nao finito ou com formato inesperado")
    # Evita uma copia float64 permanente do audio longo inteiro.
    dc = np.mean(x, axis=0, dtype=np.float64)
    power = np.mean(np.square(x, dtype=np.float64), axis=0)
    rms = np.sqrt(power)
    ac_rms = np.sqrt(np.maximum(power - dc ** 2, 0.0))
    peak = np.max(np.abs(x), axis=0)
    clips = np.mean(np.abs(x) >= CLIP_THRESHOLD, axis=0)
    mono = x.mean(axis=1, dtype=np.float64).astype(np.float32)
    mono_dc = float(np.mean(mono, dtype=np.float64))
    mono_power = float(np.mean(np.square(mono, dtype=np.float64)))
    mono_ac = float(np.sqrt(max(mono_power - mono_dc ** 2, 0.0)))
    denominator = float(np.sqrt(np.mean(ac_rms ** 2)))
    correlations = []
    for left in range(x.shape[1]):
        for right in range(left + 1, x.shape[1]):
            if ac_rms[left] > 0 and ac_rms[right] > 0:
                product = np.mean(np.multiply(x[:, left], x[:, right], dtype=np.float64))
                correlation = (product - dc[left] * dc[right]) / (ac_rms[left] * ac_rms[right])
                correlations.append(float(np.clip(correlation, -1, 1)))
    ratio = mono_ac / denominator if denominator > 0 else 0.0
    return {"original_sample_rate_hz": int(sr), "channels_original": int(x.shape[1]),
            "source_samples": len(x), "source_duration_s": len(x) / float(sr),
            "channel_dc": json.dumps(dc.tolist()), "channel_rms": json.dumps(rms.tolist()),
            "channel_ac_rms": json.dumps(ac_rms.tolist()), "channel_peak": json.dumps(peak.tolist()),
            "channel_clip_fraction": json.dumps(clips.tolist()),
            "clip_threshold": CLIP_THRESHOLD, "max_clip_fraction": float(clips.max()),
            "mono_dc_raw": mono_dc, "mono_rms_raw": float(np.sqrt(mono_power)),
            "mono_ac_rms_raw": mono_ac, "mono_peak_raw": float(np.max(np.abs(mono))),
            "mono_to_channel_ac_rms_ratio": ratio,
            "min_channel_correlation": min(correlations) if correlations else None,
            "max_channel_correlation": max(correlations) if correlations else None,
            "flag_possible_channel_cancellation": int(x.shape[1] > 1 and ratio < .1),
            "flag_any_full_scale_sample": int(np.any(clips > 0)),
            "flag_duration_over_60s": int(len(x) / float(sr) > 60)}


def process_audio48(audio, sr):
    """Mesmas operacoes e dtypes de EXP18.read_signal48, lendo o WAV uma vez."""
    quality = audio_quality(audio, sr)
    signal = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
    if int(sr) != SAMPLE_RATE:
        factor = gcd(int(sr), SAMPLE_RATE)
        signal = resample_poly(signal, SAMPLE_RATE // factor, int(sr) // factor).astype(np.float32)
    signal = signal - np.mean(signal, dtype=np.float64)
    if float(np.max(np.abs(signal))) < 1e-8:
        raise ValueError("Audio silencioso: nao alterar a coorte silenciosamente")
    return signal, quality


def read_audio48(root, row, opened):
    if row.zip_name not in opened:
        opened[row.zip_name] = zipfile.ZipFile(root / row.zip_name, "r")
    audio, sr = sf.read(io.BytesIO(opened[row.zip_name].read(row.audio_path_in_zip)),
                        dtype="float32", always_2d=True)
    return process_audio48(audio, sr)


def normalize_regions(regions, target=TARGET_RMS):
    """Um unico ganho calculado de TODA voz ativa, sem ajuste por janela."""
    count = sum(len(region) for region in regions)
    if not count or not np.isfinite(target) or target <= 0:
        raise ValueError("Regioes/target RMS invalidos")
    power = sum(float(np.sum(np.square(region, dtype=np.float64))) for region in regions) / count
    rms = float(np.sqrt(power))
    if not np.isfinite(rms) or rms <= 1e-12:
        raise ValueError("Voz ativa com RMS nulo ou invalido")
    gain = float(target / rms)
    normalized = [(np.asarray(region, dtype=np.float64) * gain).astype(np.float32) for region in regions]
    achieved = float(np.sqrt(sum(float(np.sum(np.square(region, dtype=np.float64)))
                                for region in normalized) / count))
    if not np.isclose(achieved, target, atol=1e-8, rtol=1e-6):
        raise ValueError("RMS normalizado nao corresponde ao alvo")
    return normalized, {"active_rms_before": rms, "rms_global_gain": gain,
                        "active_rms_after": achieved, "target_active_rms": target,
                        "normalized_peak": float(max(np.max(np.abs(region)) for region in normalized))}


def sequence(windows, batch_size=128):
    values = np.vstack([EXP18.descriptors_batch(windows[start:start + batch_size])
                        for start in range(0, len(windows), batch_size)])
    if values.shape != (len(windows), WIDTH) or not np.isfinite(values).all():
        raise ValueError("Sequencia WPT invalida")
    return values


def summarize(values):
    x = np.asarray(values)
    if x.ndim != 2 or x.shape[1] != WIDTH or not len(x) or not np.isfinite(x).all():
        raise ValueError("Sequencia para resumo invalida")
    return np.concatenate((x.mean(axis=0), x.std(axis=0),
                           np.quantile(x, .1, axis=0), np.quantile(x, .9, axis=0))).astype(np.float32)


def extract_signal48(signal, reference=None, batch_size=128):
    regions, fallback = LEGACY.detect_speech_regions(signal, SAMPLE_RATE)
    normalized, normalization = normalize_regions(regions)
    raw_windows = LEGACY.make_windows(regions, SAMPLE_RATE)
    rms_windows = LEGACY.make_windows(normalized, SAMPLE_RATE)
    if len(raw_windows) != len(rms_windows):
        raise ValueError("Normalizacao mudou quantidade de janelas")
    raw_values, rms_values = sequence(raw_windows, batch_size), sequence(rms_windows, batch_size)
    raw_vector, rms_vector = summarize(raw_values), summarize(rms_values)
    control = {"equivalent": True, "max_abs_error": 0., "rtol": 2e-5, "atol": 2e-5}
    if reference is not None:
        if np.asarray(reference).shape != (DIM,):
            raise ValueError("Referencia com dimensao incorreta")
        control["max_abs_error"] = float(np.max(np.abs(raw_vector - reference)))
        control["equivalent"] = bool(np.allclose(raw_vector, reference, rtol=2e-5, atol=2e-5))
        if not control["equivalent"]:
            raise ValueError(f"Raw WPT48k nao reproduziu a referencia: erro={control['max_abs_error']}")
    active = sum(len(region) for region in regions)
    diag = {"speech_regions": len(regions), "window_count": len(raw_values),
            "active_duration_s": active / SAMPLE_RATE, "recording_duration_s": len(signal) / SAMPLE_RATE,
            "speech_ratio": active / len(signal), "vad_fallback": int(fallback),
            "sample_rate_hz": SAMPLE_RATE, "source_processing_sr": SAMPLE_RATE,
            "target_active_samples": active, **normalization}
    return raw_values, rms_values, rms_vector, diag, control


def encode(array):
    x = np.ascontiguousarray(array, dtype=np.float32)
    blob = x.tobytes()
    return zlib.compress(blob, level=3), hashlib.sha256(blob).hexdigest()


def decode(blob, digest, shape):
    try:
        plain = zlib.decompress(blob)
    except zlib.error as exc:
        raise ValueError("Blob WPT comprimido corrompido") from exc
    if hashlib.sha256(plain).hexdigest() != digest:
        raise ValueError("SHA256 WPT divergente")
    x = np.frombuffer(plain, dtype=np.float32)
    if x.size != int(np.prod(shape)) or not np.isfinite(x).all():
        raise ValueError("Shape/valores WPT invalidos")
    return x.reshape(shape).copy()


def init_cache(path, signature):
    exists = path.exists()
    con = sqlite3.connect(f"file:{path.as_posix()}?mode={'rw' if exists else 'rwc'}", uri=True)
    con.execute("PRAGMA synchronous=FULL")
    if not exists:
        con.execute("CREATE TABLE config (signature TEXT NOT NULL)")
        con.execute("INSERT INTO config VALUES (?)", (signature,))
        con.execute("CREATE TABLE features (sample_id TEXT PRIMARY KEY, participant_id TEXT NOT NULL, task TEXT NOT NULL, n_windows INTEGER NOT NULL, raw BLOB NOT NULL, rms BLOB NOT NULL, vector BLOB NOT NULL, raw_sha TEXT NOT NULL, rms_sha TEXT NOT NULL, vector_sha TEXT NOT NULL, diagnostics TEXT NOT NULL, quality TEXT NOT NULL, control TEXT NOT NULL)")
        con.execute("CREATE TABLE errors (sample_id TEXT PRIMARY KEY, error TEXT NOT NULL)")
        con.commit()
    else:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != {"config", "features", "errors"}:
            con.close(); raise ValueError("SchemaSQLite desconhecido")
    if con.execute("SELECT signature FROM config").fetchall() != [(signature,)]:
        con.close(); raise ValueError("AssinaturaSQLite divergente; preservar cache")
    return con


def read_row(record):
    sid, pid, task, n, raw, rms, vector, hsraw, hsrms, hsvec, diag, quality, control = record
    if not isinstance(n, int) or n < 1:
        raise ValueError("Quantidade de janelas invalida")
    return sid, pid, task, decode(raw, hsraw, (n, WIDTH)), decode(rms, hsrms, (n, WIDTH)), \
        decode(vector, hsvec, (DIM,)), json.loads(diag), json.loads(quality), json.loads(control)


def validate_cache(con, selected, reference=None):
    expected = selected.set_index("sample_id")
    found = {}
    for record in con.execute("SELECT * FROM features"):
        sid, pid, task, raw, rms, vector, diag, quality, control = read_row(record)
        if sid not in expected.index or str(expected.loc[sid, "participant_id"]) != pid or str(expected.loc[sid, "task"]) != task:
            raise ValueError("ID/pessoa/tarefa fora do manifesto")
        if len(raw) != len(rms) or diag.get("window_count") != len(raw):
            raise ValueError("Janelas/diagnostico inconsistentes")
        np.testing.assert_array_equal(summarize(rms), vector)
        if control.get("equivalent") is not True:
            raise ValueError("Controle raw WPT48k nao equivalente")
        if reference is not None:
            target = reference[int(expected.loc[sid, "row_index"])]
            raw_vector = summarize(raw)
            maximum = float(np.max(np.abs(raw_vector - target)))
            if not np.allclose(raw_vector, target, rtol=2e-5, atol=2e-5) or maximum != control.get("max_abs_error"):
                raise ValueError("Controle raw nao corresponde a referencia auditada")
        if not np.isclose(diag.get("active_rms_after", np.nan), TARGET_RMS, atol=1e-8, rtol=1e-6):
            raise ValueError("RMS global fora do alvo")
        found[sid] = {"windows": len(raw), "diagnostics": diag, "quality": quality, "control": control}
    if any(sid not in expected.index for sid, in con.execute("SELECT sample_id FROM errors")):
        raise ValueError("ErroSQLite fora do manifesto")
    return found


def export(out, selected, found, con, names, base_names, ident, elapsed, work_seconds, new_count):
    if set(found) != set(selected.sample_id) or con.execute("SELECT count(*) FROM errors").fetchone()[0]:
        raise ValueError("Nenhuma exportacao final de coorte incompleta/comerros")
    metadata = selected.copy().reset_index(drop=True)
    diagnostics = pd.DataFrame([found[sid]["diagnostics"] for sid in selected.sample_id])
    for column in diagnostics:
        metadata[column] = diagnostics[column]
    metadata["matrix_row"] = np.arange(len(metadata))
    counts = np.asarray([found[sid]["windows"] for sid in selected.sample_id], dtype=np.int64)
    offsets = np.r_[np.int64(0), np.cumsum(counts, dtype=np.int64)]
    raw_data = np.empty((int(offsets[-1]), WIDTH), dtype=np.float32)
    rms_data = np.empty_like(raw_data)
    matrix = np.empty((len(metadata), DIM), dtype=np.float32)
    for index, sid in enumerate(selected.sample_id):
        row = con.execute("SELECT * FROM features WHERE sample_id=?", (sid,)).fetchone()
        actual, _, _, raw, rms, vector, _, _, _ = read_row(row)
        if actual != sid:
            raise ValueError("Ordem SQLite incorreta")
        raw_data[offsets[index]:offsets[index + 1]] = raw
        rms_data[offsets[index]:offsets[index + 1]] = rms
        matrix[index] = vector
    ids = {key: metadata[key].to_numpy(dtype=str) for key in ("sample_id", "participant_id", "task")}
    for filename, payload in ((EXPORTS[0], dict(wavelet=matrix, feature_names=names, **ids)),
                              (EXPORTS[1], dict(data_raw=raw_data, data_rms=rms_data, offsets=offsets,
                                               feature_names=base_names, **ids))):
        target = out / filename
        temporary = target.with_name(target.name + ".part")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        temporary.replace(target)
    EXP18.atomic_csv(out / EXPORTS[2], metadata)
    quality = pd.DataFrame([{"sample_id": row.sample_id, "participant_id": row.participant_id, "task": row.task,
                             **found[row.sample_id]["quality"], **found[row.sample_id]["diagnostics"]}
                            for row in metadata.itertuples()])
    EXP18.atomic_csv(out / EXPORTS[3], quality)
    control = pd.DataFrame([{"sample_id": sid, "source_row": int(selected.set_index("sample_id").loc[sid, "row_index"]),
                             **found[sid]["control"]} for sid in metadata.sample_id])
    EXP18.atomic_csv(out / EXPORTS[6], control)
    rate = work_seconds / new_count if new_count else None
    report = ["EXPERIMENTO23 — QUALIDADE, RMS GLOBAL E SEQUENCIAS WPT48K", f"Concluido:{EXP18.timestamp()}",
              f"Modo:{ident['mode']}; nenhuma exclusao; flags sao diagnosticos, nao filtros.",
              f"MatrizRMS:{matrix.shape}; pessoas:{metadata.participant_id.nunique()}; nao finitos:{int((~np.isfinite(matrix)).sum())}",
              f"Sequencias raw/RMS:{raw_data.shape}; janelas por gravacao min/mediana/max:{counts.min()}/{np.median(counts)}/{counts.max()}",
              f"RMS ativo alvo:{TARGET_RMS}; ganho global por gravacao; VAD fixo anterior ao ganho.",
              f"Controle48k:todos{len(control)}aprovados; maior erro={control.max_abs_error.max()}",
              f"Qualidade:clipping(raw)em{int(quality.flag_any_full_scale_sample.sum())}; suspeita_cancelamento={int(quality.flag_possible_channel_cancellation.sum())}; duracao>60s={int(quality.flag_duration_over_60s.sum())}",
              f"Tempo desta chamada:{elapsed:.3f}; novas linhas:{new_count}; segundos novas linhas:{work_seconds:.3f}; por linha:{rate}",
              "Flags de clipping usam amplitude absoluta>=1-1e-6 no WAV decodificado; nao provam etiologia nem corrigem distorcao.",
              "Normalizacao float32 nao aplica clipping, mesmo quando picos normalizados excedem1.",
              "Nenhum classificador/BA, normalizacao entre pessoas ou rotulo usado na extracao.", ""]
    atomic_text(out / EXPORTS[4], "\n".join(report))
    atomic_text(out / "ARTEFATOS_COMPLETOS.json", json_text({name: sha(out / name) for name in EXPORTS}))


def identity(paths, archives, selected, mode, batch_size):
    code = [Path(__file__).resolve(), Path(EXP18.__file__).resolve(), Path(LEGACY.__file__).resolve(), HERE / "avfad_paths.py"]
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "PyWavelets", "librosa", "soundfile"):
        environment[name] = importlib.metadata.version(name)
    return {"version": VERSION, "mode": mode, "sample_ids": selected.sample_id.tolist(),
            "configuration": {"sample_rate": SAMPLE_RATE, "target_rms": TARGET_RMS,
                              "normalization": "global_active_rms", "vad_before_normalization": True,
                              "wavelet": "sym5", "levels": [5, 6], "wavelet_extension": "symmetric",
                              "window_seconds": [.4, .2], "descriptors": list(LEGACY.DESCRIPTORS),
                              "summary_stats": list(LEGACY.SUMMARY_STATS), "sequence_width": WIDTH,
                              "sequence_dtype": "float32", "summary_dtype": "float32", "batch_windows": batch_size,
                              "quality_clip_threshold": CLIP_THRESHOLD, "label_used_to_extract": False},
            "environment": environment, "inputs": {str(p.resolve()): {"sha256": sha(p), "size_bytes": p.stat().st_size} for p in paths},
            "code": {str(p.resolve()): sha(p) for p in code},
            "raw_zip_metadata_only": {str(p.resolve()): {"size_bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in archives}}


def protocol(ident):
    return f"""EXPERIMENTO23 — QUALIDADE/RMS/SEQUENCIAS WPT; modo={ident['mode']}
Coorte original411pessoas(82normais/329patologicos),2465gravacoes CAPE-V004–009.
Nenhuma exclusao nova; gravações longas/estereo permanecem. Flags nao removem casos.
Leitura em memoria dos ZIPs, downmix/DC/resample48k identicos ao18/legado.
Qualidade medida no WAV original: RMS/pico/DC/clipping por canal, correlacao entre
canais, razao RMS mono/canais e flags; nenhuma aprendizagem ou rotulo na extracao.
VAD legado no sinal48k original UMA vez; top_db40/frame2048/hop480/merge250ms/min250ms.
RMS sobre TODAS as amostras da voz ativa; ganho GLOBAL0.1/RMS por gravacao.
Aplicar o mesmo ganho em todas as regioes antes de janelar. Sem RMS por janela,
peak-normalization, clipping adicional, denoising, limites de duracao ou descarte.
Janelas400ms/passo200ms/reflexao/ultima janela como legado; DC por janela.
WPTsym5 symmetric L5/L6 freq,96nos*5descritores=480 valores por janela.
Cinco descritores:rel_energy/log_energy/entropy/kurtosis/zcr, formulas18 inalteradas.
Salvar sequencias raw e RMS preservando ORDEM e contagem de janelas, float32.
ResumoRMS:media/DP/q10/q90,1920atributos,mesmos nomes/ordem da referencia.
Controle RAW contra NPZ48k legado em TODAS as linhas:rtol=atol=2e-5.
Normalizar amplitude pode remover informacao util; nao se presume melhoria.
Clinico/ruido/clipping nao se inferem apenas de flags. Nao altera rotulos.
Nenhum classificador ou resultado de BA; futura selecao/escala apenas no treino.
CacheSQLite zlib novo; hashes dos blobs descomprimidos, assinatura estrita eIDs.
Exportacao final apenas apos todos os casos/controles; piloto em pasta separada.
AVFAD ja explorada: desenvolvimento, nao teste cego; dados/caches/resultados antigos
somente leitura. Marcador final associa hashes dos sete artefatos completos.
"""


def run(root, out, selected, reference, names, base_names, ident, batch_size):
    text = json_text(ident)
    signature = hashlib.sha256(text.encode("utf8")).hexdigest()
    file = out / "assinatura_execucao.json"
    if file.exists():
        if file.read_text(encoding="utf8") != text:
            raise ValueError("Configuracao/codigo/dados mudaram; preservar pasta")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Pasta nao vazia sem assinatura; preservar e revisar")
        out.mkdir(parents=True, exist_ok=True)
        atomic_text(file, text); atomic_text(out / "PROTOCOLO.txt", protocol(ident))
    if (out / "PROTOCOLO.txt").read_text(encoding="utf8") != protocol(ident):
        raise ValueError("Protocolo congelado alterado")
    con = init_cache(out / "checkpoint_wpt_rms_sequencias.sqlite", signature)
    opened, started = {}, time.perf_counter()
    new_count, work_seconds = 0, 0.
    try:
        found = validate_cache(con, selected, reference)
        marker = out / "ARTEFATOS_COMPLETOS.json"
        if marker.exists():
            hashes = json.loads(marker.read_text(encoding="utf8"))
            if set(hashes) != set(EXPORTS) or set(found) != set(selected.sample_id) or con.execute("SELECT count(*) FROM errors").fetchone()[0]:
                raise ValueError("Marcador final incompleto/inconsistente")
            if any(sha(out / name) != digest for name, digest in hashes.items()):
                raise ValueError("Exportacao final alterada")
            log(out, "Retomada: cache/exportacoes integros; nenhuma releitura de audio.")
            return
        log(out, f"Inicio/retomada23:{ident['mode']}; {len(found)}/{len(selected)} em cache; sem exclusoes.")
        for row in selected.itertuples(index=False):
            if row.sample_id in found:
                continue
            one_start = time.perf_counter()
            try:
                signal, quality = read_audio48(root, row, opened)
                raw, rms, vector, diag, control = extract_signal48(signal, reference[int(row.row_index)], batch_size)
                diag["extraction_seconds"] = time.perf_counter() - one_start
                rb, rh = encode(raw); nb, nh = encode(rms); vb, vh = encode(vector)
                with con:
                    con.execute("INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (row.sample_id, row.participant_id, row.task,
                                len(raw), rb, nb, vb, rh, nh, vh, json_text(diag), json_text(quality), json_text(control)))
                    con.execute("DELETE FROM errors WHERE sample_id=?", (row.sample_id,))
                found[row.sample_id] = {"windows": len(raw), "diagnostics": diag, "quality": quality, "control": control}
                new_count += 1; work_seconds += diag["extraction_seconds"]
            except Exception as exc:
                with con:
                    con.execute("INSERT OR REPLACE INTO errors VALUES (?,?)", (row.sample_id, repr(exc)))
                log(out, f"ERRO:{row.sample_id}:{exc!r}; nenhuma exportacao parcial")
                raise
            if ident["mode"] != "completo" or len(found) % 25 == 0 or len(found) == len(selected):
                log(out, f"{len(found)}/{len(selected)};ultimo={row.sample_id};segundos={diag['extraction_seconds']:.3f};janelas={len(raw)};raw_erro={control['max_abs_error']:.3g}")
        found = validate_cache(con, selected, reference)
        export(out, selected, found, con, names, base_names, ident, time.perf_counter() - started, work_seconds, new_count)
        log(out, "Extracao23 concluida; todas as linhas raw equivalentes, RMS global e sequencias finitas.")
    finally:
        con.close()
        for archive in opened.values():
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--batch-janelas", type=int, default=128)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true")
    action.add_argument("--piloto", type=int, nargs="?", const=3)
    action.add_argument("--executar", action="store_true")
    action.add_argument("--autoteste", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        import unittest
        spec = importlib.util.spec_from_file_location("test_wpt_extracao23", HERE / "test_wpt_extracao23.py")
        tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful():
            raise SystemExit(1)
        return
    if not 1 <= args.batch_janelas <= 512:
        parser.error("Batch deve estar entre1 e512")
    if args.piloto is not None and not 1 <= args.piloto <= 10:
        parser.error("Piloto deve conter1 a10 gravacoes")
    root = args.dados.resolve()
    meta, reference, names, base, paths, archives = EXP18.load_inputs(root)
    pilot = args.piloto is not None
    selected = meta.iloc[np.unique(np.linspace(0, len(meta)-1, args.piloto, dtype=int))].copy() if pilot else meta.copy()
    ident = identity(paths, archives, selected, f"piloto_{len(selected)}" if pilot else "completo", args.batch_janelas)
    print(f"ENTRADAS23 OK:2465gravacoes/411pessoas;selecionadas={len(selected)};480descritores temporais e1920resumos;nenhum WAV lido.", flush=True)
    if not (pilot or args.executar):
        return
    out = (args.saida or root / "results/experiments" /
           (f"resultados_23_wpt_rms_sequencias_piloto_{len(selected)}" if pilot else OUTPUT_NAME)).resolve()
    if not out.is_relative_to(root / "results/experiments") or out == root / "results/experiments":
        parser.error("Saida deve ser subpasta de results/experiments")
    if pilot and out == root / "results/experiments" / OUTPUT_NAME:
        parser.error("Piloto nao pode usar pasta da coorte completa")
    run(root, out, selected, reference, names, base, ident, args.batch_janelas)


if __name__ == "__main__":
    main()
