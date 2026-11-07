# -*- coding: utf-8 -*-
"""Exp18: nova WPT sym5 a 16 kHz, preservando VAD/coorte/janelas do legado.

Nao treina modelos. --verificar somente le; --piloto escreve pasta separada;
--executar exige chamada explicita e exporta somente se toda a coorte concluir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import io
import json
from math import gcd
from pathlib import Path
import platform
import sqlite3
import sys
import time
import warnings
import zipfile

import numpy as np
import pandas as pd
import pywt
import soundfile as sf
from scipy.signal import resample_poly
from scipy.stats import kurtosis

import extrair_wavelet_temporal_avfad_reduzido as LEGACY

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/experiments/resultados_18_wpt_16khz_extracao"
VERSION = "18.1"
SOURCE_SR, TARGET_SR = 48000, 16000
LEVELS = (5, 6)
DIM = 1920
BASE_DIM = 480
EPS = 1e-12
EXPORT_FILES = ("features_wpt_sym5_16khz.npz", "metadata_wpt_sym5_16khz.csv", "RESUMO_EXTRACAO.txt",
                "CONTROLE_REFERENCIA_48KHZ.csv", "CONTROLE_VALIDADO.json")


def timestamp():
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def atomic_text(path, text):
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path, frame):
    tmp = path.with_name(path.name + ".part")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.17g")
    tmp.replace(path)


def log(out, text):
    line = f"[{timestamp()}] {text}"
    print(line, flush=True)
    with (out / "LOG_EXTRACAO.txt").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def feature_names():
    names, base = LEGACY.build_feature_names()
    names = np.asarray([s for s in names if s.startswith("WPT__sym5__")])
    base = np.asarray([s for s in base if s.startswith("WPT__sym5__")])
    if len(names) != DIM or len(base) != BASE_DIM:
        raise ValueError("Configuracao legada de nomes mudou; revisar antes de extrair")
    return names, base


def load_inputs(root):
    metadata = root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv"
    reference = root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz"
    meta = pd.read_csv(metadata, dtype={"participant_id": str, "sample_id": str, "task": str})
    required = {"row_index", "sample_id", "participant_id", "task", "y", "zip_name", "audio_path_in_zip"}
    if not required.issubset(meta) or meta[list(required)].isna().any().any():
        raise ValueError("Manifesto sem identificadores, rotulos ou caminhos completos")
    meta = meta.sort_values("row_index", kind="stable").reset_index(drop=True)
    for col in ("sample_id", "participant_id"):
        meta[col] = meta[col].str.strip().str.upper()
        if meta[col].eq("").any():
            raise ValueError("Identificador vazio")
    meta["task"] = meta.task.str.strip().str.zfill(3)
    if meta.sample_id.duplicated().any() or meta.duplicated(["participant_id", "task"]).any():
        raise ValueError("Gravacao/tarefa duplicada")
    if meta.groupby("participant_id").y.nunique().ne(1).any():
        raise ValueError("Rotulos divergentes na mesma pessoa")
    people = meta.drop_duplicates("participant_id")
    if (len(meta), len(people), people.y.value_counts().to_dict()) != (2465, 411, {1: 329, 0: 82}):
        raise ValueError("Coorte deve ser 2465 gravacoes / 411 pessoas / 82 normais / 329 patologicos")
    if set(meta.task) != {"004", "005", "006", "007", "008", "009"}:
        raise ValueError("Tarefas fora da coorte CAPE-V")
    if not np.array_equal(meta.row_index.to_numpy(), np.arange(len(meta))):
        raise ValueError("row_index legado deve corresponder a ordem auditada do NPZ")
    names, base = feature_names()
    with np.load(reference, allow_pickle=False) as z:
        wn = z["feature_names"].astype(str)
        keep = np.char.startswith(wn, "WPT__sym5__")
        ref = np.asarray(z["wavelet"][:, keep], dtype=np.float32)
        if not np.array_equal(wn[keep], names) or ref.shape != (len(meta), DIM) or not np.isfinite(ref).all():
            raise ValueError("Referencia WPT e manifesto nao tem estrutura auditada esperada")
        if "sample_id" in z and not np.array_equal(z["sample_id"].astype(str), meta.sample_id):
            raise ValueError("IDs embutidos na referencia discordam do manifesto")
    archives = sorted(root.glob("AVFAD_01_00_00_*.zip"))
    if len(archives) != 5:
        raise ValueError("Esperava os cinco ZIPs originais na raiz")
    allowed = {p.name for p in archives}
    if not set(meta.zip_name).issubset(allowed):
        raise ValueError("Manifesto aponta ZIP fora dos cinco originais")
    return meta, ref, names, base, (metadata, reference), archives


def descriptors_batch(windows):
    """Mesmas formulas por janela; WaveletPacket vetorizado somente entre linhas."""
    x = np.asarray(windows, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Janelas precisam ser uma matriz finita")
    x = x - np.mean(x, axis=1, keepdims=True)
    packet = pywt.WaveletPacket(data=x, wavelet="sym5", mode="symmetric", maxlevel=6, axis=-1)
    output = []
    for level in LEVELS:
        nodes = packet.get_level(level, order="freq")
        coefficients = [np.asarray(n.data, dtype=np.float64) for n in nodes]
        energies = np.column_stack([np.sum(c * c, axis=1) for c in coefficients])
        totals = energies.sum(axis=1) + EPS
        for j, c in enumerate(coefficients):
            energy = energies[:, j]
            rel = energy / totals
            le = np.log10(energy / c.shape[1] + EPS)
            entropy = np.zeros(len(x), dtype=np.float64)
            if c.shape[1] > 1:
                valid = energy > EPS
                power = c[valid] * c[valid]
                p = power / energy[valid, None]
                terms = np.zeros_like(p)
                positive = p > 0
                terms[positive] = p[positive] * np.log2(p[positive])
                entropy[valid] = -terms.sum(axis=1) / np.log2(c.shape[1])
            k = np.zeros(len(x), dtype=np.float64)
            if c.shape[1] > 3:
                valid = np.std(c, axis=1) > 1e-14
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    k[valid] = kurtosis(c[valid], axis=1, fisher=True, bias=False, nan_policy="omit")
                k[~np.isfinite(k)] = 0
                k = np.clip(k, -50, 50)
            zcr = np.mean(np.signbit(c[:, 1:]) != np.signbit(c[:, :-1]), axis=1)
            output.extend((rel, le, entropy, k, zcr))
    values = np.column_stack(output).astype(np.float32)
    if values.shape != (len(x), BASE_DIM) or not np.isfinite(values).all():
        raise ValueError("Descritores invalidos")
    return values


def summarize_windows(windows, batch_size=128):
    values = np.vstack([descriptors_batch(windows[i:i + batch_size])
                        for i in range(0, len(windows), batch_size)])
    result = np.concatenate((values.mean(axis=0), values.std(axis=0),
                             np.quantile(values, 0.1, axis=0), np.quantile(values, 0.9, axis=0)))
    return result.astype(np.float32)


def extract_signal48(signal, target_sr=TARGET_SR, batch_size=128):
    if target_sr not in (SOURCE_SR, TARGET_SR):
        raise ValueError("Taxa alvo permitida: 48000 para controle ou 16000 para novo vetor")
    regions48, fallback = LEGACY.detect_speech_regions(signal, SOURCE_SR)
    regions = regions48 if target_sr == SOURCE_SR else [resample_poly(r, 1, 3).astype(np.float32) for r in regions48]
    windows = LEGACY.make_windows(regions, target_sr)
    vector = summarize_windows(windows, batch_size)
    active = sum(len(r) for r in regions48) / SOURCE_SR
    diag = {"speech_regions": len(regions48), "window_count": len(windows),
            "active_duration_s": active, "recording_duration_s": len(signal) / SOURCE_SR,
            "speech_ratio": active / (len(signal) / SOURCE_SR), "vad_fallback": int(fallback),
            "source_processing_sr": SOURCE_SR, "sample_rate_hz": target_sr,
            "target_active_samples": sum(len(r) for r in regions)}
    return vector, diag


def read_signal48(root, row, opened):
    name = row.zip_name
    if name not in opened:
        opened[name] = zipfile.ZipFile(root / name, "r")
    raw = opened[name].read(row.audio_path_in_zip)
    audio, original_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    if audio.ndim != 2 or audio.shape[0] == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio vazio ou nao finito")
    signal = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
    if int(original_sr) != SOURCE_SR:
        factor = gcd(int(original_sr), SOURCE_SR)
        signal = resample_poly(signal, SOURCE_SR // factor, int(original_sr) // factor).astype(np.float32)
    signal = signal - np.mean(signal, dtype=np.float64)
    if float(np.max(np.abs(signal))) < 1e-8:
        raise ValueError("Audio silencioso")
    return signal, {"original_sample_rate_hz": int(original_sr), "channels_original": int(audio.shape[1])}


def identity(meta_paths, archives, selected, mode, batch_size):
    code = [Path(__file__).resolve(), Path(LEGACY.__file__).resolve(), ROOT / "src/avfad_paths.py"]
    packages = ("numpy", "pandas", "scipy", "PyWavelets", "librosa", "soundfile")
    return {"version": VERSION, "mode": mode, "sample_ids": selected.sample_id.tolist(),
            "inputs": {str(p.resolve()): {"sha256": digest_file(p), "size_bytes": p.stat().st_size} for p in meta_paths},
            "code": {str(p): digest_file(p) for p in code},
            "raw_zip_metadata_only": {str(p.resolve()): {"size_bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in archives},
            "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                            **{p: importlib.metadata.version(p) for p in packages}},
            "configuration": {"target_sr": TARGET_SR, "source_processing_sr": SOURCE_SR,
                              "wavelet": "sym5", "levels": list(LEVELS), "mode": "symmetric",
                              "window_seconds": [0.4, 0.2], "batch_windows": batch_size,
                              "vad": [40, 2048, 480, 0.25, 0.25],
                              "descriptors": list(LEGACY.DESCRIPTORS), "summaries": list(LEGACY.SUMMARY_STATS)}}


def protocol(ident):
    return f"""EXPERIMENTO 18 - EXTRACAO WPT SYM5 A 16 KHZ
Registro anterior a extracao desta pasta; modo={ident['mode']}.
Hipotese independente dos resultados do17: com os mesmos niveis5/6, 16kHz
produz larguras nominais de250/125Hz em vez de750/375Hz a48kHz. Isso pode preservar
informacao util em frequencias baixas; nao se presume ganho de classificacao.
Nao ha treinamento nem BA neste extrator.

Coorte completa:2465 gravacoes/411 participantes HSA/82 normais/329 patologicos.
Piloto: subset deterministico de IDs, separado dos artefatos completos.
Fonte: WAVs originais dentro dos cincoZIPs, lidos sem descompactar para disco.
Downmix mono, conversao para48kHz se necessaria e remocao da media, como legado.
VAD EXCLUSIVAMENTE a48kHz: librosa.effects.split top_db40, frame2048,hop480,
uniao de pausas ate250ms, regiao minima250ms, mesmo fallback do legado.
Cada regiao resultante e reamostrada separadamente a16kHz com resample_poly1/3.
Janelas400ms/passo200ms, reflexao para regioes curtas e janela final como legado.
Remocao da media de cada janela; WPT sym5,symmetric,niveis5/6,nos freq.
Descritores: energia relativa por nivel; log10(energia/ncoef+1e-12); entropia
de Shannon da energia normalizada por log2(ncoef); curtose Fisher bias=False
limitada[-50,50];ZCR signbit. Media/DP/q10/q90 entre janelas, float32,1920attrs.
Sem normalizacao RMS/pico, sem DWT/MFCC, sem uso de rotulos para extrair.

Antes da nova extracao, comparar engine vetorizada a48kHz com NPZ legado em
tres gravacoes deterministicas (rtol=2e-5,atol=2e-5). O piloto registra medidas,
tempo e extrapolacao aproximada de custo, sem extrapolar acuracia.
CacheSQLite novo nesta pasta: ID,pessoa,tarefa,vetor eSHA256. Falha interrompe,
e impede exportar matriz completa enquanto houver erro/ausencia. Retomada
confere identidade,hashes e IDs; nao sobrescreve caches/resultados antigos.
NPZ novo embute sample_id/participant_id/task/feature_names na ordem do CSV.
SHA256 dos insumos/codigo/versoes e metadados dosZIPs na assinatura.

Limitacoes: o antialias reduz largura de banda e a resolucao absoluta muda;
mesma arvore nao significa a mesma faixa por no. Filtros nas bordas das regioes
podem afetar coeficientes. Isso compara pipelines com taxas diferentes, sem
atribuir ganhos exclusivamente a uma propriedade matematica da WPT.
AVFAD ja explorada: futura avaliacao19 sera desenvolvimento por participante,
com controles e particoes congelados. Esta extracao nao e conclusao doTG.
"""


def init_cache(path, signature):
    exists = path.exists()
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode={'rw' if exists else 'rwc'}", uri=True)
    conn.execute("PRAGMA synchronous=FULL")
    if exists:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != {"config", "features", "errors"}:
            conn.close()
            raise ValueError("Schema de cache inesperado; preservar e revisar")
    else:
        conn.execute("CREATE TABLE config (signature TEXT NOT NULL)")
        conn.execute("INSERT INTO config VALUES (?)", (signature,))
        conn.execute("CREATE TABLE features (sample_id TEXT PRIMARY KEY, participant_id TEXT NOT NULL, task TEXT NOT NULL, vector BLOB NOT NULL, sha256 TEXT NOT NULL, diagnostics TEXT NOT NULL)")
        conn.execute("CREATE TABLE errors (sample_id TEXT PRIMARY KEY, error TEXT NOT NULL)")
        conn.commit()
    rows = conn.execute("SELECT signature FROM config").fetchall()
    if rows != [(signature,)]:
        conn.close()
        raise ValueError("Assinatura de cache diverge; nao sobrescrever")
    return conn


def validate_cache(conn, selected):
    expected = selected.set_index("sample_id")
    found = {}
    for sid, pid, task, blob, digest, diagnostics in conn.execute("SELECT * FROM features"):
        if sid not in expected.index or str(expected.loc[sid, "participant_id"]) != pid or str(expected.loc[sid, "task"]) != task:
            raise ValueError("ID/pessoa/tarefa de cache fora do manifesto")
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"Vetor corrompido no cache: {sid}")
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.shape != (DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"Vetor com dimensao/valores invalidos: {sid}")
        found[sid] = (vector.copy(), json.loads(diagnostics))
    if any(sid not in expected.index for sid, in conn.execute("SELECT sample_id FROM errors")):
        raise ValueError("Registro de erro fora do manifesto")
    return found


def reference_check(root, meta, reference, out, batch_size):
    indices = np.unique(np.linspace(0, len(meta) - 1, 3, dtype=int))
    opened, rows = {}, []
    try:
        for index in indices:
            row = meta.iloc[int(index)]
            start = time.perf_counter()
            signal, _ = read_signal48(root, row, opened)
            vector, diag = extract_signal48(signal, SOURCE_SR, batch_size)
            original = reference[int(index)]
            maximum = float(np.max(np.abs(vector - original)))
            ok = bool(np.allclose(vector, original, rtol=2e-5, atol=2e-5))
            rows.append({"sample_id": row.sample_id, "source_row": int(index), "max_abs_error": maximum,
                         "rtol": 2e-5, "atol": 2e-5, "equivalent": ok,
                         "windows": diag["window_count"], "seconds": time.perf_counter() - start})
            log(out, f"CONTROLE48k {row.sample_id}: erro_max={maximum:.3g}; equivalente={ok}")
            if not ok:
                atomic_csv(out / "CONTROLE_REFERENCIA_48KHZ.csv", pd.DataFrame(rows))
                raise ValueError("Engine vetorizada nao reproduziu WPT48k do NPZ; parar antes16k")
    finally:
        for archive in opened.values():
            archive.close()
    atomic_csv(out / "CONTROLE_REFERENCIA_48KHZ.csv", pd.DataFrame(rows))
    return rows


def export_complete(out, selected, found, names, base, elapsed, work_seconds, new_count, pilot, control):
    if set(found) != set(selected.sample_id):
        raise ValueError("Matriz incompleta: nenhuma exportacao final permitida")
    matrix = np.vstack([found[sid][0] for sid in selected.sample_id])
    diagnostics = pd.DataFrame([found[sid][1] for sid in selected.sample_id])
    metadata = selected.copy().reset_index(drop=True)
    for col in diagnostics:
        metadata[col] = diagnostics[col]
    metadata["matrix_row"] = np.arange(len(metadata))
    target = out / "features_wpt_sym5_16khz.npz"
    temporary = target.with_name(target.name + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, wavelet=matrix, feature_names=names, base_feature_names=base,
                            sample_id=metadata.sample_id.to_numpy(dtype=str),
                            participant_id=metadata.participant_id.to_numpy(dtype=str),
                            task=metadata.task.to_numpy(dtype=str))
    temporary.replace(target)
    atomic_csv(out / "metadata_wpt_sym5_16khz.csv", metadata)
    rate = work_seconds / new_count if new_count else None
    estimate = rate * 2465 if rate else None
    report = ["EXPERIMENTO18 - EXTRACAO WPT16KHZ", f"Concluido: {timestamp()}",
              "PILOTO: nao misturar com matriz completa." if pilot else "COORTE COMPLETA: exportacao sem excluir casos.",
              f"Matriz: {matrix.shape}; pessoas: {metadata.participant_id.nunique()}; nao finitos: {int((~np.isfinite(matrix)).sum())}",
              f"Linhas novas nesta chamada: {new_count}; segundos totais desta chamada: {elapsed:.3f}",
              f"Segundos das novas gravacoes16k (sem controle48k, exportacao ou verificacao): {work_seconds:.3f}",
              f"Segundos por gravacao nova: {rate}; estimativa aproximada para2465: {estimate}",
              "A estimativa depende de duracao/ZIP/cache; piloto nao mede desempenho de classificador.",
              f"Controles48k realizados: {len(control)}; todos passaram: {all(r['equivalent'] for r in control)}",
              f"Janelas por gravacao min/mediana/max: {metadata.window_count.min()}/{metadata.window_count.median()}/{metadata.window_count.max()}",
              f"FallbacksVAD: {metadata.vad_fallback.sum()}; taxa alvo16k; VAD mantido48k.",
              "IDs embutidos no NPZ e matrix_row no CSV permitem verificar alinhamento.",
              "Nenhum treino, BA, selecao supervisionada ou descarte por desempenho nesta etapa.",
              "Ver PROTOCOLO.txt, assinatura_execucao.json, controle48k e checkpointSQLite novo.", ""]
    atomic_text(out / "RESUMO_EXTRACAO.txt", "\n".join(report))
    exports = [out / name for name in EXPORT_FILES]
    atomic_text(out / "ARTEFATOS_COMPLETOS.json", json_text({p.name: digest_file(p) for p in exports}))


def run(root, out, meta, reference, names, base, ident, pilot, batch_size):
    signature_text = json_text(ident)
    signature = hashlib.sha256(signature_text.encode("utf-8")).hexdigest()
    sfpath = out / "assinatura_execucao.json"
    if sfpath.exists():
        if sfpath.read_text(encoding="utf-8") != signature_text:
            raise ValueError("Configuracao/insumos/codigo mudaram; nao reutilizar pasta")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Pasta nao vazia sem assinatura; preservar e revisar")
        out.mkdir(parents=True, exist_ok=True)
        atomic_text(sfpath, signature_text)
        atomic_text(out / "PROTOCOLO.txt", protocol(ident))
    if (out / "PROTOCOLO.txt").read_text(encoding="utf-8") != protocol(ident):
        raise ValueError("Protocolo congelado diverge")
    selected = meta[meta.sample_id.isin(ident["sample_ids"])].copy()
    conn = init_cache(out / "checkpoint_wpt_16khz.sqlite", signature)
    opened, new_count, work_seconds = {}, 0, 0.0
    started = time.perf_counter()
    try:
        found = validate_cache(conn, selected)
        completed = out / "ARTEFATOS_COMPLETOS.json"
        if completed.exists():
            exported = json.loads(completed.read_text(encoding="utf-8"))
            if set(exported) != set(EXPORT_FILES):
                raise ValueError("Marcador de exportacao final nao possui os artefatos esperados")
            if set(found) != set(selected.sample_id) or conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0]:
                raise ValueError("Exportacao marcada completa mas cache incompleto/com erros")
            if any(digest_file(out / name) != digest for name, digest in exported.items()):
                raise ValueError("Artefato final alterado; preservar e revisar")
            log(out, "Retomada: cache e artefatos completos integros; nenhuma gravacao reextraida.")
            return
        control_file = out / "CONTROLE_REFERENCIA_48KHZ.csv"
        control_identity = out / "CONTROLE_VALIDADO.json"
        if control_file.exists():
            if not control_identity.exists():
                raise ValueError("Controle48k parcial sem hash; preservar e revisar")
            recorded = json.loads(control_identity.read_text(encoding="utf-8"))
            if recorded != {"csv_sha256": digest_file(control_file), "signature": signature}:
                raise ValueError("Controle48k alterado ou de outra configuracao")
            control = pd.read_csv(control_file).to_dict("records")
        else:
            if control_identity.exists():
                raise ValueError("Controle48k parcial sem CSV; preservar e revisar")
            control = reference_check(root, meta, reference, out, batch_size)
            atomic_text(control_identity, json_text({"csv_sha256": digest_file(control_file), "signature": signature}))
        expected_controls = set(meta.iloc[np.unique(np.linspace(0, len(meta) - 1, 3, dtype=int))].sample_id)
        if len(control) != 3 or {r["sample_id"] for r in control} != expected_controls or not all(r["equivalent"] for r in control):
            raise ValueError("Controle48k ausente/incompleto/reprovado")
        log(out, f"Inicio/retomada {ident['mode']}: {len(selected)} alvos, {len(found)} em cache.")
        for row in selected.itertuples(index=False):
            if row.sample_id in found:
                continue
            one_start = time.perf_counter()
            try:
                signal, audio_diag = read_signal48(root, row, opened)
                vector, diag = extract_signal48(signal, TARGET_SR, batch_size)
                diag.update(audio_diag)
                diag["extraction_seconds"] = time.perf_counter() - one_start
                blob = vector.tobytes()
                with conn:
                    conn.execute("INSERT INTO features VALUES (?,?,?,?,?,?)", (row.sample_id, row.participant_id, row.task,
                                 blob, hashlib.sha256(blob).hexdigest(), json_text(diag)))
                    conn.execute("DELETE FROM errors WHERE sample_id=?", (row.sample_id,))
                found[row.sample_id] = (vector, diag)
                new_count += 1
                work_seconds += diag["extraction_seconds"]
            except Exception as exc:
                with conn:
                    conn.execute("INSERT OR REPLACE INTO errors VALUES (?,?)", (row.sample_id, repr(exc)))
                log(out, f"ERRO {row.sample_id}: {exc!r}; interrompido sem exportar matriz incompleta.")
                raise
            if pilot or len(found) % 25 == 0 or len(found) == len(selected):
                log(out, f"{len(found)}/{len(selected)}; ultimo={row.sample_id}; segundos_ultimo={time.perf_counter()-one_start:.3f}")
        if conn.execute("SELECT COUNT(*) FROM errors").fetchone()[0]:
            raise ValueError("Ainda ha erros no cache; nao exportar")
        verified = validate_cache(conn, selected)
        export_complete(out, selected, verified, names, base, time.perf_counter() - started, work_seconds, new_count, pilot, control)
        log(out, "Exportacao completa e finita; nenhum participante excluido.")
    finally:
        conn.close()
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
        import test_wpt_extracao18 as tests
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    if not 1 <= args.batch_janelas <= 512:
        parser.error("--batch-janelas deve estar entre1 e512")
    if args.piloto is not None and not 1 <= args.piloto <= 10:
        parser.error("Piloto deve ter entre1 e10 gravacoes")
    root = args.dados.resolve()
    meta, reference, names, base, paths, archives = load_inputs(root)
    pilot = args.piloto is not None
    selected = meta.iloc[np.unique(np.linspace(0, len(meta) - 1, args.piloto, dtype=int))] if pilot else meta
    mode = f"piloto_{len(selected)}" if pilot else "completo"
    ident = identity(paths, archives, selected, mode, args.batch_janelas)
    print(f"ENTRADAS OK:2465 gravacoes/411pessoas; WPTsym5=1920; modo={mode}; nenhum WAV lido na verificacao.", flush=True)
    if not (pilot or args.executar):
        return
    default = root / "results/experiments" / (f"resultados_18_wpt_16khz_piloto_{len(selected)}" if pilot else DEFAULT_OUTPUT.name)
    out = (args.saida or default).resolve()
    if not out.is_relative_to(root / "results/experiments") or out == root / "results/experiments":
        parser.error("Saida precisa ser subpasta de results/experiments")
    if pilot and out == root / "results/experiments" / DEFAULT_OUTPUT.name:
        parser.error("Piloto nao pode escrever na pasta completa")
    run(root, out, meta, reference, names, base, ident, pilot, args.batch_janelas)


if __name__ == "__main__":
    main()
