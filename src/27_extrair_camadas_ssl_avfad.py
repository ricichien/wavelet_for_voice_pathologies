# -*- coding: utf-8 -*-
"""Exp27: camadas 5/12/16 do XLS-R congelado; preparação não importa Torch.

Não altera o experimento20. --verificar apenas lê; --autoteste usa dados
sintéticos; --piloto/--executar exigem o término da fila25–26 antes da GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sqlite3
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp20_for_exp27", HERE / "20_extrair_wav2vec_avfad.py")
EXP20 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP20)
EXP18 = EXP20.EXP18
VERSION = "27.1"
MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
LAYERS = (5, 12, 16)
CONTROL_LAYER = 24
WIDTH, DIM = EXP20.HIDDEN_SIZE, EXP20.DIM
TOTAL_DIM = len(LAYERS) * DIM
CONTROL_ATOL, CONTROL_RTOL = 2e-5, 2e-5
OUTPUT_NAME = "resultados_27_camadas_ssl_extracao"
REFERENCE20 = Path("results/experiments/resultados_20_wav2vec_extracao")
EXPORTS = ("features_wav2vec_camadas.npz", "metadata_wav2vec_camadas.csv",
           "CONTROLE_REFERENCIA_20.csv", "assinatura_execucao.json", "PROTOCOLO.txt", "RESUMO_EXTRACAO.txt")
DIAG_REQUIRED = {"hidden_frames", "control24_allclose", "control24_exact", "control24_max_abs_error",
                 "control24_mean_abs_error", "control24_reference_frames", "chunk_count", "vad_fallback"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def feature_names(layer):
    require(layer in LAYERS, "Camada não predefinida")
    return np.asarray([f"WAV2VEC__xlsr300m__layer{layer:02d}__{stat}__{i:04d}"
                       for stat in ("mean", "std") for i in range(WIDTH)])


def validate_reference(meta, reference_meta, matrix, embedded, names, identity, real=True):
    config = identity["configuration"]
    require(identity.get("mode") == "completo", "Referência20 deve ser a extração completa")
    require(config.get("model_id") == EXP20.MODEL_ID and config.get("model_revision") == MODEL_REVISION,
            "Referência20 usa outro checkpoint/revisão")
    require(config.get("frozen") is True and config.get("hidden_layer") == "last" and
            config.get("pooling") == "mean_std" and config.get("feature_dimension") == DIM and
            config.get("num_hidden_layers") == CONTROL_LAYER, "Contrato da referência20 mudou")
    require(config == EXP20.configuration(MODEL_REVISION, config.get("use_safetensors"), config.get("device")),
            "Configuração20 não corresponde ao pipeline congelado reutilizado")
    require(matrix.shape == (len(meta), DIM) and matrix.dtype == np.float32 and np.isfinite(matrix).all(),
            "Embeddings20 inválidos")
    require(np.array_equal(names, EXP20.feature_names()), "Nomes da referência20 mudaram")
    for key in ("sample_id", "participant_id", "task", "y"):
        require(key in reference_meta and reference_meta[key].notna().all() and
                np.array_equal(meta[key].astype(str), reference_meta[key].astype(str)), f"Metadata20 diverge: {key}")
    for key in ("sample_id", "participant_id", "task"):
        require(key in embedded and np.array_equal(embedded[key].astype(str), meta[key].astype(str)),
                f"IDs NPZ20 divergem: {key}")
    require(not meta.sample_id.duplicated().any() and not reference_meta.sample_id.duplicated().any(), "IDs duplicados")
    require(meta.groupby("participant_id").y.nunique().eq(1).all(), "Rótulos divergentes por pessoa")
    require(np.array_equal(reference_meta.matrix_row, np.arange(len(meta))), "matrix_row20 incorreto")
    require(identity.get("sample_ids") == meta.sample_id.tolist(), "Manifesto20 diverge da coorte")
    if real:
        people = meta.drop_duplicates("participant_id")
        require((len(meta), len(people), people.y.value_counts().to_dict()) == (2465, 411, {1: 329, 0: 82}),
                "Coorte diferente de HSA411/2465/82/329")


def load_inputs(root, data20):
    """Lê somente vetores, metadados, hashes e ZIP stats; não decodifica WAV."""
    meta, _, _, _, paths, archives = EXP18.load_inputs(root)
    marker = json.loads((data20 / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    require(set(marker) == set(EXP20.EXPORTS), "Referência20 sem todos os exports completos")
    for name, digest in marker.items():
        require(Path(name).name == name and EXP20.sha(data20 / name) == digest, f"Export20 alterado: {name}")
    identity20 = json.loads((data20 / "assinatura_execucao.json").read_text(encoding="utf8"))
    for category in ("code",):
        for name, digest in identity20[category].items():
            require(EXP20.sha(name) == digest, f"Fonte20 alterada: {name}")
    for name, info in identity20["inputs"].items():
        path = Path(name)
        require(path.stat().st_size == info["size_bytes"] and EXP20.sha(path) == info["sha256"], f"Insumo20 alterado: {name}")
    require(set(identity20["raw_zip_metadata_only"]) == {str(path.resolve()) for path in archives},
            "ZIPs27 diferem da referência20")
    for path in archives:
        original = identity20["raw_zip_metadata_only"][str(path.resolve())]
        require(path.stat().st_size == original["size_bytes"] and path.stat().st_mtime_ns == original["mtime_ns"],
                f"ZIP original mudou desde20: {path.name}")
    reference_meta = pd.read_csv(data20 / "metadata_wav2vec.csv", dtype={"sample_id": str, "participant_id": str, "task": str},
                                 float_precision="round_trip")
    reference_meta["task"] = reference_meta.task.str.zfill(3)
    with np.load(data20 / "features_wav2vec_xlsr300m.npz", allow_pickle=False) as z:
        matrix = z["embeddings"]
        embedded = {key: z[key] for key in ("sample_id", "participant_id", "task")}
        validate_reference(meta, reference_meta, matrix, embedded, z["feature_names"].astype(str), identity20)
    reference = {sid: (matrix[i].copy(), reference_meta.iloc[i].to_dict()) for i, sid in enumerate(meta.sample_id)}
    paths = tuple(paths) + tuple(data20 / name for name in (*EXP20.EXPORTS, "ARTEFATOS_COMPLETOS.json"))
    return meta, reference, paths, archives, identity20


def configuration(revision, use_safe, device):
    config = EXP20.configuration(revision, use_safe, device)
    config.update(hidden_layer="multiple_intermediate", hidden_layers=list(LAYERS),
                  feature_dimension=TOTAL_DIM, feature_dimension_per_layer=DIM,
                  hidden_state_indexing="HF hidden_states[k]: after k blocks; 0 before block1; 24 after final LayerNorm",
                  intermediate_final_layer_norm=False, control_layer=CONTROL_LAYER,
                  control_reference="complete_exp20_same_recording", control_atol=CONTROL_ATOL,
                  control_rtol=CONTROL_RTOL, output_hidden_states=True,
                  layer_selection="none_in_extraction; fixed5_12_16_before_inference")
    return config


def identity(paths, archives, selected, mode, model_dir, files, use_safe, device, data20):
    ident = EXP20.identity(paths, archives, selected, mode, model_dir, MODEL_REVISION, files, use_safe, device)
    ident["version"] = VERSION
    ident["configuration"] = configuration(MODEL_REVISION, use_safe, device)
    ident["code"][str(Path(__file__).resolve())] = EXP20.sha(__file__)
    ident["reference20_directory"] = str(data20)
    return ident


def protocol(ident):
    return f"""EXPERIMENTO 27 — CAMADAS INTERMEDIÁRIAS XLS-R CONGELADAS
Modo: {ident['mode']}. Extração sem treinamento, busca, exclusões ou uso de rótulos.
Hipótese a avaliar depois: camadas intermediárias podem discriminar a voz melhor
que o último estado contextual. Nenhuma melhoria é presumida nesta extração.
Coorte completa: 411 participantes HSA, 82 normais/329 patológicos, 2.465 frases.
Checkpoint: {EXP20.MODEL_ID}, revisão {MODEL_REVISION}; somente arquivos locais.
Camadas FIXAS 5, 12 e 16, declaradas antes de inferência na AVFAD neste ciclo.
Índices são os de output.hidden_states do Transformers4.57.6, encoder StableLayerNorm:
índice0 é entrada contextual após posição/dropout, antes do bloco1; índice k<24
é saída após k blocos, antes do próximo, sem a LayerNorm final do encoder.
Índice24 é saída após24 blocos e LayerNorm final, igual a last_hidden_state.
Não renomear índice5 como o sexto bloco; não aplicar LayerNorm adicional.
Cada camada exporta 1.024 médias e 1.024 DPs populacionais: 2.048 atributos;
três matrizes alinhadas, total lógico 6.144 atributos por gravação.
Downmix/DC, VAD48k, reamostragem e chunks reutilizam exatamente o experimento20:
VAD40dB/frame2048/hop480/unir250ms/mínimo250ms; resample_poly1/3 por região;
concatenar regiões, ceil(N/128000) chunks quase iguais <=8s, sem padding/overlap/truncamento.
Preprocessor do checkpoint normaliza cada chunk. Encoder eval/frozen/inference_mode,
float32, eager, TF32 off, algoritmos determinísticos; pesos e versões assinados.
Média/DP combinadas por Welfordfloat64 ponderando TODOS os frames, não os chunks.
Última camada24 é calculada apenas para controle em TODAS as gravações contra
o cache20 completo: mesmos IDs, frames, chunks e áudio preparado; tolerâncias
pré-fixadas rtol={CONTROL_RTOL}, atol={CONTROL_ATOL}; registrar erro e exatidão.
Não usar o controle24 recém-calculado como um novo resultado classificatório.
Camada24 continua disponível no cache20 para uma comparação subsequente.
As camadas não são escolhidas por desempenho do teste. Seleção/escala/classificação
deverão acontecer somente nos folds de treino no experimento posterior.
Piloto determinístico em pasta própria; nenhum WAV será lido por --verificar.
--piloto/--executar recusam início enquanto a fila25–26 não estiver completa.
SQLite próprio com assinatura, hashes de vetores/diagnósticos e retomada por gravação.
Falha preserva os dados e não publica matriz final parcial; fontes20–26 intocadas.
AVFAD explorada é desenvolvimento, não teste cego; nenhuma BA ou validação SVD aqui.
"""


def previous_campaign_complete(root):
    path = root / "results/experiments/campanha_25_26_execucao/STATUS.json"
    require(path.is_file(), "Fila25–26 ainda sem estado final; nenhuma inferência27 iniciada")
    status = json.loads(path.read_text(encoding="utf8"))
    require(status.get("phase") == "completo", f"Fila25–26 ainda não completa: {status.get('phase')}")
    for number, folder in ((25, "resultados_25_wavelet_cnn_tcn"), (26, "resultados_26_hibrido_wavelet_temporal_wav2vec")):
        marker = root / "results/experiments" / folder / "ARTEFATOS_COMPLETOS.json"
        require(marker.is_file() and EXP20.sha(marker) == status.get(f"marker{number}_sha256"),
                f"Conclusão{number} não corresponde ao estado final da fila")
        for name, digest in json.loads(marker.read_text(encoding="utf8")).items():
            require(Path(name).name == name and EXP20.sha(marker.parent / name) == digest, f"Export{number} alterado")
    return status


def hidden_tensors(output):
    states = output.hidden_states
    require(isinstance(states, (tuple, list)) and len(states) == CONTROL_LAYER + 1, "Esperava25 hidden_states (0..24)")
    selected = {layer: states[layer] for layer in (*LAYERS, CONTROL_LAYER)}
    shape = output.last_hidden_state.shape
    require(len(shape) == 3 and shape[0] == 1 and shape[2] == WIDTH and shape[1] > 0, "Última camada fora do contrato")
    require(all(value.shape == shape for value in selected.values()), "Camadas com shapes incompatíveis")
    return selected


def infer_vectors(chunks, torch, processor, model, device):
    moments = {layer: EXP20.FrameMoments() for layer in (*LAYERS, CONTROL_LAYER)}
    with torch.inference_mode():
        for chunk in chunks:
            batch = processor(chunk, sampling_rate=EXP20.SAMPLE_RATE, return_tensors="pt",
                              return_attention_mask=True, padding=False)
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch, output_hidden_states=True, return_dict=True)
            tensors = hidden_tensors(output)
            for layer, hidden in tensors.items():
                frames = hidden[0].cpu().numpy()
                moments[layer].update(frames)
                if layer == CONTROL_LAYER:
                    last = output.last_hidden_state[0].cpu().numpy()
                    require(np.array_equal(frames, last), "hidden_states[24] não equivale exatamente a last_hidden_state")
    counts = {value.count for value in moments.values()}
    require(len(counts) == 1 and next(iter(counts)) > 0, "Quantidade de frames diferente entre camadas")
    return np.stack([moments[layer].vector() for layer in LAYERS]), moments[CONTROL_LAYER].vector(), next(iter(counts))


def check_reference20(actual, expected, diag, reference_diag):
    require(actual.shape == expected.shape == (DIM,) and np.isfinite(actual).all() and np.isfinite(expected).all(),
            "Controle24 com vetor inválido")
    require(np.allclose(actual, expected, rtol=CONTROL_RTOL, atol=CONTROL_ATOL), "Controle24 diverge dos embeddings20")
    for field in ("hidden_frames", "chunk_count", "target_active_samples", "speech_regions", "vad_fallback",
                  "original_sample_rate_hz", "channels_original"):
        require(diag[field] == reference_diag[field], f"Controle20 difere em {field}")
    for field in ("active_duration_s", "recording_duration_s", "speech_ratio"):
        require(abs(diag[field] - reference_diag[field]) <= 1e-12, f"Preparação de áudio20 difere em {field}")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"control24_allclose": True, "control24_exact": bool(np.array_equal(actual, expected)),
            "control24_max_abs_error": float(difference.max()), "control24_mean_abs_error": float(difference.mean()),
            "control24_reference_frames": int(reference_diag["hidden_frames"])}


def init_cache(path, signature):
    exists = path.exists()
    con = sqlite3.connect(f"file:{path.as_posix()}?mode={'rw' if exists else 'rwc'}", uri=True)
    con.execute("PRAGMA synchronous=FULL")
    if exists:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = [row[1] for row in con.execute("PRAGMA table_info(features)")]
        if tables != {"config", "features", "errors"} or columns != ["sample_id", "participant_id", "task", "vectors", "vectors_sha256", "diagnostics", "diagnostics_sha256"]:
            con.close(); raise ValueError("SchemaSQLite27 desconhecido; preservar")
    else:
        con.execute("CREATE TABLE config (signature TEXT NOT NULL)")
        con.execute("INSERT INTO config VALUES (?)", (signature,))
        con.execute("CREATE TABLE features (sample_id TEXT PRIMARY KEY, participant_id TEXT NOT NULL, task TEXT NOT NULL, vectors BLOB NOT NULL, vectors_sha256 TEXT NOT NULL, diagnostics TEXT NOT NULL, diagnostics_sha256 TEXT NOT NULL)")
        con.execute("CREATE TABLE errors (sample_id TEXT PRIMARY KEY, error TEXT NOT NULL)")
        con.commit()
    if con.execute("SELECT signature FROM config").fetchall() != [(signature,)]:
        con.close(); raise ValueError("AssinaturaSQLite27 diferente; preservar")
    return con


def validate_cache(con, selected):
    expected = selected.set_index("sample_id")
    require(not expected.index.duplicated().any(), "Manifesto27 duplicado")
    found = {}
    for sid, pid, task, blob, digest, diagnostic_text, diagnostic_digest in con.execute("SELECT * FROM features"):
        require(sid in expected.index and str(expected.loc[sid, "participant_id"]) == pid and
                str(expected.loc[sid, "task"]) == task, "IDsSQLite27 divergentes")
        require(hashlib.sha256(blob).hexdigest() == digest and
                hashlib.sha256(diagnostic_text.encode("utf8")).hexdigest() == diagnostic_digest, f"HashSQLite27 corrompido: {sid}")
        values = np.frombuffer(blob, dtype=np.float32)
        require(values.shape == (TOTAL_DIM,) and np.isfinite(values).all(), f"Vetor27 inválido: {sid}")
        diag = json.loads(diagnostic_text)
        require(DIAG_REQUIRED.issubset(diag) and diag["control24_allclose"] is True and
                isinstance(diag["control24_exact"], bool) and diag["hidden_frames"] == diag["control24_reference_frames"] and
                diag["hidden_frames"] > 0 and diag["chunk_count"] > 0, "Diagnósticos27 inválidos")
        require(all(np.isfinite(diag[key]) and diag[key] >= 0 for key in ("control24_max_abs_error", "control24_mean_abs_error")),
                "Erro de controle27 não finito/negativo")
        found[sid] = (values.reshape(len(LAYERS), DIM).copy(), diag)
    require(all(sid in expected.index for sid, in con.execute("SELECT sample_id FROM errors")), "ErroSQLite fora do manifesto")
    return found


def store(con, row, values, diag):
    require(values.shape == (len(LAYERS), DIM) and values.dtype == np.float32 and np.isfinite(values).all(), "Vetor27 inválido")
    blob = values.tobytes()
    diagnostic_text = EXP20.json_text(diag)
    with con:
        con.execute("INSERT INTO features VALUES (?,?,?,?,?,?,?)",
                    (row.sample_id, row.participant_id, row.task, blob, hashlib.sha256(blob).hexdigest(),
                     diagnostic_text, hashlib.sha256(diagnostic_text.encode("utf8")).hexdigest()))
        con.execute("DELETE FROM errors WHERE sample_id=?", (row.sample_id,))


def export(out, selected, found, ident, elapsed, new_count, runtime):
    require(set(found) == set(selected.sample_id), "Nenhuma exportação27 de matriz parcial")
    meta = selected.copy().reset_index(drop=True)
    diagnostics = pd.DataFrame([found[sid][1] for sid in meta.sample_id])
    for column in diagnostics:
        meta[column] = diagnostics[column]
    meta["matrix_row"] = np.arange(len(meta))
    values = np.stack([found[sid][0] for sid in meta.sample_id])
    payload = {key: meta[key].to_numpy(dtype=str) for key in ("sample_id", "participant_id", "task")}
    payload["hidden_layer_indices"] = np.asarray(LAYERS, dtype=np.int16)
    for index, layer in enumerate(LAYERS):
        payload[f"embeddings_layer{layer:02d}"] = values[:, index]
        payload[f"feature_names_layer{layer:02d}"] = feature_names(layer)
    stream = io.BytesIO(); np.savez_compressed(stream, **payload)
    temporary = out / (EXPORTS[0] + ".part")
    temporary.write_bytes(stream.getvalue()); temporary.replace(out / EXPORTS[0])
    EXP18.atomic_csv(out / EXPORTS[1], meta)
    control_columns = ["sample_id", "participant_id", "task", "hidden_frames", "control24_reference_frames",
                       "control24_allclose", "control24_exact", "control24_max_abs_error", "control24_mean_abs_error"]
    EXP18.atomic_csv(out / EXPORTS[2], meta[control_columns])
    lines = ["EXPERIMENTO 27 — CAMADAS INTERMEDIÁRIAS XLS-R CONGELADAS", f"Concluído: {EXP18.timestamp()}",
             f"Modo: {ident['mode']}; camadas: {LAYERS}; shapes: {values.shape}; pessoas: {meta.participant_id.nunique()}",
             f"Controle24/20: {int(meta.control24_allclose.sum())}/{len(meta)} aprovados; exatos: {int(meta.control24_exact.sum())}",
             f"Maior erro absoluto controle24: {meta.control24_max_abs_error.max()}",
             f"Frames: {int(meta.hidden_frames.sum())}; novas gravações nesta chamada: {new_count}; segundos: {elapsed:.3f}",
             f"Runtime: {json.dumps(runtime, ensure_ascii=False, sort_keys=True)}",
             "Três matrizes float32 alinhadas por IDs; nenhuma seleção de camada ou classificação executada.",
             "Média/DP por todos os frames; sem exclusões, padding, sobreposição ou truncamento.",
             "Fontes/modelo/insumos assinados, checkpoint próprio e controle de reprodução do experimento20.",
             "AVFAD é desenvolvimento; nenhum resultado clínico, BA ou validação SVD nesta extração.", ""]
    EXP20.atomic_text(out / EXPORTS[-1], "\n".join(lines))
    EXP20.atomic_text(out / "ARTEFATOS_COMPLETOS.json", EXP20.json_text({name: EXP20.sha(out / name) for name in EXPORTS}))


def run(root, out, selected, reference, ident, model_dir, use_safe, device):
    previous_campaign_complete(root)
    text = EXP20.json_text(ident)
    signature = hashlib.sha256(text.encode("utf8")).hexdigest()
    signature_file = out / "assinatura_execucao.json"
    if signature_file.exists():
        require(signature_file.read_text(encoding="utf8") == text, "Configuração27 mudou; preservar pasta")
    else:
        require(not out.exists() or not any(out.iterdir()), "Pasta27 não vazia sem assinatura; preservar")
        out.mkdir(parents=True, exist_ok=True)
        EXP20.atomic_text(signature_file, text)
        EXP20.atomic_text(out / "PROTOCOLO.txt", protocol(ident))
    require((out / "PROTOCOLO.txt").read_text(encoding="utf8") == protocol(ident), "Protocolo27 alterado")
    con = init_cache(out / "checkpoint_camadas_ssl.sqlite", signature)
    opened, new_count = {}, 0
    started = time.perf_counter()
    try:
        found = validate_cache(con, selected)
        marker = out / "ARTEFATOS_COMPLETOS.json"
        if marker.exists():
            exports = json.loads(marker.read_text(encoding="utf8"))
            require(set(exports) == set(EXPORTS) and set(found) == set(selected.sample_id) and
                    con.execute("SELECT COUNT(*) FROM errors").fetchone()[0] == 0, "Conclusão27 inconsistente")
            require(all(EXP20.sha(out / name) == digest for name, digest in exports.items()), "Export27 final alterado")
            EXP20.log(out, "Retomada27: todos os vetores/diagnósticos/exports íntegros; nenhuma reextração.")
            return
        torch, processor, model, runtime = EXP20.load_encoder(model_dir, use_safe, device)
        EXP20.log(out, f"Início/retomada27: {ident['mode']}; {len(found)}/{len(selected)}; camadas={LAYERS}; GPU={runtime['gpu']}")
        for row in selected.itertuples(index=False):
            if row.sample_id in found:
                continue
            one_start = time.perf_counter()
            try:
                signal, audio_diag = EXP18.read_signal48(root, row, opened)
                chunks, diag = EXP20.prepare_signal(signal)
                vectors, control, frames = infer_vectors(chunks, torch, processor, model, device)
                diag.update(audio_diag)
                diag.update(hidden_frames=frames)
                expected, reference_diag = reference[row.sample_id]
                diag.update(check_reference20(control, expected, diag, reference_diag))
                diag["extraction_seconds"] = time.perf_counter() - one_start
                store(con, row, vectors, diag)
                found[row.sample_id] = (vectors, diag)
                new_count += 1
            except Exception as exc:
                with con:
                    con.execute("INSERT OR REPLACE INTO errors VALUES (?,?)", (row.sample_id, repr(exc)))
                EXP20.log(out, f"ERRO27: {row.sample_id}: {exc!r}; nenhuma exportação final parcial")
                raise
            if ident["mode"] != "completo" or len(found) % 25 == 0 or len(found) == len(selected):
                EXP20.log(out, f"{len(found)}/{len(selected)}; último={row.sample_id}; segundos={diag['extraction_seconds']:.3f}; controle24erro={diag['control24_max_abs_error']}")
        require(con.execute("SELECT COUNT(*) FROM errors").fetchone()[0] == 0, "Cache27 ainda contém erros")
        export(out, selected, validate_cache(con, selected), ident, time.perf_counter() - started, new_count, runtime)
        EXP20.log(out, "Extração27 concluída; todas as gravações preservadas; controle24 aprovado.")
    finally:
        con.close()
        for archive in opened.values():
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--dados20", type=Path)
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--modelo", type=Path)
    parser.add_argument("--revisao", default=MODEL_REVISION)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verificar", action="store_true")
    actions.add_argument("--autoteste", action="store_true")
    actions.add_argument("--piloto", type=int, nargs="?", const=3)
    actions.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        import unittest
        spec = importlib.util.spec_from_file_location("test_camadas_ssl27", HERE / "test_camadas_ssl27.py")
        tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful():
            raise SystemExit(1)
        return
    if args.revisao != MODEL_REVISION or not re.fullmatch(r"[0-9a-f]{40}", args.revisao):
        parser.error("Revisão27 é fixa; nenhum download automático")
    if args.piloto is not None and not 1 <= args.piloto <= 10:
        parser.error("Piloto deve ter1..10 gravações")
    root = args.dados.resolve()
    data20 = (args.dados20 or root / REFERENCE20).resolve()
    model_dir = (args.modelo or root / "models/wav2vec2-xls-r-300m" / MODEL_REVISION).resolve()
    meta, reference, paths, archives, identity20 = load_inputs(root, data20)
    files, use_safe = EXP20.model_files(model_dir, MODEL_REVISION)
    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf8"))
    require(model_config.get("do_stable_layer_norm") is True, "Índices27 exigem o encoder StableLayerNorm auditado")
    pilot = args.piloto is not None
    selected = meta.iloc[np.unique(np.linspace(0, len(meta)-1, args.piloto, dtype=int))].copy() if pilot else meta.copy()
    mode = f"piloto_{len(selected)}" if pilot else "completo"
    ident = identity(paths, archives, selected, mode, model_dir, files, use_safe, args.device, data20)
    print(f"ENTRADAS27 OK: 411 pessoas/2465 gravações; selecionadas={len(selected)}; camadas={LAYERS}; 3x2048 atributos.", flush=True)
    if not (args.executar or pilot):
        print("Somente verificação: nenhum WAV lido, Torch importado, inferência, download ou escrita.", flush=True)
        return
    for package in ("numpy", "pandas", "scipy", "librosa", "soundfile", "torch", "transformers"):
        require(ident["environment"].get(package) == identity20["environment"].get(package) and
                ident["environment"].get(package) is not None,
                f"Versão {package} diferente da referência20; usar .venv_wav2vec preservada")
    default = root / "results/experiments" / (f"resultados_27_camadas_ssl_piloto_{len(selected)}" if pilot else OUTPUT_NAME)
    out = (args.saida or default).resolve()
    if not out.is_relative_to(root / "results/experiments") or out == root / "results/experiments":
        parser.error("Saída deve ser subpasta de results/experiments")
    if pilot and out == root / "results/experiments" / OUTPUT_NAME:
        parser.error("Piloto não pode escrever na pasta completa")
    run(root, out, selected, reference, ident, model_dir, use_safe, args.device)


if __name__ == "__main__":
    main()
