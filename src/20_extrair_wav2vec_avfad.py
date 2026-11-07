# -*- coding: utf-8 -*-
"""Exp20: embeddings XLS-R300M congelados; --verificar nao le WAV nem importa torch."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import re
import sqlite3
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp18_for_exp20", HERE / "18_extrair_wpt_16khz_avfad.py")
EXP18 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP18)
LEGACY = EXP18.LEGACY
VERSION = "20.1"
MODEL_ID = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE, HIDDEN_SIZE, DIM = 16000, 1024, 2048
MAX_CHUNK = 8 * SAMPLE_RATE
MIN_INPUT = 400  # receptive field da pilha convolucional XLS-R
OUTPUT_NAME = "resultados_20_wav2vec_extracao"
EXPORTS = ("features_wav2vec_xlsr300m.npz", "metadata_wav2vec.csv",
           "assinatura_execucao.json", "RESUMO_EXTRACAO.txt")


def sha(path):
    return EXP18.digest_file(Path(path))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(text.encode("utf-8"))
    temporary.replace(path)


def log(out, value):
    EXP18.log(out, value)


def feature_names():
    return np.asarray([f"WAV2VEC__xlsr300m__{stat}__{i:04d}"
                       for stat in ("mean", "std") for i in range(HIDDEN_SIZE)])


def split_chunks(signal, maximum=MAX_CHUNK):
    """Particiona todas as amostras em blocos quaseiguais <=8s, sem padding."""
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1 or len(x) < MIN_INPUT or not np.isfinite(x).all():
        raise ValueError("Sinal deve ser mono, finito e ter pelo menos400amostras")
    if maximum < 2 * MIN_INPUT:
        raise ValueError("Tamanho maximo de chunk insuficiente")
    count = (len(x) + maximum - 1) // maximum
    width, extra = divmod(len(x), count)
    if width < MIN_INPUT:
        raise ValueError("Particao produziria chunk menor que receptive field")
    chunks, start = [], 0
    for index in range(count):
        end = start + width + int(index < extra)
        chunks.append(x[start:end])
        start = end
    if start != len(x) or any(len(c) > maximum for c in chunks):
        raise AssertionError("Particao invalida")
    return chunks


class FrameMoments:
    """Media/DP populacional ponderados pelos frames, com combinacao de Welford."""
    def __init__(self, width=HIDDEN_SIZE):
        self.count = 0
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, frames):
        x = np.asarray(frames, dtype=np.float64)
        if x.ndim != 2 or x.shape[1:] != self.mean.shape or len(x) == 0 or not np.isfinite(x).all():
            raise ValueError("Hidden states invalidos")
        n = len(x)
        mean = x.mean(axis=0)
        m2 = np.sum((x - mean) ** 2, axis=0)
        total = self.count + n
        delta = mean - self.mean
        self.m2 += m2 + delta ** 2 * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total

    def vector(self):
        if self.count == 0:
            raise ValueError("Nenhum frame para resumir")
        return np.concatenate((self.mean, np.sqrt(np.maximum(self.m2 / self.count, 0.0)))).astype(np.float32)


def prepare_signal(signal48):
    regions48, fallback = LEGACY.detect_speech_regions(signal48, 48000)
    regions16 = [resample_poly(region, 1, 3).astype(np.float32) for region in regions48]
    signal16 = np.concatenate(regions16)
    chunks = split_chunks(signal16)
    active = sum(map(len, regions48)) / 48000
    return chunks, {"speech_regions": len(regions48), "active_duration_s": active,
                    "recording_duration_s": len(signal48) / 48000,
                    "speech_ratio": active / (len(signal48) / 48000),
                    "vad_fallback": int(fallback), "source_processing_sr": 48000,
                    "sample_rate_hz": SAMPLE_RATE, "target_active_samples": len(signal16),
                    "chunk_count": len(chunks), "chunk_min_samples": min(map(len, chunks)),
                    "chunk_max_samples": max(map(len, chunks)), "padding_samples": 0}


def model_files(model_dir, revision):
    required = [model_dir / "config.json", model_dir / "preprocessor_config.json"]
    if any(not p.is_file() for p in required):
        raise FileNotFoundError("Modelo local precisa de config.json e preprocessor_config.json")
    weights = sorted(model_dir.glob("*.safetensors"))
    use_safe = bool(weights)
    if not weights:
        weights = sorted(model_dir.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError("Pesos locais ausentes; este script nunca baixa arquivos")
    indices = sorted(model_dir.glob("*.index.json"))
    config = json.loads(required[0].read_text(encoding="utf-8"))
    pre = json.loads(required[1].read_text(encoding="utf-8"))
    if config.get("model_type") != "wav2vec2" or config.get("hidden_size") != HIDDEN_SIZE or config.get("num_hidden_layers") != 24:
        raise ValueError("Esperava arquitetura XLS-R300M: wav2vec2, hidden1024 e24camadas")
    if pre.get("sampling_rate") != SAMPLE_RATE or pre.get("do_normalize") is not True:
        raise ValueError("Preprocessor deve usar16kHz e normalizacao por waveform")
    provenance_file = model_dir / "ORIGEM_MODELO.json"
    if not provenance_file.is_file():
        raise FileNotFoundError("ORIGEM_MODELO.json ausente; usar preparador com revisao fixada")
    provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
    if provenance.get("model_id") != MODEL_ID or provenance.get("model_revision") != revision:
        raise ValueError("Modelo/revisaoCLI nao correspondem a ORIGEM_MODELO.json")
    recorded = provenance.get("files", {})
    needed = {p.name for p in required + weights + indices}
    if not isinstance(recorded, dict) or not needed.issubset(recorded):
        raise ValueError("Manifesto de origem nao cobre configuracao/pesos utilizados")
    for name, info in recorded.items():
        if not isinstance(name, str) or Path(name).name != name or name == provenance_file.name:
            raise ValueError("Nome de arquivo invalido no manifesto de origem")
        target = model_dir / name
        if not target.is_file() or target.stat().st_size != info.get("size_bytes") or sha(target) != info.get("sha256"):
            raise ValueError(f"Arquivo local diverge da origem registrada:{name}")
    # Inclui README/origem, alem dos pesos, para vincular todo o download
    # verificado pelo preparador e nao apenas a arquitetura declarada.
    return [model_dir / name for name in sorted(recorded)] + [provenance_file], use_safe


def configuration(revision, use_safe, device):
    return {"model_id": MODEL_ID, "model_revision": revision, "sample_rate": SAMPLE_RATE,
            "frozen": True, "hidden_layer": "last", "hidden_size": HIDDEN_SIZE,
            "pooling": "mean_std", "population_std_ddof": 0, "pool_accumulator_dtype": "float64",
            "output_dtype": "float32", "feature_dimension": DIM, "num_hidden_layers": 24,
            "source_processing_sr": 48000, "vad": [40, 2048, 480, 0.25, 0.25],
            "resample": "scipy.signal.resample_poly; cada regiao separadamente; up1_down3",
            "chunking": "concatenar regioes ativas; ceil(N/128000) blocos quaseiguais",
            "max_chunk_samples": MAX_CHUNK, "padding": False, "overlap": False,
            "truncation": False, "waveform_normalization": "preprocessor por chunk: media0,variancia1",
            "attention_implementation": "eager", "inference_dtype": "float32", "tf32": False,
            "deterministic_algorithms": True, "use_safetensors": use_safe, "device": device,
            "trust_remote_code": False, "local_files_only": True, "weights_only": True}


def identity(paths, archives, selected, mode, model_dir, revision, files, use_safe, device):
    code = [Path(__file__).resolve(), Path(EXP18.__file__).resolve(), Path(LEGACY.__file__).resolve(), HERE / "avfad_paths.py"]
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for package in ("numpy", "pandas", "scipy", "PyWavelets", "librosa", "soundfile", "torch", "transformers", "safetensors"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    return {"version": VERSION, "mode": mode, "sample_ids": selected.sample_id.tolist(),
            "configuration": configuration(revision, use_safe, device), "environment": environment,
            "inputs": {str(p.resolve()): {"sha256": sha(p), "size_bytes": p.stat().st_size} for p in paths},
            "code": {str(p.resolve()): sha(p) for p in code},
            "model_directory": str(model_dir), "model_files": {p.name: sha(p) for p in files},
            "raw_zip_metadata_only": {str(p.resolve()): {"size_bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in archives}}


def protocol(ident):
    return f"""EXPERIMENTO20 - XLS-R300M CONGELADO; modo={ident['mode']}
Sem treinamento, rotulos como entrada, selecao supervisionada ou exclusoes.
Coorte alvo completa:2465gravacoes/411pessoas; piloto deterministico em pasta separada.
Checkpoint {MODEL_ID}; revisao {ident['configuration']['model_revision']}.
Fonte: WAVs nosZIPs originais; downmix/DC e VAD48k exatamente reutilizados do18.
VAD top_db40/frame2048/hop480/uniao250ms/minregiao250ms; mesmo fallback legado.
Reamostrar cada regiao a16k por resample_poly1/3, concatenar em ordem temporal.
Dividir em ceil(N/128000) blocos quaseiguais, max8s, sem sobreposicao/padding/truncamento.
Preprocessor normaliza cada bloco individualmente, conforme checkpoint16k.
Modelo eval/frozen/inference_mode, float32, TF32desligado, atencao eager determinista.
Ultima camada contextual24; media e DP populacional de TODOS os frames, ponderados
por numero de frames por bloco; Welfordfloat64 e exportacaofloat32,2048atributos.
Ordem:1024medias, depois1024DPs. Nao usa logits/CTC/transcricao nem testa varias camadas.
Limites:concatenacaoVAD e fronteiras8s mudam o contexto; preprocessamento diferente
da WPT por exigencia do checkpoint. Embeddings podem reter demografia/aquisicao.
Checkpoint externo fixo permite cache sem rotulos antes dos folds; classificador,
selecao e escala serao ajustados somente no treino no experimento21.
AVFAD ja explorada: desenvolvimento, nao teste cego; nao afirma melhoria clinica.
ZIPs/caches/resultados anteriores somente leitura. Falha interrompe sem exportar
matriz parcial; assinatura/hashSQLite impedem retomar com outra configuracao.
"""


def validate_loading_info(info):
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise RuntimeError(f"Pesos incompletos/incompativeis: {info}")
    # Wav2Vec2ForPreTraining tem encoder wav2vec2 + quantizador e duas
    # projecoes do objetivo contrastivo. Apenas esses cabecalhos podem sobrar
    # ao carregar Wav2Vec2Model; nunca aceitar outras chaves silenciosamente.
    unexpected = info.get("unexpected_keys", [])
    allowed = ("quantizer.", "project_hid.", "project_q.")
    if any(not key.removeprefix("wav2vec2.").startswith(allowed) for key in unexpected):
        raise RuntimeError(f"Pesos nao reconhecidos fora dos cabecalhos de pretreino:{unexpected}")


def load_encoder(model_dir, use_safe, device):
    # Deve preceder o primeiro contextoCUDA para matmul deterministico.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    if not use_safe:
        parsed = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
        if parsed < (2, 6):
            raise RuntimeError("Pesos.bin exigem torch>=2.6 para carregamento seguro")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponivel; nenhuma mudanca automatica paraCPU")
    torch.manual_seed(20261106)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    processor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
    model, info = Wav2Vec2Model.from_pretrained(str(model_dir), local_files_only=True,
                                               trust_remote_code=False, use_safetensors=use_safe,
                                               weights_only=True, attn_implementation="eager",
                                               output_loading_info=True)
    validate_loading_info(info)
    model.requires_grad_(False)
    model.eval()
    model.to(device=device, dtype=torch.float32)
    runtime = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "cuda_runtime": torch.version.cuda, "torch": torch.__version__,
               "ignored_pretraining_head_keys": info.get("unexpected_keys", []),
               "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    return torch, processor, model, runtime


def infer_vector(chunks, torch, processor, model, device):
    moments = FrameMoments()
    with torch.inference_mode():
        for chunk in chunks:
            batch = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt",
                              return_attention_mask=True, padding=False)
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch, output_hidden_states=False, return_dict=True).last_hidden_state
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != HIDDEN_SIZE:
                raise ValueError("Hidden states fora do contrato XLS-R300M")
            moments.update(hidden[0].cpu().numpy())
    vector = moments.vector()
    if vector.shape != (DIM,) or not np.isfinite(vector).all():
        raise ValueError("Embedding final invalido")
    return vector, moments.count


def init_cache(path, signature):
    exists = path.exists()
    con = sqlite3.connect(f"file:{path.as_posix()}?mode={'rw' if exists else 'rwc'}", uri=True)
    con.execute("PRAGMA synchronous=FULL")
    if exists:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if tables != {"config", "features", "errors"}:
            con.close()
            raise ValueError("SchemaSQLite desconhecido")
    else:
        con.execute("CREATE TABLE config (signature TEXT NOT NULL)")
        con.execute("INSERT INTO config VALUES (?)", (signature,))
        con.execute("CREATE TABLE features (sample_id TEXT PRIMARY KEY, participant_id TEXT NOT NULL, task TEXT NOT NULL, vector BLOB NOT NULL, sha256 TEXT NOT NULL, diagnostics TEXT NOT NULL)")
        con.execute("CREATE TABLE errors (sample_id TEXT PRIMARY KEY, error TEXT NOT NULL)")
        con.commit()
    if con.execute("SELECT signature FROM config").fetchall() != [(signature,)]:
        con.close()
        raise ValueError("AssinaturaSQLite diferente; preservar cache existente")
    return con


def validate_cache(con, selected):
    expected = selected.set_index("sample_id")
    found = {}
    for sid, pid, task, blob, digest, diag in con.execute("SELECT * FROM features"):
        if sid not in expected.index or str(expected.loc[sid, "participant_id"]) != pid or str(expected.loc[sid, "task"]) != task:
            raise ValueError("IDsSQLite divergentes do manifesto")
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(f"Hashembedding corrompido:{sid}")
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.shape != (DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"EmbeddingSQLite invalido:{sid}")
        found[sid] = (vector.copy(), json.loads(diag))
    if any(sid not in expected.index for sid, in con.execute("SELECT sample_id FROM errors")):
        raise ValueError("ErroSQLite fora do manifesto")
    return found


def export(out, selected, found, ident, elapsed, work_seconds, new_count, runtime):
    if set(found) != set(selected.sample_id):
        raise ValueError("Nenhuma exportacao final de matriz parcial")
    metadata = selected.copy().reset_index(drop=True)
    diagnostics = pd.DataFrame([found[sid][1] for sid in metadata.sample_id])
    for column in diagnostics:
        metadata[column] = diagnostics[column]
    metadata["matrix_row"] = np.arange(len(metadata))
    matrix = np.vstack([found[sid][0] for sid in metadata.sample_id])
    stream = io.BytesIO()
    np.savez_compressed(stream, embeddings=matrix, feature_names=feature_names(),
                        sample_id=metadata.sample_id.to_numpy(dtype=str),
                        participant_id=metadata.participant_id.to_numpy(dtype=str),
                        task=metadata.task.to_numpy(dtype=str))
    target = out / EXPORTS[0]
    temporary = target.with_name(target.name + ".part")
    temporary.write_bytes(stream.getvalue()); temporary.replace(target)
    EXP18.atomic_csv(out / EXPORTS[1], metadata)
    rate = work_seconds / new_count if new_count else None
    text = ["EXPERIMENTO20 - EMBEDDINGS XLS-R300M CONGELADOS", f"Concluido:{EXP18.timestamp()}",
            f"Modo:{ident['mode']}; revisao:{ident['configuration']['model_revision']}",
            f"Matriz:{matrix.shape}; pessoas:{metadata.participant_id.nunique()}; nao finitos:{int((~np.isfinite(matrix)).sum())}",
            f"Segundos desta chamada:{elapsed:.3f}; novas linhas:{new_count}; segundos novas linhas:{work_seconds:.3f}",
            f"Segundos por nova linha:{rate}; estimativa2465linhas:{rate * 2465 if rate else None}",
            "Estimativa depende de duracao/GPU/ZIP; piloto nao avalia classificacao.",
            f"Runtime:{json.dumps(runtime, ensure_ascii=False, sort_keys=True)}",
            f"Frames por gravacao min/mediana/max:{metadata.hidden_frames.min()}/{metadata.hidden_frames.median()}/{metadata.hidden_frames.max()}",
            f"Chunks por gravacao min/mediana/max:{metadata.chunk_count.min()}/{metadata.chunk_count.median()}/{metadata.chunk_count.max()}",
            f"FallbacksVAD:{metadata.vad_fallback.sum()}; padding/truncamento:0; nenhum caso excluido.",
            "1024medias seguidas de1024DPs populacionais sobreframes da ultima camada24.",
            "Sem ajuste do encoder/rotulos comoentrada; nenhuma BA calculada nesta etapa.",
            "Ver assinatura_execucao.json, PROTOCOLO.txt e checkpointSQLite novo.", ""]
    atomic_text(out / EXPORTS[3], "\n".join(text))
    atomic_text(out / "ARTEFATOS_COMPLETOS.json", json_text({name: sha(out / name) for name in EXPORTS}))


def run(root, out, selected, ident, model_dir, use_safe, device):
    text = json_text(ident)
    signature = hashlib.sha256(text.encode("utf-8")).hexdigest()
    signature_file = out / "assinatura_execucao.json"
    if signature_file.exists():
        if signature_file.read_text(encoding="utf-8") != text:
            raise ValueError("Configuracao/insumos/codigo/modelo mudaram; nao reutilizar pasta")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Pasta nao vazia sem assinatura; preservar e revisar")
        out.mkdir(parents=True, exist_ok=True)
        atomic_text(signature_file, text)
        atomic_text(out / "PROTOCOLO.txt", protocol(ident))
    if (out / "PROTOCOLO.txt").read_text(encoding="utf-8") != protocol(ident):
        raise ValueError("Protocolo congelado alterado")
    con = init_cache(out / "checkpoint_wav2vec.sqlite", signature)
    opened = {}
    started = time.perf_counter()
    new_count, work_seconds = 0, 0.0
    try:
        found = validate_cache(con, selected)
        marker = out / "ARTEFATOS_COMPLETOS.json"
        if marker.exists():
            digests = json.loads(marker.read_text(encoding="utf-8"))
            if set(digests) != set(EXPORTS) or set(found) != set(selected.sample_id) or con.execute("SELECT COUNT(*) FROM errors").fetchone()[0]:
                raise ValueError("Marcador final inconsistente/incompleto")
            if any(sha(out / name) != digest for name, digest in digests.items()):
                raise ValueError("Exportacao final alterada")
            log(out, "Retomada: cache e exportacoes completos integros; nenhuma reextracao.")
            return
        torch, processor, model, runtime = load_encoder(model_dir, use_safe, device)
        log(out, f"Inicio/retomada20:{ident['mode']}; {len(found)}/{len(selected)} em cache; GPU={runtime['gpu']}")
        for row in selected.itertuples(index=False):
            if row.sample_id in found:
                continue
            one_start = time.perf_counter()
            try:
                signal48, audio_diag = EXP18.read_signal48(root, row, opened)
                chunks, diag = prepare_signal(signal48)
                vector, frames = infer_vector(chunks, torch, processor, model, device)
                diag.update(audio_diag)
                diag.update(hidden_frames=frames, extraction_seconds=time.perf_counter() - one_start)
                blob = vector.tobytes()
                with con:
                    con.execute("INSERT INTO features VALUES (?,?,?,?,?,?)", (row.sample_id, row.participant_id, row.task,
                                blob, hashlib.sha256(blob).hexdigest(), json_text(diag)))
                    con.execute("DELETE FROM errors WHERE sample_id=?", (row.sample_id,))
                found[row.sample_id] = (vector, diag)
                new_count += 1
                work_seconds += diag["extraction_seconds"]
            except Exception as exc:
                with con:
                    con.execute("INSERT OR REPLACE INTO errors VALUES (?,?)", (row.sample_id, repr(exc)))
                log(out, f"ERRO:{row.sample_id}:{exc!r}; nenhuma exportacao parcial")
                raise
            if ident["mode"] != "completo" or len(found) % 25 == 0 or len(found) == len(selected):
                log(out, f"{len(found)}/{len(selected)};ultimo={row.sample_id};segundos={diag['extraction_seconds']:.3f};frames={frames}")
        if con.execute("SELECT COUNT(*) FROM errors").fetchone()[0]:
            raise ValueError("Cache ainda contem erros")
        export(out, selected, validate_cache(con, selected), ident, time.perf_counter() - started,
               work_seconds, new_count, runtime)
        log(out, "Extracao20 concluida, sem exclusoes; embeddings finitos e IDs embutidos.")
    finally:
        con.close()
        for archive in opened.values():
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--modelo", type=Path)
    parser.add_argument("--revisao")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true")
    action.add_argument("--piloto", type=int, nargs="?", const=3)
    action.add_argument("--executar", action="store_true")
    action.add_argument("--autoteste", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        import unittest
        import test_wav2vec_extracao20 as tests
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful():
            raise SystemExit(1)
        return
    if args.modelo is None or args.revisao is None or not re.fullmatch(r"[0-9a-f]{40}", args.revisao):
        parser.error("Informar --modelo local e --revisao commitSHA40; nenhum download automatico")
    if args.piloto is not None and not 1 <= args.piloto <= 10:
        parser.error("Piloto precisa ter entre1e10 gravacoes")
    root, model_dir = args.dados.resolve(), args.modelo.resolve()
    meta, _, _, _, paths, archives = EXP18.load_inputs(root)
    pilot = args.piloto is not None
    selected = meta.iloc[np.unique(np.linspace(0, len(meta) - 1, args.piloto, dtype=int))].copy() if pilot else meta.copy()
    mode = f"piloto_{len(selected)}" if pilot else "completo"
    files, use_safe = model_files(model_dir, args.revisao)
    ident = identity(paths, archives, selected, mode, model_dir, args.revisao, files, use_safe, args.device)
    print(f"ENTRADAS20 OK:2465gravacoes/411pessoas;selecionadas={len(selected)};modo={mode};2048atributos;modelo local.", flush=True)
    if not (args.executar or pilot):
        print("Somente verificacao:nenhum WAV lido, torch importado, download ou escrita.", flush=True)
        return
    default = root / "results/experiments" / (f"resultados_20_wav2vec_piloto_{len(selected)}" if pilot else OUTPUT_NAME)
    out = (args.saida or default).resolve()
    if not out.is_relative_to(root / "results/experiments") or out == root / "results/experiments":
        parser.error("Saida precisa ser subpasta de results/experiments")
    if pilot and out == root / "results/experiments" / OUTPUT_NAME:
        parser.error("Piloto nao pode escrever na pasta completa")
    run(root, out, selected, ident, model_dir, use_safe, args.device)


if __name__ == "__main__":
    main()
