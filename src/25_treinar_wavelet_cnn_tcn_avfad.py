# -*- coding: utf-8 -*-
"""Exp25: CNN/TCN Wavelet compactas, treino e avaliação por participante.

Requer .venv_wav2vec. --verificar não ajusta modelos. --autoteste usa somente
dados sintéticos. --executar é explícito e preserva todos os artefatos antigos.
"""
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
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp21_for_exp25", HERE / "21_avaliar_wpt_wav2vec_avfad.py")
EXP21 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP21)
EXP19, EXP17, EXP13 = EXP21.EXP19, EXP21.EXP17, EXP21.EXP13
VERSION = "25.1"
SEEDS = EXP21.SEEDS
ARMS = ("A_CNN_RAW", "B_CNN_RMS", "C_TCN_RAW", "D_TCN_RMS", "E_MLP_RESUMOS_RMS")
ARM_INPUT = {ARMS[0]: "raw", ARMS[1]: "rms", ARMS[2]: "raw", ARMS[3]: "rms", ARMS[4]: "summaries_rms"}
ARM_MODEL = {ARMS[0]: "cnn", ARMS[1]: "cnn", ARMS[2]: "tcn", ARMS[3]: "tcn", ARMS[4]: "mlp"}
STEPS, WIDTH = 128, 480
DATA23_REL = Path("results/experiments/resultados_23_wpt_rms_sequencias")
OUT_REL = Path("results/experiments/resultados_25_wavelet_cnn_tcn")
DEFAULT_OPTIONS = {"max_epochs": 40, "min_epochs": 5, "patience": 8, "min_delta": 1e-4,
                   "learning_rate": 1e-3, "weight_decay": 1e-3, "dropout": .25,
                   "batch_people": 32, "prediction_batch": 64, "steps": STEPS,
                   "gradient_clip": 5.0}
CSV_FILES = {"predicoes.csv", "predicoes_treino_participante.csv", "predicoes_teste_gravacao.csv",
             "predicoes_treino_gravacao.csv", "metricas.csv", "historico.csv", "particoes_internas.csv"}
ARTIFACTS = CSV_FILES | {"parametros_preprocessamento.npz", "modelo_refit.pt", "modelo_selecao.pt", "selecao_epocas.json"}
CONTRASTS = {"B-A_RMS_CNN_primaria": {ARMS[1]: 1, ARMS[0]: -1},
             "D-C_RMS_TCN_primaria": {ARMS[3]: 1, ARMS[2]: -1},
             "D-B_TCN_CNN_RMS_secundaria": {ARMS[3]: 1, ARMS[1]: -1},
             "C-A_TCN_CNN_raw_secundaria": {ARMS[2]: 1, ARMS[0]: -1},
             "B-E_CNN_resumos_secundaria": {ARMS[1]: 1, ARMS[4]: -1},
             "D-E_TCN_resumos_secundaria": {ARMS[3]: 1, ARMS[4]: -1}}


def protocol(options=DEFAULT_OPTIONS):
    return f"""EXPERIMENTO 25 — WAVELET TEMPORAL, CNN E TCN COMPACTAS
Hipótese: a ordem temporal perdida nos resumos pode conter informação útil.
411 participantes HSA, 82 normais/329 patológicos, 2.465 frases, sem exclusão.
Entradas23: sequências WPT sym5 de48k, L5/L6,480 descritores por janela400/200ms;
raw e RMS0.1 por gravação; controle com1920 resumos RMS. VAD original mantido.
Cada sequência é interpolada linearmente sobre sua extensão completa para128
passos (uma janela é repetida). Sem truncar início/fim. Esta compressão pode
atenuar ou omitir transientes; não preserva integralmente cada janela original.
Interpolação não aprende estatísticas, não usa rótulos nem aumenta amostra.
A_CNN_RAW/B_CNN_RMS: projeção480→32, convoluções temporais32→48→64,
GroupNorm, GELU, pooling média/DP e vetor32. C_TCN_RAW/D_TCN_RMS:
projeção480→32, quatro blocos residuais32 com dilatações1/2/4/8, duas
convoluções de kernel3 por bloco, GroupNorm e pooling média/DP→vetor32.
TCN offline NÃO causal: padding simétrico. E_MLP_RESUMOS_RMS:1920→32→32.
Todos têm menos de80mil parâmetros; dropout={options['dropout']}; sem BatchNorm.
Comparações são entre pipelines; E não isola perfeitamente capacidade/ordem.
Padronização por atributo usando média/DP dos passos128 das gravações de treino;
as gravações têm o mesmo peso nas estatísticas. Sem ANOVA ou ajuste global.
Treino por PARTICIPANTE: batch={options['batch_people']} pessoas com todas suas
frases; média dos logits das frases ANTES da BCE. Pesos N/(2*Nclasse) calculados
nos participantes de treino. Não contar janelas/frases como pessoas novas.
Cinco folds externos por pessoa, mesmos13/17/21; seeds20261105 principal e
101..105 estabilidade. Uma validação interna por pessoa: primeiro fold do
StratifiedKFold5 com seed=outerseed+100+outerfold, dentro do treino externo.
Escala da seleção usa só treino interno. Escolhe época por BCE ponderada por
pessoa na validação interna, usando pesos das classes de treino interno.
Máximo{options['max_epochs']} épocas; candidatos desde época{options['min_epochs']};
patience={options['patience']}, min_delta={options['min_delta']}. Melhor perda,
sem empate favorável a época tardia. Não escolher época por BA do teste.
Refit DO ZERO no treino externo inteiro pela época selecionada, com nova escala
do treino externo e os mesmos pesos iniciais. Avaliar teste externo só depois.
AdamW lr={options['learning_rate']}, weight_decay={options['weight_decay']}; sem scheduler,
gradientes limitados por norma global{options['gradient_clip']} e rejeição de gradiente não finito;
sem augmentation, SMOTE, ajuste de limiar, busca de arquitetura ou encoder.
Inicialização pareada raw/RMS dentro de cada família; CNN/TCN/MLP offsets distintos.
Float32, TF32 desligado, operações determinísticas, quatro threads CPU.
Decisão principal por pessoa: média dos logits, limiar0; por gravação é apoio.
B-A e D-C: contrastes primários de amplitude por arquitetura; D-B,C-A,B-E,D-E
secundários. Sementes principal/estabilidade separadas. IC95 condicional pareado
por pessoa,2000 bootstraps; repetições da pessoa ficam no mesmo cluster.
AUC por fold ponderada por pares; AUC de logits concatenados exploratória.
AVFAD já explorada: desenvolvimento, não teste cego; sem validação SVD neste ciclo.
Salvar partições, épocas/históricos, normalizadores, estados e hashes. Nenhum
resultado antigo sobrescrito; resultado negativo preservado; sem conclusão clínica.
"""


def interpolate_sequence(sequence, steps=STEPS):
    """Interpolação determinística na extensão inteira; comprime informação."""
    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or steps < 2:
        raise ValueError("Sequência deve ser matriz finita não vazia; passos>=2")
    if len(x) == 1:
        return np.repeat(x.T, steps, axis=1).copy()
    if len(x) == steps:
        return x.T.copy()
    positions = np.linspace(0, len(x) - 1, steps, dtype=np.float64)
    lo = np.floor(positions).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    weight = (positions - lo).astype(np.float32)
    result = x[lo] * (1 - weight[:, None]) + x[hi] * weight[:, None]
    return result.T.copy()


def validate_contract(meta, embedded, offsets, raw, rms, summaries, names, real=True):
    if not {"sample_id", "participant_id", "task", "y", "matrix_row"}.issubset(meta):
        raise ValueError("Metadados23 incompletos")
    if meta[["sample_id", "participant_id", "task", "y"]].isna().any().any() or meta.sample_id.duplicated().any():
        raise ValueError("IDs ausentes/duplicados")
    if meta.groupby("participant_id").y.nunique().ne(1).any() or set(meta.y.unique()) != {0, 1}:
        raise ValueError("Rótulos inconsistentes por participante")
    if meta.duplicated(["participant_id", "task"]).any() or not np.array_equal(meta.matrix_row, np.arange(len(meta))):
        raise ValueError("Ordem/tarefa da metadata incorreta")
    for key in ("sample_id", "participant_id", "task"):
        if not np.array_equal(meta[key].astype(str), np.asarray(embedded[key]).astype(str)):
            raise ValueError(f"IDs NPZ/metadata diferentes: {key}")
    if offsets.dtype.kind not in "iu" or offsets.shape != (len(meta) + 1,) or offsets[0] != 0 or not np.all(np.diff(offsets) > 0):
        raise ValueError("Offsets inválidos")
    if raw.shape != rms.shape or raw.shape != (int(offsets[-1]), WIDTH) or summaries.shape != (len(meta), 1920):
        raise ValueError("Dimensão de sequências/resumos incorreta")
    if len(names) != WIDTH or len(set(names)) != WIDTH or not np.isfinite(raw).all() or not np.isfinite(rms).all() or not np.isfinite(summaries).all():
        raise ValueError("Nomes ou atributos inválidos")
    if real:
        people = meta[["participant_id", "y"]].drop_duplicates()
        if (len(meta), len(people), people.y.value_counts().to_dict()) != (2465, 411, {1: 329, 0: 82}):
            raise ValueError("Coorte não corresponde a HSA2465/411/82/329")
        if meta.groupby("participant_id").size().value_counts().to_dict() != {6: 410, 5: 1}:
            raise ValueError("Número de frases inesperado")


def interpolation_cache(data23, meta, offsets, steps, cache_dir):
    """Cache não supervisionado em memmaps; nunca mantém raw+rms expandidos na RAM."""
    cache_dir = Path(cache_dir).resolve()
    if not cache_dir.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("Cache deve ser uma pasta nova dentro de checkpoints")
    ident = {"version": VERSION, "source_sha256": EXP17.sha(Path(__file__)), "steps": int(steps),
             "sequences_sha256": EXP17.sha(data23 / "sequencias_wpt.npz"),
             "sample_ids": meta.sample_id.tolist(), "shape": [len(meta), WIDTH, steps],
             "method": "linear_full_extent_float32", "labels_used": False}
    signature_file = cache_dir / "assinatura_cache.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf8")) != ident:
            raise RuntimeError("Cache interpolado de outra configuração; preservar")
    else:
        if cache_dir.exists() and any(cache_dir.iterdir()):
            raise RuntimeError("Cache não vazio sem assinatura; preservar")
        cache_dir.mkdir(parents=True, exist_ok=True)
        EXP17.json_write(signature_file, ident)
    for key in ("raw", "rms"):
        target, marker = cache_dir / f"{key}_{steps}.npy", cache_dir / f"{key}_{steps}.json"
        if marker.exists():
            recorded = json.loads(marker.read_text(encoding="utf8"))
            if not target.is_file() or recorded != {"sha256": EXP17.sha(target), "signature": EXP17.sha(signature_file)}:
                raise ValueError("Cache interpolado alterado ou de outra assinatura")
            continue
        if target.exists():
            raise RuntimeError("Cache parcial preservado; não sobrescrever")
        temporary = target.with_name(target.name + ".part")
        if temporary.exists():
            raise RuntimeError("Cache parcial .part preservado")
        with np.load(data23 / "sequencias_wpt.npz", allow_pickle=False) as z:
            source = np.asarray(z[f"data_{key}"], dtype=np.float32)
            destination = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                                    shape=(len(meta), WIDTH, steps))
            for i in range(len(meta)):
                destination[i] = interpolate_sequence(source[offsets[i]:offsets[i + 1]], steps)
                if (i + 1) % 64 == 0:
                    destination.flush()
            destination.flush()
            del destination, source
        temporary.replace(target)
        EXP17.json_write(marker, {"sha256": EXP17.sha(target), "signature": EXP17.sha(signature_file)})
    return {key: np.load(cache_dir / f"{key}_{steps}.npy", mmap_mode="r", allow_pickle=False) for key in ("raw", "rms")}


def load_inputs(root=ROOT, data23=None, steps=STEPS, materialize=True, cache_dir=None):
    root, data23 = Path(root).resolve(), Path(data23 or Path(root) / DATA23_REL).resolve()
    marker = json.loads((data23 / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    required = {"sequencias_wpt.npz", "features_wpt_rms.npz", "metadata_wpt_rms.csv", "assinatura_execucao.json"}
    if not required.issubset(marker):
        raise ValueError("Exportação23 incompleta")
    for name, digest in marker.items():
        if Path(name).name != name or EXP17.sha(data23 / name) != digest:
            raise ValueError(f"Exportação23 alterada ou caminho inválido: {name}")
    meta = pd.read_csv(data23 / "metadata_wpt_rms.csv", dtype={"sample_id": str, "participant_id": str, "task": str}, float_precision="round_trip")
    meta["task"] = meta.task.str.zfill(3)
    with np.load(data23 / "sequencias_wpt.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"])
        raw, rms = np.asarray(z["data_raw"], dtype=np.float32), np.asarray(z["data_rms"], dtype=np.float32)
        names = z["feature_names"].astype(str)
        embedded = {key: z[key].astype(str) for key in ("sample_id", "participant_id", "task")}
    with np.load(data23 / "features_wpt_rms.npz", allow_pickle=False) as z:
        summaries = np.asarray(z["wavelet"], dtype=np.float32)
        for key in embedded:
            if not np.array_equal(z[key].astype(str), embedded[key]):
                raise ValueError("NPZ23 resumido/temporal discordam em IDs")
    validate_contract(meta, embedded, offsets, raw, rms, summaries, names)
    _, old_meta, _ = EXP17.load_inputs(root)
    for key in ("sample_id", "participant_id", "task", "y"):
        if not np.array_equal(meta[key].astype(str), old_meta[key].astype(str)):
            raise ValueError("Coorte23 mudou em relação à referência")
    if not materialize:
        return {"raw": raw, "rms": rms, "summaries_rms": summaries, "offsets": offsets}, meta, names
    del raw, rms
    result = interpolation_cache(data23, meta, offsets, steps, cache_dir or root / "checkpoints/cache_25_interpolacao")
    result["summaries_rms"] = summaries
    return result, meta, names


def configure_torch(device="cuda", threads=4):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível; nenhuma mudança automática para CPU")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_everything(seed, device="cpu"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def arm_seed(seed, fold, arm):
    # Mesma inicialização raw/RMS, controlando a ablação de amplitude.
    family = {"cnn": 1, "tcn": 2, "mlp": 3}[ARM_MODEL[arm]]
    return int((int(seed) * 1009 + int(fold) * 67 + family * 7919) % (2**32 - 1))


class StatsPool(nn.Module):
    def forward(self, x):
        return torch.cat([x.mean(dim=-1), x.std(dim=-1, unbiased=False)], dim=1)


class CNNWavelet(nn.Module):
    def __init__(self, width=WIDTH, dropout=.25):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(width, 32, 1), nn.GroupNorm(4, 32), nn.GELU(),
                                  nn.Conv1d(32, 48, 5, stride=2, padding=2), nn.GroupNorm(4, 48), nn.GELU(),
                                  nn.Conv1d(48, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.GELU())
        self.pool = StatsPool()
        self.embedding = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(32, 1)

    def encode(self, x):
        return self.embedding(self.pool(self.body(x)))

    def forward(self, x):
        return self.classifier(self.encode(x)).squeeze(-1)


class ResidualTemporal(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation),
                                  nn.GroupNorm(4, channels), nn.GELU(), nn.Dropout(dropout),
                                  nn.Conv1d(channels, channels, 3, dilation=dilation, padding=dilation),
                                  nn.GroupNorm(4, channels), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.body(x)


class TCNWavelet(nn.Module):
    def __init__(self, width=WIDTH, dropout=.25):
        super().__init__()
        self.project = nn.Sequential(nn.Conv1d(width, 32, 1), nn.GroupNorm(4, 32), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualTemporal(32, d, dropout) for d in (1, 2, 4, 8)])
        self.pool = StatsPool()
        self.embedding = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(32, 1)

    def encode(self, x):
        return self.embedding(self.pool(self.blocks(self.project(x))))

    def forward(self, x):
        return self.classifier(self.encode(x)).squeeze(-1)


class SummaryMLP(nn.Module):
    def __init__(self, width=1920, dropout=.25):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(width, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout),
                                       nn.Linear(32, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(32, 1)

    def encode(self, x):
        return self.embedding(x)

    def forward(self, x):
        return self.classifier(self.encode(x)).squeeze(-1)


def model_for(arm, options=DEFAULT_OPTIONS):
    classes = {"cnn": CNNWavelet, "tcn": TCNWavelet, "mlp": SummaryMLP}
    result = classes[ARM_MODEL[arm]](dropout=options["dropout"])
    if sum(parameter.numel() for parameter in result.parameters()) >= 80000:
        raise ValueError("Modelo excede o orçamento fixado de parâmetros")
    return result


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit_scaler(values, train):
    """Mesmos128 passos por gravação: comprimentos originais não dominam."""
    x = np.asarray(values)
    if x.ndim not in (2, 3) or not len(train):
        raise ValueError("Entrada inválida para normalização")
    width = x.shape[1]
    count, mean, m2 = 0, np.zeros(width, np.float64), np.zeros(width, np.float64)
    for start in range(0, len(train), 32):
        batch = np.asarray(x[train[start:start + 32]], dtype=np.float64)
        if not np.isfinite(batch).all():
            raise ValueError("Atributos de treino não finitos")
        if x.ndim == 3:
            batch = batch.transpose(0, 2, 1).reshape(-1, width)
        n = len(batch)
        local_mean = batch.mean(axis=0)
        local_m2 = np.sum((batch - local_mean) ** 2, axis=0)
        total = count + n
        delta = local_mean - mean
        m2 += local_m2 + delta**2 * (count * n / total)
        mean += delta * (n / total)
        count = total
    scale = np.sqrt(np.maximum(m2 / count, 0.))
    scale[scale < 1e-8] = 1.
    return {"mean": mean, "scale": scale, "n_frames": np.asarray(count), "train_rows": np.asarray(train, np.int64)}


def transform(values, rows, scaler):
    x = np.asarray(values[rows], dtype=np.float32)
    shape = (1, -1, 1) if x.ndim == 3 else (1, -1)
    result = (x - scaler["mean"].astype(np.float32).reshape(shape)) / scaler["scale"].astype(np.float32).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError("Normalização produziu atributos não finitos")
    return np.ascontiguousarray(result)


def people_rows(meta, rows):
    subset = meta.iloc[rows]
    people = EXP13.person_table(subset)
    if subset.groupby("participant_id").y.nunique().ne(1).any():
        raise ValueError("Rótulo divergente entre frases")
    original_ids = meta.participant_id.to_numpy()
    groups = {pid: rows[original_ids[rows] == pid] for pid in people.participant_id}
    return people, groups


def class_weights(meta, rows):
    people, _ = people_rows(meta, rows)
    counts = people.y.value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("Treino precisa conter ambas as classes")
    return np.asarray([len(people) / (2 * counts[label]) for label in (0, 1)], dtype=np.float64)


def person_logits(record_logits, counts):
    if sum(counts) != len(record_logits) or any(n <= 0 for n in counts):
        raise ValueError("Agrupamento de frases inválido")
    # Reduções contíguas evitam scatter/index_add CUDA não determinísticos.
    return torch.stack([chunk.mean() for chunk in torch.split(record_logits, counts)])


def batches(meta, rows, batch_size, rng=None):
    people, groups = people_rows(meta, rows)
    order = np.arange(len(people))
    if rng is not None:
        order = rng.permutation(order)
    for start in range(0, len(order), batch_size):
        selected = people.iloc[order[start:start + batch_size]]
        group_rows = [groups[pid] for pid in selected.participant_id]
        yield np.concatenate(group_rows), [len(group) for group in group_rows], selected.y.to_numpy(dtype=np.int64)


def train_epoch(model, values, meta, rows, scaler, optimizer, weights, device, options, seed, epoch):
    model.train()
    total_loss, n_people = 0., 0
    for record_rows, counts, labels in batches(meta, rows, options["batch_people"], np.random.default_rng(seed + epoch)):
        x = torch.from_numpy(transform(values, record_rows, scaler)).to(device)
        y = torch.as_tensor(labels, device=device, dtype=torch.float32)
        w = torch.as_tensor(weights[labels], device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = person_logits(model(x), counts)
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Perda de treino não finita")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(labels)
        n_people += len(labels)
    return total_loss / n_people


def predict_records(model, values, rows, scaler, device, batch_size=64):
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            x = torch.from_numpy(transform(values, rows[start:start + batch_size], scaler)).to(device)
            scores.append(model(x).detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise RuntimeError("Logits não finitos")
    return result


def person_loss(meta, scores, weights):
    frame = EXP17.collapse(meta, scores)
    logits, y = frame.score.to_numpy(), frame.y.to_numpy(dtype=np.int64)
    return float(np.mean((np.logaddexp(0., logits) - y * logits) * weights[y]))


def internal_split(meta, train, seed, fold):
    inner_meta = meta.iloc[train].reset_index(drop=True)
    fit, validation = next(EXP13.row_splits(inner_meta, 5, int(seed) + 100 + int(fold)))
    fit, validation = train[fit], train[validation]
    if set(meta.iloc[fit].participant_id) & set(meta.iloc[validation].participant_id):
        raise ValueError("Vazamento de participante na validação interna")
    return fit, validation


def train_arm(inputs, meta, train, test, seed, fold, arm, device="cuda", options=None, log=lambda _: None):
    """APIs reutilizáveis: nenhuma seleção/normalização usa X/y externos."""
    options = dict(DEFAULT_OPTIONS if options is None else options)
    train, test = np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)
    if set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise ValueError("Participante compartilhado entre treino e teste externo")
    fit, validation = internal_split(meta, train, seed, fold)
    values = inputs[ARM_INPUT[arm]]
    selection_scaler = fit_scaler(values, fit)
    weights = class_weights(meta, fit)
    random_seed = arm_seed(seed, fold, arm)
    seed_everything(random_seed, device)
    model = model_for(arm, options).to(device)
    initial_digest = state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    history, best_loss, best_epoch, stale, best_state = [], float("inf"), None, 0, None
    started = time.monotonic()
    for epoch in range(1, options["max_epochs"] + 1):
        loss = train_epoch(model, values, meta, fit, selection_scaler, optimizer, weights, device, options, random_seed, epoch)
        validation_scores = predict_records(model, values, validation, selection_scaler, device, options["prediction_batch"])
        validation_loss = person_loss(meta.iloc[validation].reset_index(drop=True), validation_scores, weights)
        if epoch >= options["min_epochs"]:
            if validation_loss < best_loss - options["min_delta"]:
                best_loss, best_epoch, stale = validation_loss, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        history.append({"phase": "selecao", "epoch": epoch, "train_loss": loss, "validation_loss": validation_loss})
        if epoch >= options["min_epochs"] and stale >= options["patience"]:
            break
    if best_epoch is None:
        raise RuntimeError("Nenhuma época elegível selecionada")
    # Não carrega os pesos da seleção. Mesma inicialização, refit do zero.
    del model, optimizer
    seed_everything(random_seed, device)
    model = model_for(arm, options).to(device)
    if state_digest(model.state_dict()) != initial_digest:
        raise RuntimeError("Refit não foi reinicializado exatamente")
    outer_scaler = fit_scaler(values, train)
    outer_weights = class_weights(meta, train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    for epoch in range(1, best_epoch + 1):
        loss = train_epoch(model, values, meta, train, outer_scaler, optimizer, outer_weights, device, options, random_seed, epoch)
        history.append({"phase": "refit", "epoch": epoch, "train_loss": loss, "validation_loss": np.nan})
    train_score = predict_records(model, values, train, outer_scaler, device, options["prediction_batch"])
    # Primeiro acesso aos atributos de teste depois de terminar toda seleção/refit.
    test_score = predict_records(model, values, test, outer_scaler, device, options["prediction_batch"])
    metadata = {"arm": arm, "seed": int(seed), "fold": int(fold), "random_seed": random_seed,
                "selected_epoch": int(best_epoch), "selection_epochs_run": int(sum(row["phase"] == "selecao" for row in history)),
                "best_validation_loss": float(best_loss), "selection_class_weights": weights.tolist(),
                "outer_class_weights": outer_weights.tolist(), "initial_state_sha256": initial_digest,
                "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "seconds": time.monotonic() - started}
    arrays = {f"selection__{key}": value for key, value in selection_scaler.items()}
    arrays.update({f"outer__{key}": value for key, value in outer_scaler.items()})
    partitions = []
    for role, rows in (("treino_interno", fit), ("validacao_interna", validation), ("teste_externo", test)):
        for row in meta.iloc[rows].drop_duplicates("participant_id").itertuples():
            partitions.append({"seed": seed, "fold": fold, "arm": arm, "role": role,
                               "participant_id": row.participant_id, "y": row.y})
    frames = prediction_frames(meta, train, test, train_score, test_score, arm, seed, fold)
    frames["historico.csv"] = pd.DataFrame(history)
    frames["particoes_internas.csv"] = pd.DataFrame(partitions)
    result = {"frames": frames, "arrays": arrays, "selection": metadata, "selection_state": best_state,
              "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
    log(f"25 seed={seed} fold={fold} {arm}: epoca={best_epoch}; treino={EXP17.measures(frames['predicoes_treino_participante.csv'])['BA']:.6f}; teste={EXP17.measures(frames['predicoes.csv'])['BA']:.6f}; segundos={metadata['seconds']:.2f}")
    return result


def prediction_frames(meta, train, test, train_score, test_score, arm, seed, fold):
    frames, metrics = {}, []
    for phase, rows, scores, people_name, recording_name in (
        ("treino", train, train_score, "predicoes_treino_participante.csv", "predicoes_treino_gravacao.csv"),
        ("teste", test, test_score, "predicoes.csv", "predicoes_teste_gravacao.csv")):
        local = meta.iloc[rows].reset_index(drop=True)
        people = EXP17.collapse(local, scores)
        people["arm"], people["seed"], people["fold"], people["limiar"] = arm, seed, fold, 0.
        records = EXP19.recording_predictions(local, scores, arm, seed, fold)
        frames[people_name], frames[recording_name] = people, records
        for unit, group in (("participante", people), ("gravacao", records)):
            metrics.append({"arm": arm, "seed": seed, "fold": fold, "phase": phase, "unit": unit, **EXP17.measures(group)})
    frames["metricas.csv"] = pd.DataFrame(metrics)
    return frames


def validate_result(result, meta, train, test, seed, fold, arm, options=DEFAULT_OPTIONS):
    frames, arrays, selected = result["frames"], result["arrays"], result["selection"]
    if set(frames) != CSV_FILES or selected["arm"] != arm or selected["seed"] != seed or selected["fold"] != fold:
        raise ValueError("Resultado incompleto ou de outra partição")
    if not options["min_epochs"] <= selected["selected_epoch"] <= options["max_epochs"]:
        raise ValueError("Época selecionada fora dos limites")
    inner_fit, inner_validation = internal_split(meta, np.asarray(train), seed, fold)
    if not np.array_equal(arrays["selection__train_rows"], inner_fit) or not np.array_equal(arrays["outer__train_rows"], train):
        raise ValueError("Normalizadores não pertencem às linhas de treino")
    for prefix in ("selection__", "outer__"):
        if not np.isfinite(arrays[prefix + "mean"]).all() or not np.isfinite(arrays[prefix + "scale"]).all() or (arrays[prefix + "scale"] <= 0).any():
            raise ValueError("Normalizador inválido")
    partition = frames["particoes_internas.csv"]
    expected = {"treino_interno": set(meta.iloc[inner_fit].participant_id),
                "validacao_interna": set(meta.iloc[inner_validation].participant_id),
                "teste_externo": set(meta.iloc[test].participant_id)}
    for role, ids in expected.items():
        actual = partition[partition.role == role]
        if actual.participant_id.duplicated().any() or set(actual.participant_id) != ids:
            raise ValueError("Partição interna/externa incorreta")
    if any(expected[a] & expected[b] for a, b in (("treino_interno", "validacao_interna"), ("treino_interno", "teste_externo"), ("validacao_interna", "teste_externo"))):
        raise ValueError("Participante compartilhado entre partições")
    history = frames["historico.csv"]
    candidates = history[(history.phase == "selecao") & (history.epoch >= options["min_epochs"])]
    best, loss = None, float("inf")
    for row in candidates.itertuples():
        if not np.isfinite(row.validation_loss):
            raise ValueError("Perda de validação não finita")
        if row.validation_loss < loss - options["min_delta"]:
            best, loss = row.epoch, row.validation_loss
    if best != selected["selected_epoch"] or abs(loss - selected["best_validation_loss"]) > 1e-12:
        raise ValueError("Época não corresponde à perda interna registrada")
    if len(history[history.phase == "refit"]) != best:
        raise ValueError("Refit não executou exatamente a época escolhida")
    labels = meta.drop_duplicates("participant_id").set_index("participant_id").y
    for people_name, records_name, rows in (("predicoes.csv", "predicoes_teste_gravacao.csv", test),
                                           ("predicoes_treino_participante.csv", "predicoes_treino_gravacao.csv", train)):
        people, records = frames[people_name], frames[records_name]
        for frame, key, expected_ids in ((people, "participant_id", set(meta.iloc[rows].participant_id)),
                                         (records, "sample_id", set(meta.iloc[rows].sample_id))):
            if frame[key].duplicated().any() or set(frame[key]) != expected_ids or not frame.arm.eq(arm).all():
                raise ValueError("Previsões com identidades incorretas")
            if not frame.seed.eq(seed).all() or not frame.fold.eq(fold).all() or not np.array_equal(frame.y, frame.participant_id.map(labels)):
                raise ValueError("Rótulos ou partição incorretos")
            if not np.isfinite(frame.score).all() or not np.array_equal(frame.predicao, (frame.score >= 0).astype(int)):
                raise ValueError("Score/decisão inválidos")
        aggregate = EXP17.collapse(records, records.score.to_numpy()).set_index("participant_id").sort_index()
        observed = people.set_index("participant_id").sort_index()
        np.testing.assert_allclose(observed.score, aggregate.score, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(observed.n_gravacoes, aggregate.n_gravacoes)
    if len(frames["metricas.csv"]) != 4:
        raise ValueError("Métricas incompletas")
    for row in frames["metricas.csv"].itertuples():
        name = {("teste", "participante"): "predicoes.csv", ("teste", "gravacao"): "predicoes_teste_gravacao.csv",
                ("treino", "participante"): "predicoes_treino_participante.csv", ("treino", "gravacao"): "predicoes_treino_gravacao.csv"}[row.phase, row.unit]
        for key, value in EXP17.measures(frames[name]).items():
            if not np.isclose(value, getattr(row, key), atol=1e-12, rtol=0):
                raise ValueError("Métrica inconsistente")


def save_checkpoint(destination, result, signature):
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in result["frames"].items():
        EXP17.csv_write(destination / name, frame)
    stream = io.BytesIO()
    np.savez_compressed(stream, **result["arrays"])
    EXP17.atomic_bytes(destination / "parametros_preprocessamento.npz", stream.getvalue())
    for name, state in (("modelo_refit.pt", result["state"]), ("modelo_selecao.pt", result["selection_state"])):
        stream = io.BytesIO(); torch.save(state, stream)
        EXP17.atomic_bytes(destination / name, stream.getvalue())
    EXP17.json_write(destination / "selecao_epocas.json", result["selection"])
    EXP17.json_write(destination / "CONCLUIDO.json", {"finished": EXP17.now(), "run_signature": signature,
                      "sha256": {name: EXP17.sha(destination / name) for name in ARTIFACTS}})


def load_checkpoint(destination, signature):
    marker = json.loads((destination / "CONCLUIDO.json").read_text(encoding="utf8"))
    if marker["run_signature"] != signature or set(marker["sha256"]) != ARTIFACTS:
        raise ValueError("Checkpoint incompleto ou de outra configuração")
    if any(EXP17.sha(destination / name) != digest for name, digest in marker["sha256"].items()):
        raise ValueError("Checkpoint adulterado")
    with np.load(destination / "parametros_preprocessamento.npz", allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    return {"frames": {name: pd.read_csv(destination / name, float_precision="round_trip") for name in CSV_FILES},
            "arrays": arrays, "selection": json.loads((destination / "selecao_epocas.json").read_text(encoding="utf8")),
            "state": torch.load(destination / "modelo_refit.pt", map_location="cpu", weights_only=True),
            "selection_state": torch.load(destination / "modelo_selecao.pt", map_location="cpu", weights_only=True)}


def identity(root, data23, seeds, arms, options, device):
    inputs = [data23 / name for name in ("sequencias_wpt.npz", "features_wpt_rms.npz", "metadata_wpt_rms.csv",
                                        "ARTEFATOS_COMPLETOS.json", "assinatura_execucao.json")]
    code = [Path(__file__).resolve(), HERE / "21_avaliar_wpt_wav2vec_avfad.py", HERE / "19_avaliar_wpt_16khz_avfad.py",
            HERE / "17_robustez_distribuicao_wpt_avfad.py", HERE / "13_otimizar_wpt_sym5.py"]
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                   "cuda_runtime": torch.version.cuda}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "torch"):
        environment[package] = importlib.metadata.version(package)
    return {"version": VERSION, "configuration": {"seeds": list(seeds), "arms": list(arms), **options,
            "outer_folds": 5, "inner_validation": "first_of_5_by_person", "inner_seed": "outer_seed+100+fold",
            "scaler_unit": "recording_equal_length128", "loss_unit": "participant_mean_logit",
            "class_balance": "Nperson/(2*Nclass_train)", "threshold": 0., "paired_initialization_raw_RMS": True,
            "interpolation": "linear_full_extent", "temporal_causality": "noncausal_symmetric_padding"},
            "environment": environment, "inputs": {str(p.resolve()): EXP17.sha(p) for p in inputs},
            "code": {str(p.resolve()): EXP17.sha(p) for p in code},
            "protocol_sha256": hashlib.sha256(protocol(options).encode("utf8")).hexdigest()}


def consolidate(out, meta, seeds, arms, log):
    destinations = [out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" / arm for seed in seeds for fold in range(1, 6) for arm in arms]
    raw = pd.concat([pd.read_csv(p / "predicoes.csv", float_precision="round_trip") for p in destinations], ignore_index=True)
    folds = pd.concat([pd.read_csv(p / "metricas.csv", float_precision="round_trip") for p in destinations], ignore_index=True)
    history = pd.concat([pd.read_csv(p / "historico.csv").assign(arm=p.name, seed=int(p.parent.name.split('_')[1]), fold=int(p.parent.name.split('_')[-1])) for p in destinations], ignore_index=True)
    selections = pd.DataFrame([json.loads((p / "selecao_epocas.json").read_text(encoding="utf8")) for p in destinations])
    summaries = []
    for (seed, arm), group in raw.groupby(["seed", "arm"]):
        if len(group) != meta.participant_id.nunique() or group.participant_id.duplicated().any():
            raise ValueError("OOF incompleto/duplicado")
        measured = EXP17.measures(group)
        measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        sub = folds[(folds.seed == seed) & (folds.arm == arm) & (folds.phase == "teste") & (folds.unit == "participante")]
        measured["AUC_media_folds_ponderada_pares"] = float(np.average(sub.AUC, weights=(sub.TN + sub.FP) * (sub.TP + sub.FN)))
        summaries.append({"seed": seed, "arm": arm, **measured})
    summary = pd.DataFrame(summaries)
    scopes = [("principal_seed20261105", [20261105])] if 20261105 in seeds else []
    secondary = [seed for seed in seeds if seed != 20261105]
    if secondary:
        scopes.append(("estabilidade_secundaria", secondary))
    comparisons = []
    for scope, chosen in scopes:
        for name, coefficients in CONTRASTS.items():
            if set(coefficients).issubset(arms):
                comparisons.append({"escopo": scope, "comparacao": name, **EXP17.intervals(raw, chosen, coefficients, iterations=2000)})
    outputs = {"predicoes_oof_participante.csv": raw, "metricas_por_fold.csv": folds,
               "metricas_por_seed.csv": summary, "historico_todos.csv": history,
               "selecoes_epocas.csv": selections, "comparacoes_pareadas.csv": pd.DataFrame(comparisons)}
    for name, frame in outputs.items():
        EXP17.csv_write(out / name, frame)
    lines = ["EXPERIMENTO25 — CNN/TCN WAVELET POR PARTICIPANTE", f"Consolidado: {EXP17.now()}",
             "Todos os folds solicitados concluídos; seleção de época somente interna.",
             "Ciclo completo de cinco braços/seis sementes." if tuple(arms) == ARMS and tuple(seeds) == SEEDS else "Ciclo parcial/diferente dos cinco braços/seis sementes.", ""]
    for row in summary.itertuples():
        lines.append(f"seed={row.seed} {row.arm}: BA={row.BA:.6f}; sens={row.sensibilidade:.6f}; esp={row.especificidade:.6f}; acc={row.acuracia:.6f}; AUCfold={row.AUC_media_folds_ponderada_pares:.6f}")
    lines += ["", "CONTRASTES PAREADOS — IC95 CONDICIONAL"]
    for row in comparisons:
        lines.append(f"{row['escopo']} {row['comparacao']}: deltaBA={row['delta_BA']:+.6f}; IC95=[{row['IC95_inferior']:+.6f},{row['IC95_superior']:+.6f}]")
    if secondary:
        lines += ["", "ESTABILIDADE NAS CINCO SEMENTES SECUNDÁRIAS; DP NÃO É IC"]
        for arm, group in summary[summary.seed.isin(secondary)].groupby("arm"):
            lines.append(f"{arm}: BA média={group.BA.mean():.6f}; min={group.BA.min():.6f}; max={group.BA.max():.6f}; DP={group.BA.std():.6f}")
    lines += ["", "DIAGNÓSTICO DE TREINO"]
    for arm, group in folds[(folds.phase == "treino") & (folds.unit == "participante")].groupby("arm"):
        lines.append(f"{arm}: BA treino média={group.BA.mean():.6f}")
    lines += ["", "LIMITES", "AVFAD historicamente explorada: desenvolvimento, não teste independente.",
              "Interpolação128 cobre a extensão completa, mas comprime e pode perder transientes.",
              "CNN/TCN/MLP e entradas têm capacidades distintas; controles comparam pipelines.",
              "TCN não causal e avaliação offline; nenhuma alegação de sistema causal/em tempo real.",
              "Época por perda interna, limiar0; não houve busca pela melhor BA externa.",
              "ICs condicionais não refazem seleção/treino nem corrigem tentativas históricas.",
              "Nenhum experimento SVD ou encoder profundo foi treinado neste ciclo.",
              "Dados e resultados anteriores preservados; nenhuma conclusão clínica automática.\n"]
    EXP17.text_write(out / "RELATORIO_RESULTADOS.txt", "\n".join(lines))
    names = list(outputs) + ["RELATORIO_RESULTADOS.txt", "assinatura_execucao.json", "PROTOCOLO.txt", "particoes_participantes.csv"]
    EXP17.json_write(out / "ARTEFATOS_COMPLETOS.json", {name: EXP17.sha(out / name) for name in names})
    log("Consolidação25 concluída; todos os braços e resultados preservados.")


def selftest():
    import unittest
    spec = importlib.util.spec_from_file_location("test_wavelet_cnn_tcn25", HERE / "test_wavelet_cnn_tcn25.py")
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful():
        raise SystemExit(1)


def benchmark_synthetic(device="cuda", max_epochs=40):
    """Somente dados aleatórios: mede custo antes dos ajustes AVFAD."""
    configure_torch(device)
    rng = np.random.default_rng(20261106)
    meta = pd.DataFrame([{"participant_id": f"BENCH{i:03d}", "sample_id": f"BENCH{i:03d}_{j}",
                          "task": str(j), "y": int(i % 4 != 0)} for i in range(85) for j in range(6)])
    rows = np.arange(len(meta))
    options = {**DEFAULT_OPTIONS, "max_epochs": max_epochs}
    for arm in (ARMS[1], ARMS[3], ARMS[4]):
        values = rng.normal(size=(len(meta), 1920) if ARM_MODEL[arm] == "mlp" else (len(meta), WIDTH, STEPS)).astype(np.float32)
        scaler = fit_scaler(values, rows)
        weights = class_weights(meta, rows)
        seed_everything(20261106, device)
        model = model_for(arm, options).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
        # Warm-up separado do tempo usado na projeção.
        train_epoch(model, values, meta, rows, scaler, optimizer, weights, device, options, 20261106, 0)
        if device == "cuda": torch.cuda.synchronize()
        started = time.perf_counter()
        for epoch in range(1, max_epochs + 1):
            train_epoch(model, values, meta, rows, scaler, optimizer, weights, device, options, 20261106, epoch)
        if device == "cuda": torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        # Cada treino externo contém~329pessoas; benchmark85. Seleção/refit
        # podem encerrar cedo. Não inclui validação/exportação/I/O final.
        projection = seconds / max_epochs * (329 / 85) * 2 * DEFAULT_OPTIONS["max_epochs"]
        print(json.dumps({"synthetic_only": True, "arm": arm, "people": 85, "recordings": len(meta),
                          "epochs_measured": max_epochs, "batch_people": options["batch_people"],
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "seconds": seconds, "seconds_per_epoch": seconds / max_epochs,
                          "upper_projection_seconds_per_outer_model_40_plus_40": projection}, sort_keys=True), flush=True)
        del model, optimizer, values
        if device == "cuda": torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--dados23", type=Path)
    parser.add_argument("--saida", type=Path, default=ROOT / OUT_REL)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_OPTIONS["max_epochs"])
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verificar", action="store_true")
    actions.add_argument("--autoteste", action="store_true")
    actions.add_argument("--benchmark", action="store_true")
    actions.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        selftest(); return
    if args.benchmark:
        if args.max_epochs < 1: parser.error("Benchmark requer pelo menos uma época")
        benchmark_synthetic(args.device, args.max_epochs); return
    root, out = args.dados.resolve(), args.saida.resolve()
    seeds, arms = tuple(args.seeds), tuple(args.arms)
    if not out.is_relative_to(root / "results/experiments") or out == root / "results/experiments":
        parser.error("Saída precisa ser subpasta de results/experiments")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed > 2**32 - 1024 for seed in seeds):
        parser.error("Sementes únicas e válidas obrigatórias")
    if len(set(arms)) != len(arms) or args.max_epochs < DEFAULT_OPTIONS["min_epochs"]:
        parser.error("Braços únicos; max-epochs>=5")
    options = {**DEFAULT_OPTIONS, "max_epochs": args.max_epochs}
    data23 = (args.dados23 or root / DATA23_REL).resolve()
    inputs, meta, _ = load_inputs(root, data23, options["steps"], materialize=args.executar)
    configure_torch(args.device)
    print(f"ENTRADAS25 OK: 411 pessoas/2465 gravações; braços={arms}; seeds={seeds}", flush=True)
    if not args.executar:
        print("Somente verificação; nenhum ajuste ou escrita.", flush=True); return
    ident = identity(root, data23, seeds, arms, options, args.device)
    signature_file = out / "assinatura_execucao.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf8")) != ident:
            raise RuntimeError("Configuração/dados/código/ambiente mudaram; preservar pasta")
    else:
        if out.exists() and any(path.name != "PROTOCOLO.txt" for path in out.iterdir()):
            raise RuntimeError("Pasta sem assinatura contém artefatos; preservar")
        out.mkdir(parents=True, exist_ok=True)
        EXP17.freeze_text(out / "PROTOCOLO.txt", protocol(options))
        EXP17.json_write(signature_file, ident)
    EXP17.freeze_text(out / "PROTOCOLO.txt", protocol(options))
    EXP17.freeze_text(out / "particoes_participantes.csv", EXP19.fold_table(meta, seeds).to_csv(index=False, lineterminator="\n"))
    signature = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    log = EXP17.Logger(out / "LOG_EXECUCAO.txt")
    log(f"Início/retomada25; máximo{options['max_epochs']} épocas, seleção interna e refit do zero.")
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            for arm in arms:
                destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" / arm
                if (destination / "CONCLUIDO.json").exists():
                    result = load_checkpoint(destination, signature)
                    validate_result(result, meta, train, test, seed, fold, arm, options)
                    log(f"Checkpoint25 confirmado: seed={seed} fold={fold} {arm}; nenhum reajuste")
                    continue
                if destination.exists() and any(destination.iterdir()):
                    raise RuntimeError(f"Checkpoint parcial preservado: {destination}")
                result = train_arm(inputs, meta, train, test, seed, fold, arm, args.device, options, log)
                validate_result(result, meta, train, test, seed, fold, arm, options)
                save_checkpoint(destination, result, signature)
    consolidate(out, meta, seeds, arms, log)


if __name__ == "__main__":
    main()
