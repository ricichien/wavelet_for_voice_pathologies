# -*- coding: utf-8 -*-
"""Treinamento AVFAD: wavelet temporal vs MFCC (e ComParE como referencia).

O script:
- usa os NPZ/CSV produzidos pelas extracoes ja concluidas;
- alinha Wavelet, MFCC e ComParE por sample_id (nao presume ordem igual);
- cria 5 folds estratificados no nivel de PARTICIPANTE;
- mantem todas as gravacoes de uma pessoa no mesmo fold;
- faz imputacao, remocao de constantes, SelectKBest e escala somente no treino;
- treina SVC linear com pesos balanceados;
- agrega decision scores das frases para produzir uma predicao por pessoa;
- executa analise principal e sensibilidade removendo apenas WAVs > 60 s;
- calcula intervalos bootstrap e diferencas pareadas Wavelet - MFCC nos mesmos participantes;
- escreve CSVs, relatorio Markdown/TXT e pacote de contexto para Claude.

Execute em raiz do repositorio com .venv ativado:
    python .\\modelar_wavelet_temporal_avfad.py

Arquivos necessarios:
    features_wavelet_temporal_avfad_reduzido.npz
    metadata_wavelet_temporal_avfad_reduzido.csv
    features_capev_compare_avfad.npz
    metadata_capev_compare_avfad.csv

O NPZ ComParE deve conter a matriz `acoustic` e `acoustic_names` (MFCC) e,
preferencialmente, `compare` e `compare_names` (baseline ComParE 2016).
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import json
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import rankdata

ROOT = project_root()
WAVELET_NPZ = asset(ROOT, "features_wavelet_temporal_avfad_reduzido.npz")
WAVELET_META = asset(ROOT, "metadata_wavelet_temporal_avfad_reduzido.csv")
BASELINE_NPZ = asset(ROOT, "features_capev_compare_avfad.npz")
# O extrator `extrair_features_capev_compare_avfad.py` fornecido no projeto
# grava o metadata com o nome `metadata_capev_avfad.csv` (sem o sufixo compare).
# Mantemos o nome alternativo como compatibilidade com versões anteriores.
BASELINE_META_CANDIDATES = (
    asset(ROOT, "metadata_capev_avfad.csv"),
    asset(ROOT, "metadata_capev_compare_avfad.csv"),
)

OUT_SUMMARY = asset(ROOT, "resultados_wavelet_vs_mfcc_resumo.csv")
OUT_FOLDS = asset(ROOT, "resultados_wavelet_vs_mfcc_folds.csv")
OUT_OOF = asset(ROOT, "predicoes_oof_wavelet_vs_mfcc.csv")
OUT_PAIRED = asset(ROOT, "comparacao_pareada_wavelet_mfcc.csv")
OUT_FEATURES = asset(ROOT, "estabilidade_selecao_features_wavelet.csv")
OUT_ASSIGNMENT = asset(ROOT, "folds_participantes_wavelet_mfcc.csv")
OUT_EXCLUDED = asset(ROOT, "gravacoes_excluidas_sensibilidade_60s.csv")
OUT_REPORT_MD = asset(ROOT, "relatorio_detalhado_wavelet_vs_mfcc.md")
OUT_REPORT_TXT = asset(ROOT, "relatorio_detalhado_wavelet_vs_mfcc.txt")
OUT_CLAUDE = asset(ROOT, "pacote_claude_resultados_wavelet_mfcc.md")
CONTEXT_FILE = asset(ROOT, "contexto_claude_tcc_avfad.md")
PROGRESS_DIR_BASE = asset(ROOT, "checkpoint_modelagem_wavelet_mfcc")
OUT_PROGRESS_SUMMARY = asset(ROOT, "progresso_modelagem_wavelet_mfcc.csv")
PIPELINE_VERSION = "2026-11-05-v2-resumable"

N_SPLITS = 5
RANDOM_STATE = 42
TOP_K = 128
SVC_C = 1.0
SENSITIVITY_MAX_DURATION_S = 60.0
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20261105
EXPECTED_PARTICIPANTS = 411
EXPECTED_RECORDINGS = 2465
EXPECTED_WAVELET_FEATURES = 4160
EXPECTED_NORMAL = 82
EXPECTED_PATHOLOGICAL = 329

class ModelingError(RuntimeError):
    pass


def say(message: str) -> None:
    print(message, flush=True)


def csv_out(path: Path, data) -> None:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def prepare_progress_dir(baseline_meta_path: Path) -> Path:
    """Cria/retoma checkpoints por configuracao sem misturar execucoes distintas."""
    input_paths = [WAVELET_NPZ, WAVELET_META, BASELINE_NPZ, baseline_meta_path]
    inputs = {}
    for path in input_paths:
        stat = path.stat()
        inputs[path.name] = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
    signature = {
        "pipeline_version": PIPELINE_VERSION,
        "inputs": inputs,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "top_k": TOP_K,
        "svc_c": SVC_C,
        "sensitivity_max_duration_s": SENSITIVITY_MAX_DURATION_S,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    signature_text = json.dumps(signature, sort_keys=True, ensure_ascii=False, indent=2)

    progress_dir = PROGRESS_DIR_BASE
    signature_path = progress_dir / "run_signature.json"
    if progress_dir.exists():
        existing = None
        if signature_path.exists():
            try:
                existing = json.loads(signature_path.read_text(encoding="utf-8"))
            except Exception:
                existing = None
        if existing != signature:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            progress_dir = asset(ROOT, f"{PROGRESS_DIR_BASE.name}_{suffix}")
            counter = 1
            while progress_dir.exists():
                progress_dir = asset(ROOT, f"{PROGRESS_DIR_BASE.name}_{suffix}_{counter}")
                counter += 1
            say(f"Checkpoint anterior preservado; nova pasta de progresso: {progress_dir.name}")
    progress_dir.mkdir(parents=True, exist_ok=True)
    signature_path = progress_dir / "run_signature.json"
    if not signature_path.exists():
        signature_path.write_text(signature_text + "\n", encoding="utf-8")
    return progress_dir


def progress_file_paths(progress_dir: Path, scenario: str, config_name: str) -> dict:
    stem = f"{scenario}__{config_name}"
    return {
        "pred": progress_dir / f"pred_{stem}.csv",
        "folds": progress_dir / f"folds_{stem}.csv",
        "selected": progress_dir / f"selected_{stem}.csv",
    }


def save_config_progress(progress_dir: Path, scenario: str, config_name: str, pred: pd.DataFrame, fold_rows: list, selected_counts: Counter) -> None:
    paths = progress_file_paths(progress_dir, scenario, config_name)
    pred.to_csv(paths["pred"], index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(paths["folds"], index=False, encoding="utf-8-sig")
    selected_rows = [
        {"feature": feature, "selected_in_folds": int(count)}
        for feature, count in selected_counts.items()
    ]
    pd.DataFrame(selected_rows, columns=["feature", "selected_in_folds"]).to_csv(
        paths["selected"], index=False, encoding="utf-8-sig"
    )


def load_config_progress(progress_dir: Path, scenario: str, config_name: str):
    """Retorna None se nao houver checkpoint completo e estruturalmente valido."""
    paths = progress_file_paths(progress_dir, scenario, config_name)
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        pred = pd.read_csv(paths["pred"])
        folds = pd.read_csv(paths["folds"])
        selected_df = pd.read_csv(paths["selected"])
        if (len(pred) != EXPECTED_PARTICIPANTS
                or pred["participant_id"].nunique() != EXPECTED_PARTICIPANTS
                or pred["configuration"].nunique() != 1
                or pred["configuration"].iloc[0] != config_name
                or pred["scenario"].nunique() != 1
                or pred["scenario"].iloc[0] != scenario
                or len(folds) != N_SPLITS
                or set(pd.to_numeric(folds["fold"], errors="raise").astype(int)) != set(range(1, N_SPLITS + 1))):
            return None
        selected_counts = Counter()
        if not selected_df.empty:
            selected_counts.update({str(row.feature): int(row.selected_in_folds)
                                    for row in selected_df.itertuples(index=False)})
        return pred, folds.to_dict("records"), selected_counts
    except Exception:
        return None


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    y_pred = (scores >= 0.0).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, scores) if np.unique(y_true).size == 2 else np.nan
    ap = average_precision_score(y_true, scores) if np.unique(y_true).size == 2 else np.nan
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc),
        "average_precision": float(ap),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_participants": int(len(y_true)),
        "n_normal": int(np.sum(y_true == 0)),
        "n_pathological": int(np.sum(y_true == 1)),
    }


def fast_ba_auc(y, scores):
    """Balanced accuracy (limiar 0) e AUC por ranks, com tratamento de empates."""
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return np.nan, np.nan
    ba = 0.5 * (float(np.mean(pos >= 0.0)) + float(np.mean(neg < 0.0)))
    ranks = rankdata(np.concatenate([pos, neg]), method="average")
    n_pos, n_neg = len(pos), len(neg)
    auc = (float(np.sum(ranks[:n_pos])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return ba, float(auc)


def bootstrap_metric_intervals(y, scores, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    """IC percentil estratificado por classe, reamostrando participantes OOF."""
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if not len(pos) or not len(neg):
        return {"balanced_accuracy_ci95_low": np.nan, "balanced_accuracy_ci95_high": np.nan,
                "auc_ci95_low": np.nan, "auc_ci95_high": np.nan, "bootstrap_valid_iterations": 0}
    rng = np.random.default_rng(seed)
    ba_values = np.empty(iterations, dtype=np.float64)
    auc_values = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        bp = pos[rng.integers(0, len(pos), size=len(pos))]
        bn = neg[rng.integers(0, len(neg), size=len(neg))]
        ba_values[i], auc_values[i] = fast_ba_auc(
            np.concatenate([np.ones(len(bp), dtype=np.int8), np.zeros(len(bn), dtype=np.int8)]),
            np.concatenate([bp, bn]),
        )
    return {
        "balanced_accuracy_ci95_low": float(np.quantile(ba_values, 0.025)),
        "balanced_accuracy_ci95_high": float(np.quantile(ba_values, 0.975)),
        "auc_ci95_low": float(np.quantile(auc_values, 0.025)),
        "auc_ci95_high": float(np.quantile(auc_values, 0.975)),
        "bootstrap_valid_iterations": int(iterations),
    }


def bootstrap_paired_delta(y, score_a, score_b, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    """IC bootstrap pareado: mesmas pessoas reamostradas nos dois classificadores."""
    y = np.asarray(y, dtype=np.int8)
    score_a = np.asarray(score_a, dtype=np.float64)
    score_b = np.asarray(score_b, dtype=np.float64)
    if not (len(y) == len(score_a) == len(score_b)):
        raise ModelingError("Bootstrap pareado recebeu vetores de tamanhos diferentes.")
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if not len(pos_idx) or not len(neg_idx):
        raise ModelingError("Bootstrap pareado precisa das duas classes.")
    rng = np.random.default_rng(seed)
    delta_ba = np.empty(iterations, dtype=np.float64)
    delta_auc = np.empty(iterations, dtype=np.float64)
    wins_ba = wins_auc = 0
    for i in range(iterations):
        idx = np.concatenate([
            pos_idx[rng.integers(0, len(pos_idx), size=len(pos_idx))],
            neg_idx[rng.integers(0, len(neg_idx), size=len(neg_idx))],
        ])
        yb = y[idx]
        ba_a, auc_a = fast_ba_auc(yb, score_a[idx])
        ba_b, auc_b = fast_ba_auc(yb, score_b[idx])
        delta_ba[i] = ba_a - ba_b
        delta_auc[i] = auc_a - auc_b
        wins_ba += int(delta_ba[i] > 0)
        wins_auc += int(delta_auc[i] > 0)
    return {
        "delta_balanced_accuracy_ci95_low": float(np.quantile(delta_ba, 0.025)),
        "delta_balanced_accuracy_ci95_high": float(np.quantile(delta_ba, 0.975)),
        "bootstrap_fraction_balanced_accuracy_wavelet_better": float(wins_ba / iterations),
        "delta_auc_ci95_low": float(np.quantile(delta_auc, 0.025)),
        "delta_auc_ci95_high": float(np.quantile(delta_auc, 0.975)),
        "bootstrap_fraction_auc_wavelet_better": float(wins_auc / iterations),
        "bootstrap_valid_iterations": int(iterations),
    }


def load_metadata(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise ModelingError(f"Metadata {name} nao encontrado: {path}")
    df = pd.read_csv(path)
    required = {"sample_id", "participant_id", "task", "y", "duration_s"}
    missing = required - set(df.columns)
    if missing:
        raise ModelingError(f"Colunas ausentes no metadata {name}: {sorted(missing)}")
    df = df.copy()
    for col in ("sample_id", "participant_id"):
        if df[col].isna().any():
            raise ModelingError(f"Metadata {name}: {col} possui valores ausentes.")
        df[col] = df[col].astype(str).str.strip().str.upper()
    df["task"] = df["task"].astype(str).str.strip().str.zfill(3)
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(np.int64)
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    if df["sample_id"].duplicated().any():
        raise ModelingError(f"Metadata {name}: sample_id duplicado.")
    if not set(df["y"].unique()).issubset({0, 1}):
        raise ModelingError(f"Metadata {name}: y deve ser 0 ou 1.")
    return df


def align_by_sample_id(reference: pd.DataFrame, other_meta: pd.DataFrame, matrix: np.ndarray, description: str):
    """Reordena matriz auxiliar para coincidir exatamente com a ordem Wavelet."""
    ref_ids = reference["sample_id"].tolist()
    other_ids = other_meta["sample_id"].tolist()
    missing = sorted(set(ref_ids) - set(other_ids))
    extra = sorted(set(other_ids) - set(ref_ids))
    if missing or extra:
        raise ModelingError(
            f"A matriz {description} nao corresponde ao manifesto Wavelet. "
            f"Ausentes={len(missing)} (ex.: {missing[:5]}), extras={len(extra)} (ex.: {extra[:5]}). "
            "Nao farei comparacao em coortes diferentes."
        )
    lookup = {sid: i for i, sid in enumerate(other_ids)}
    order = np.asarray([lookup[sid] for sid in ref_ids], dtype=np.int64)
    aligned_meta = other_meta.iloc[order].reset_index(drop=True)
    for col in ("participant_id", "task", "y"):
        left = reference[col].astype(str).to_numpy()
        right = aligned_meta[col].astype(str).to_numpy()
        if not np.array_equal(left, right):
            bad = np.flatnonzero(left != right)[:5]
            raise ModelingError(f"Divergencia de {col} entre Wavelet e {description}, linhas {bad.tolist()}.")
    if matrix.shape[0] != len(other_meta):
        raise ModelingError(f"Linhas de matriz {description} ({matrix.shape[0]}) != metadata ({len(other_meta)}).")
    return matrix[order]


def load_inputs():
    for path in (WAVELET_NPZ, WAVELET_META, BASELINE_NPZ):
        if not path.exists():
            raise ModelingError(
                f"Arquivo necessario nao encontrado: {path}\n"
                "Confira se as extracoes Wavelet temporal e ComParE/MFCC foram concluidas."
            )

    baseline_meta_path = next(
        (path for path in BASELINE_META_CANDIDATES if path.exists()), None
    )
    if baseline_meta_path is None:
        expected = " ou ".join(str(p) for p in BASELINE_META_CANDIDATES)
        raise ModelingError(
            "Metadata da extracao ComParE/MFCC nao encontrado. Procurei: " + expected +
            "\nO script extrair_features_capev_compare_avfad.py informado no projeto "
            "declara OUTPUT_META = metadata_capev_avfad.csv."
        )
    say(f"Metadata ComParE/MFCC utilizado: {baseline_meta_path.name}")

    say("Carregando caracteristicas Wavelet...")
    with np.load(WAVELET_NPZ, allow_pickle=False) as data:
        if not {"wavelet", "feature_names"}.issubset(data.files):
            raise ModelingError("NPZ Wavelet precisa conter wavelet e feature_names.")
        X_wave = np.asarray(data["wavelet"], dtype=np.float32)
        wave_names = np.asarray(data["feature_names"]).astype(str)

    wave_meta = load_metadata(WAVELET_META, "Wavelet")
    if X_wave.ndim != 2 or X_wave.shape != (len(wave_meta), len(wave_names)):
        raise ModelingError(f"Dimensoes Wavelet inconsistentes: matriz={X_wave.shape}, metadata={len(wave_meta)}, nomes={len(wave_names)}.")
    if X_wave.shape[1] != EXPECTED_WAVELET_FEATURES:
        raise ModelingError(f"Esperadas {EXPECTED_WAVELET_FEATURES} features Wavelet; encontrei {X_wave.shape[1]}.")

    say("Carregando MFCC (e ComParE, se disponivel) da extracao de baseline...")
    with np.load(BASELINE_NPZ, allow_pickle=False) as data:
        if "acoustic" not in data.files or "acoustic_names" not in data.files:
            raise ModelingError("NPZ baseline precisa conter acoustic e acoustic_names para comparacao MFCC.")
        acoustic = np.asarray(data["acoustic"], dtype=np.float32)
        acoustic_names = np.asarray(data["acoustic_names"]).astype(str)
        compare = np.asarray(data["compare"], dtype=np.float32) if "compare" in data.files else None
        compare_names = np.asarray(data["compare_names"]).astype(str) if "compare_names" in data.files else None

    baseline_meta = load_metadata(baseline_meta_path, "MFCC/ComParE")
    if acoustic.ndim != 2 or acoustic.shape != (len(baseline_meta), len(acoustic_names)):
        raise ModelingError(f"Dimensoes acoustic inconsistentes: matriz={acoustic.shape}, metadata={len(baseline_meta)}, nomes={len(acoustic_names)}.")
    mfcc_cols = np.asarray([
        i for i, name in enumerate(acoustic_names)
        if name.startswith(("mfcc_", "delta_", "delta2_"))
    ], dtype=np.int64)
    if len(mfcc_cols) < 100:
        raise ModelingError(f"Encontrei apenas {len(mfcc_cols)} features MFCC/delta; esperava aproximadamente 150.")
    acoustic_aligned = align_by_sample_id(wave_meta, baseline_meta, acoustic, "MFCC")
    X_mfcc = acoustic_aligned[:, mfcc_cols]
    mfcc_names = acoustic_names[mfcc_cols]

    X_compare = None
    if compare is not None and compare_names is not None:
        if compare.ndim != 2 or compare.shape != (len(baseline_meta), len(compare_names)):
            raise ModelingError("Dimensoes da matriz ComParE nao correspondem ao metadata.")
        compare_aligned = align_by_sample_id(wave_meta, baseline_meta, compare, "ComParE")
        X_compare = compare_aligned
        compare_names = compare_names

    # Infinitos viram NaN para serem tratados pelo imputador dentro do fold.
    def sanitize_nonfinite(matrix):
        arr = np.asarray(matrix, dtype=np.float32).copy()
        arr[~np.isfinite(arr)] = np.nan
        return arr

    X_wave = sanitize_nonfinite(X_wave)
    X_mfcc = sanitize_nonfinite(X_mfcc)
    if X_compare is not None:
        X_compare = sanitize_nonfinite(X_compare)

    if len(wave_meta) != EXPECTED_RECORDINGS or wave_meta["participant_id"].nunique() != EXPECTED_PARTICIPANTS:
        raise ModelingError("A coorte Wavelet nao tem as 2465 gravacoes e 411 participantes esperados.")
    people = wave_meta.drop_duplicates("participant_id")
    class_counts = people["y"].value_counts().to_dict()
    if class_counts.get(0, 0) != EXPECTED_NORMAL or class_counts.get(1, 0) != EXPECTED_PATHOLOGICAL:
        raise ModelingError(f"Classes inesperadas: {class_counts}; esperado Normal=82 e Pathological=329.")
    labels_person = wave_meta.groupby("participant_id")["y"].nunique()
    if (labels_person != 1).any():
        raise ModelingError("Existe participante com rotulos divergentes entre gravacoes.")
    return X_wave, wave_names, X_mfcc, mfcc_names, X_compare, compare_names, wave_meta


def make_configurations(wave_names, X_wave, X_mfcc, mfcc_names, X_compare, compare_names):
    masks = {
        "DWT_db4": np.array([n.startswith("DWT__db4__") for n in wave_names]),
        "DWT_sym5": np.array([n.startswith("DWT__sym5__") for n in wave_names]),
        "WPT_db4": np.array([n.startswith("WPT__db4__") for n in wave_names]),
        "WPT_sym5": np.array([n.startswith("WPT__sym5__") for n in wave_names]),
    }
    config = {
        "DWT_db4": (X_wave, np.flatnonzero(masks["DWT_db4"]), wave_names),
        "DWT_sym5": (X_wave, np.flatnonzero(masks["DWT_sym5"]), wave_names),
        "WPT_db4": (X_wave, np.flatnonzero(masks["WPT_db4"]), wave_names),
        "WPT_sym5": (X_wave, np.flatnonzero(masks["WPT_sym5"]), wave_names),
        "db4_DWT_WPT": (X_wave, np.flatnonzero(masks["DWT_db4"] | masks["WPT_db4"]), wave_names),
        "sym5_DWT_WPT": (X_wave, np.flatnonzero(masks["DWT_sym5"] | masks["WPT_sym5"]), wave_names),
        "db4_sym5_DWT_WPT": (X_wave, np.flatnonzero(masks["DWT_db4"] | masks["WPT_db4"] | masks["DWT_sym5"] | masks["WPT_sym5"]), wave_names),
        "MFCC_baseline": (X_mfcc, np.arange(X_mfcc.shape[1]), mfcc_names),
    }
    if X_compare is not None and compare_names is not None:
        config["ComParE2016_baseline"] = (X_compare, np.arange(X_compare.shape[1]), compare_names)

    expected = {"DWT_db4": 160, "DWT_sym5": 160, "WPT_db4": 1920, "WPT_sym5": 1920,
                "db4_DWT_WPT": 2080, "sym5_DWT_WPT": 2080, "db4_sym5_DWT_WPT": 4160}
    for name, count in expected.items():
        actual = len(config[name][1])
        if actual != count:
            raise ModelingError(f"Configuracao {name}: {actual} features; esperado {count}.")
    say("\n=== CONFIGURACOES ===")
    for name, (_, idx, _) in config.items():
        say(f"{name}: {len(idx)} features originais")
    return config


def create_participant_folds(df: pd.DataFrame):
    """Estratifica uma linha por participante e projeta os folds nas gravacoes."""
    people = df[["participant_id", "y"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True)
    if people["participant_id"].duplicated().any():
        raise ModelingError("A tabela de participantes nao e unica por pessoa.")
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_by_person = {}
    dummy = np.zeros((len(people), 1), dtype=np.uint8)
    for fold, (_, test_idx) in enumerate(splitter.split(dummy, people["y"].to_numpy(), groups=people["participant_id"].to_numpy())):
        for idx in test_idx:
            pid = people.iloc[idx]["participant_id"]
            if pid in fold_by_person:
                raise ModelingError(f"Participante {pid} em mais de um fold de teste.")
            fold_by_person[pid] = fold
    fold_by_row = df["participant_id"].map(fold_by_person).to_numpy(dtype=np.int64)
    if pd.isna(fold_by_row).any():
        raise ModelingError("Existem gravacoes sem fold.")
    assignments = people.copy()
    assignments["fold"] = assignments["participant_id"].map(fold_by_person) + 1
    assignments["n_recordings"] = assignments["participant_id"].map(df.groupby("participant_id").size())
    for fold in range(N_SPLITS):
        sub = assignments[assignments["fold"] == fold + 1]
        say(f"Fold {fold + 1}: {len(sub)} participantes; Normal={(sub.y == 0).sum()}; Pathological={(sub.y == 1).sum()}; gravacoes={(fold_by_row == fold).sum()}")
    return fold_by_person, fold_by_row, assignments


def fit_one_fold(X, feature_indices, feature_names, df, train_idx, test_idx, scenario, config_name, fold):
    y_all = df["y"].to_numpy(dtype=np.int64)
    # Seleciona apenas colunas do modelo; os nao finitos sao imputados no treino.
    x_train = X[np.ix_(train_idx, feature_indices)]
    x_test = X[np.ix_(test_idx, feature_indices)]
    y_train = y_all[train_idx]
    # Remove colunas inteiramente ausentes no treino antes da imputacao;
    # assim os nomes das features continuam alinhados mesmo em sklearn antigo.
    nonempty = ~np.isnan(x_train).all(axis=0)
    x_train = x_train[:, nonempty]
    x_test = x_test[:, nonempty]
    candidate_names = feature_names[feature_indices][nonempty]
    if x_train.shape[1] == 0:
        raise ModelingError(f"Todas as features estao ausentes em {config_name}, fold {fold}.")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    variance = VarianceThreshold(threshold=0.0)
    x_train = variance.fit_transform(x_train)
    x_test = variance.transform(x_test)
    remaining_names = candidate_names[variance.get_support()]
    if x_train.shape[1] == 0:
        raise ModelingError(f"Nenhuma feature variavel em {config_name}, fold {fold}.")
    k = min(TOP_K, x_train.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = SelectKBest(score_func=f_classif, k=k)
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
    selected_names = remaining_names[selector.get_support()]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = SVC(kernel="linear", C=SVC_C, class_weight="balanced", cache_size=512)
    model.fit(x_train, y_train)
    recording_scores = model.decision_function(x_test)
    test_meta = df.iloc[test_idx][["participant_id", "y"]].copy()
    test_meta["decision_score"] = recording_scores
    people = test_meta.groupby("participant_id", as_index=False).agg(
        y=("y", "first"), decision_score=("decision_score", "mean"),
        n_recordings=("decision_score", "size"),
    )
    people["fold"] = fold
    people["scenario"] = scenario
    people["configuration"] = config_name
    people["label"] = np.where(people["y"] == 1, "Pathological", "Normal")
    people["prediction"] = (people["decision_score"] >= 0).astype(np.int64)
    people["correct"] = (people["prediction"] == people["y"]).astype(np.int8)
    fold_metrics = metrics_from_scores(people["y"].to_numpy(), people["decision_score"].to_numpy())
    fold_row = {
        "scenario": scenario, "configuration": config_name, "fold": fold,
        "input_features": int(len(feature_indices)), "features_after_variance": int(len(remaining_names)),
        "selected_features": int(len(selected_names)),
        "train_recordings": int(len(train_idx)), "test_recordings": int(len(test_idx)),
        "train_participants": int(df.iloc[train_idx]["participant_id"].nunique()),
        "test_participants": int(people["participant_id"].nunique()),
        **fold_metrics,
    }
    return people, fold_row, selected_names


def evaluate_config(config_name, X, indices, names, df, fold_by_row, keep_mask, scenario):
    pred_parts, fold_rows = [], []
    selected_counts = Counter()
    for fold in range(1, N_SPLITS + 1):
        train_mask = keep_mask & (fold_by_row != fold - 1)
        test_mask = keep_mask & (fold_by_row == fold - 1)
        train_idx, test_idx = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
        train_people = set(df.iloc[train_idx]["participant_id"])
        test_people = set(df.iloc[test_idx]["participant_id"])
        if train_people & test_people:
            raise ModelingError(f"Vazamento de participante em {config_name}, fold {fold}.")
        if not test_people or set(df.iloc[test_idx]["y"].unique()) != {0, 1}:
            raise ModelingError(f"Fold {fold} sem participantes ou sem ambas as classes no teste.")
        pred, fold_row, selected = fit_one_fold(
            X, indices, names, df, train_idx, test_idx, scenario, config_name, fold
        )
        pred_parts.append(pred)
        fold_rows.append(fold_row)
        if scenario == "completo":
            selected_counts.update(selected.tolist())
        m = fold_row
        say(f"  {scenario} | {config_name} | fold {fold}/{N_SPLITS} | BA={m['balanced_accuracy']:.3f} | AUC={m['auc']:.3f}")
    pred_df = pd.concat(pred_parts, ignore_index=True).sort_values("participant_id").reset_index(drop=True)
    if len(pred_df) != EXPECTED_PARTICIPANTS or pred_df["participant_id"].nunique() != EXPECTED_PARTICIPANTS:
        raise ModelingError(f"{scenario}/{config_name}: esperava 411 predicoes OOF unicas; encontrei {len(pred_df)}.")
    if pred_df["participant_id"].duplicated().any():
        raise ModelingError(f"Predicao OOF duplicada em {scenario}/{config_name}.")
    return pred_df, fold_rows, selected_counts


def metric_summary_row(scenario, name, n_features, pred_df, fold_rows, bootstrap_seed):
    pooled = metrics_from_scores(pred_df["y"].to_numpy(), pred_df["decision_score"].to_numpy())
    ci = bootstrap_metric_intervals(pred_df["y"].to_numpy(), pred_df["decision_score"].to_numpy(), seed=bootstrap_seed)
    fdf = pd.DataFrame(fold_rows)
    row = {"scenario": scenario, "configuration": name, "input_features": int(n_features),
           "selected_features_mean": float(fdf["selected_features"].mean()),
           **{f"{k}_oof": v for k, v in pooled.items()}, **ci}
    for metric in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "average_precision", "mcc"]:
        row[f"{metric}_fold_mean"] = float(fdf[metric].mean())
        row[f"{metric}_fold_sd"] = float(fdf[metric].std(ddof=1))
    return row


def paired_comparisons(predictions: pd.DataFrame, fold_table: pd.DataFrame, wavelet_names):
    rows = []
    for scenario in predictions["scenario"].unique():
        subset = predictions[predictions["scenario"] == scenario]
        ref = subset[subset["configuration"] == "MFCC_baseline"].sort_values("participant_id")
        for name in wavelet_names:
            wav = subset[subset["configuration"] == name].sort_values("participant_id")
            if len(wav) != len(ref) or not np.array_equal(wav["participant_id"].to_numpy(), ref["participant_id"].to_numpy()):
                raise ModelingError(f"Comparacao pareada {name} vs MFCC sem participantes identicos ({scenario}).")
            y = ref["y"].to_numpy(dtype=np.int64)
            sm = metrics_from_scores(ref["y"].to_numpy(), ref["decision_score"].to_numpy())
            sw = metrics_from_scores(wav["y"].to_numpy(), wav["decision_score"].to_numpy())
            boot = bootstrap_paired_delta(y, wav["decision_score"].to_numpy(), ref["decision_score"].to_numpy(), seed=BOOTSTRAP_SEED + len(rows))
            row = {"scenario": scenario, "wavelet_configuration": name, "reference": "MFCC_baseline",
                   "n_participants": len(y)}
            for metric in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "average_precision", "mcc"]:
                row[f"{metric}_wavelet"] = sw[metric]
                row[f"{metric}_mfcc"] = sm[metric]
                row[f"delta_{metric}_wavelet_minus_mfcc"] = sw[metric] - sm[metric]
            row.update(boot)
            fw = fold_table[(fold_table["scenario"] == scenario) & (fold_table["configuration"] == name)]
            fm = fold_table[(fold_table["scenario"] == scenario) & (fold_table["configuration"] == "MFCC_baseline")]
            joined = fw.merge(fm, on="fold", suffixes=("_wavelet", "_mfcc"))
            for metric in ["balanced_accuracy", "auc"]:
                diffs = joined[f"{metric}_wavelet"] - joined[f"{metric}_mfcc"]
                row[f"delta_{metric}_fold_mean"] = float(diffs.mean())
                row[f"delta_{metric}_fold_sd"] = float(diffs.std(ddof=1))
                row[f"folds_wavelet_better_{metric}"] = int((diffs > 0).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(x, digits=4):
    return "n/d" if pd.isna(x) else f"{float(x):.{digits}f}"


def pct(x):
    return "n/d" if pd.isna(x) else f"{100*float(x):.2f}%"


def build_report(summary, folds, paired, feature_freq, assignments, excluded, metadata, configs, elapsed):
    md = []
    md += [
        "# Relatório técnico detalhado — Wavelet temporal vs MFCC na AVFAD",
        "",
        f"**Execução:** {datetime.now().isoformat(timespec='seconds')}  ",
        f"**Python / scikit-learn:** {sys.version.split()[0]} / {sklearn.__version__}  ",
        f"**Tempo de modelagem:** {elapsed/60:.1f} min  ",
        f"**Bootstrap:** {BOOTSTRAP_ITERATIONS} reamostragens por participante, seed {BOOTSTRAP_SEED}",
        "",
        "> **Escopo deste relatório:** documenta a extração temporal Wavelet já realizada, reavalia MFCC com os mesmos folds desta execução e compara diretamente os modelos nos mesmos participantes. Não confundir esta análise com os resultados históricos, que são apresentados separadamente.",
        "",
        "## 1. Resumo executivo",
    ]
    primary = summary[summary["scenario"] == "completo"]
    wavelet_names = ["DWT_db4", "DWT_sym5", "WPT_db4", "WPT_sym5", "db4_DWT_WPT", "sym5_DWT_WPT", "db4_sym5_DWT_WPT"]
    wave_primary = primary[primary["configuration"].isin(wavelet_names)].sort_values(["balanced_accuracy_oof", "auc_oof"], ascending=False)
    mfcc_row = primary[primary["configuration"] == "MFCC_baseline"].iloc[0]
    compare_row = primary[primary["configuration"] == "ComParE2016_baseline"]
    if not wave_primary.empty:
        best = wave_primary.iloc[0]
        paired_best = paired[(paired["scenario"] == "completo") & (paired["wavelet_configuration"] == best["configuration"])].iloc[0]
        md += [
            f"- **Melhor configuração Wavelet por balanced accuracy OOF:** `{best['configuration']}` — BA {fmt(best['balanced_accuracy_oof'])}, AUC {fmt(best['auc_oof'])}.",
            f"- **MFCC reavaliado nos mesmos folds:** BA {fmt(mfcc_row['balanced_accuracy_oof'])}, AUC {fmt(mfcc_row['auc_oof'])}, sensibilidade {fmt(mfcc_row['sensitivity_oof'])}, especificidade {fmt(mfcc_row['specificity_oof'])}.",
            f"- **Diferença pareada do melhor Wavelet contra MFCC:** ΔBA {fmt(paired_best['delta_balanced_accuracy_wavelet_minus_mfcc'], 4)} (IC95% bootstrap {fmt(paired_best['delta_balanced_accuracy_ci95_low'])} a {fmt(paired_best['delta_balanced_accuracy_ci95_high'])}); ΔAUC {fmt(paired_best['delta_auc_wavelet_minus_mfcc'], 4)} (IC95% {fmt(paired_best['delta_auc_ci95_low'])} a {fmt(paired_best['delta_auc_ci95_high'])}).",
            f"- **Folds em que o melhor Wavelet superou MFCC:** BA {int(paired_best['folds_wavelet_better_balanced_accuracy'])}/5; AUC {int(paired_best['folds_wavelet_better_auc'])}/5.",
        ]
        ci_low, ci_high = paired_best["delta_balanced_accuracy_ci95_low"], paired_best["delta_balanced_accuracy_ci95_high"]
        if ci_low > 0:
            md.append("- **Leitura provisória:** o intervalo bootstrap pareado de ΔBA ficou acima de zero nesta avaliação; isso favorece a configuração Wavelet selecionada, mas não substitui validação externa nem corrige o otimismo de escolher o melhor entre sete configurações no mesmo CV.")
        elif ci_high < 0:
            md.append("- **Leitura provisória:** o intervalo bootstrap pareado de ΔBA ficou abaixo de zero; nesta avaliação, MFCC teve vantagem sobre a configuração Wavelet selecionada.")
        else:
            md.append("- **Leitura provisória:** o intervalo bootstrap pareado de ΔBA inclui zero; não há evidência suficiente nesta avaliação para afirmar superioridade estatística de um sobre o outro.")
    md += [
        "",
        "> A conclusão principal deve considerar simultaneamente balanced accuracy, AUC, sensibilidade, especificidade, variabilidade entre folds e diferença pareada. Acurácia isolada não é adequada para esta coorte desbalanceada.",
        "",
        "## 2. População, coorte e unidade de análise",
        f"- Centro: HSA; participantes: {metadata['participant_id'].nunique()}.",
        f"- Classes: Normal={int((metadata.drop_duplicates('participant_id')['y']==0).sum())}; Pathological={int((metadata.drop_duplicates('participant_id')['y']==1).sum())}.",
        f"- Gravações disponíveis: {len(metadata)}; tarefas CAPE-V: 004–009.",
        f"- Participantes com seis gravações: {int((metadata.groupby('participant_id').size()==6).sum())}; com cinco gravações: {int((metadata.groupby('participant_id').size()==5).sum())}.",
        "- CCO não possui tarefa 004 (arquivo anteriormente diagnosticado como truncado, aproximadamente 0,093 s); CCO foi mantido com as outras cinco gravações válidas.",
        "- Unidade final das métricas: participante. As gravações do participante são usadas no treino e os decision scores de suas frases são calculados em conjunto no fold de teste, depois agregados pela média para uma única predição OOF por pessoa.",
        "",
        "### Distribuição de folds por participante",
        "| Fold | Participantes | Normais | Patológicos | Gravações |",
        "|---:|---:|---:|---:|---:|",
    ]
    for fold, group in assignments.groupby("fold"):
        n = len(group)
        normal = int((group["y"] == 0).sum())
        path = int((group["y"] == 1).sum())
        records = int(group["n_recordings"].sum())
        md.append(f"| {int(fold)} | {int(n)} | {normal} | {path} | {records} |")
    md += [
        "",
        "## 3. Extração Wavelet temporal (já concluída)",
        "- Famílias: db4 e sym5.",
        "- DWT: nível 7 (A7 e D7 até D1). WPT: níveis 5 e 6.",
        "- Descritores por banda/nó: energia relativa, log-energy, entropia, curtose e ZCR.",
        "- VAD: `librosa.effects.split`, `top_db=40`, `frame_length=2048`, `hop_length=480`; pausas até 0,25 s unidas e região mínima de 0,25 s.",
        "- Janelas: 400 ms, passo 200 ms.",
        "- Estatísticas temporais por descritor: média, desvio-padrão, q10 e q90.",
        "- Dimensão: 4.160 por gravação (db4 2.080 + sym5 2.080). Matriz observada: 2.465 × 4.160, float32.",
        "- Extração: 2.465/2.465 gravações, zero erros registrados, zero NaN e zero infinito na auditoria anterior; tempo informado: 192,4 min.",
        "- Regiões de fala por gravação: mínimo 1, mediana 1, máximo 3. Janelas: mínimo 13, mediana 47, máximo 1.385.",
        "",
        "## 4. MFCC usado como referência direta",
        f"- Dimensão MFCC selecionada do NPZ de baseline: {len(configs['MFCC_baseline'][1])} características por gravação.",
        "- Composição documentada na extração anterior: 25 MFCC + delta + delta-delta; média e desvio por gravação (150 dimensões esperadas).",
        "- O MFCC de referência foi calculado sobre o WAV completo, sem VAD/recorte temporal por janelas. A Wavelet temporal usa VAD e resume descritores de janelas de 400 ms. Portanto, a comparação controla participantes, folds, classificador e pipeline de aprendizagem, mas não iguala integralmente o front-end temporal. Esta diferença deve ser explicitada na interpretação.",
        "- O script alinha as matrizes por `sample_id` e confirma participante, tarefa e rótulo antes de treinar. Não depende da ordem original dos arquivos NPZ.",
        "",
        "## 5. Modelos e configurações comparadas",
        "| Configuração | Entrada antes da seleção |",
        "|---|---:|",
    ]
    for name, (_, idx, _) in configs.items():
        md.append(f"| {name} | {len(idx):,} |")
    md += [
        "",
        f"- Classificador: SVC linear, C={SVC_C}, `class_weight='balanced'`.",
        f"- Seleção: `SelectKBest(f_classif)`, até {TOP_K} características por fold.",
        "- Pré-processamento: imputação pela mediana → remoção de colunas constantes → seleção → StandardScaler. Todos os componentes são ajustados somente com os registros do treino de cada fold.",
        "- Limiar: decision score ≥ 0 classifica como Pathological. AUC usa decision score contínuo, não probabilidades calibradas. A classe positiva (y=1) é Pathological; sensibilidade mede detecção de patologia e especificidade mede reconhecimento de Normal.",
        "- Mesmo particionamento de participantes para todas as configurações e cenários.",
        "",
        "## 6. Resultados principais — todos os áudios",
        "| Modelo | Features de entrada | BA OOF | AUC OOF | Sens. | Esp. | F1 | MCC | BA folds (média ± DP) | AUC folds (média ± DP) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary.sort_values("balanced_accuracy_oof", ascending=False).itertuples(index=False):
        d = row._asdict()
        md.append(f"| {d['configuration']} | {int(d['input_features']):,} | {fmt(d['balanced_accuracy_oof'])} | {fmt(d['auc_oof'])} | {fmt(d['sensitivity_oof'])} | {fmt(d['specificity_oof'])} | {fmt(d['f1_oof'])} | {fmt(d['mcc_oof'])} | {fmt(d['balanced_accuracy_fold_mean'])} ± {fmt(d['balanced_accuracy_fold_sd'])} | {fmt(d['auc_fold_mean'])} ± {fmt(d['auc_fold_sd'])} |")
    md += [
        "",
        "### Intervalos bootstrap dos resultados OOF",
        "Os ICs abaixo são percentis de bootstrap de participantes sobre as predições OOF já obtidas. Eles são condicionais a este conjunto de folds e não incorporam toda a variabilidade de repetir o treinamento e o particionamento.",
        "| Modelo | IC95% BA | IC95% AUC |",
        "|---|---:|---:|",
    ]
    for row in primary.sort_values("balanced_accuracy_oof", ascending=False).itertuples(index=False):
        d=row._asdict(); md.append(f"| {d['configuration']} | {fmt(d['balanced_accuracy_ci95_low'])}–{fmt(d['balanced_accuracy_ci95_high'])} | {fmt(d['auc_ci95_low'])}–{fmt(d['auc_ci95_high'])} |")
    md += [
        "",
        "### Matrizes de confusão agregadas (OOF, participante)",
        "TN = Normal corretamente predito; FP = Normal predito como Pathological; FN = Pathological predito como Normal; TP = Pathological corretamente predito.",
        "| Modelo | TN | FP | FN | TP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary.sort_values("balanced_accuracy_oof", ascending=False).itertuples(index=False):
        d = row._asdict()
        md.append(f"| {d['configuration']} | {int(d['tn_oof'])} | {int(d['fp_oof'])} | {int(d['fn_oof'])} | {int(d['tp_oof'])} |")
    md += [
        "",
        "## 7. Comparação pareada Wavelet vs MFCC",
        "Cada delta é calculado para exatamente os mesmos participantes OOF: valor positivo favorece Wavelet; negativo favorece MFCC. O IC bootstrap também é pareado, reamostrando os mesmos participantes para os dois modelos.",
        "| Wavelet | ΔBA (Wavelet−MFCC) | IC95% ΔBA | Fração bootstrap ΔBA>0 | ΔAUC | IC95% ΔAUC | Fração bootstrap ΔAUC>0 | Folds BA melhores/5 | Folds AUC melhores/5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    pp = paired[paired["scenario"] == "completo"].sort_values("delta_balanced_accuracy_wavelet_minus_mfcc", ascending=False)
    for row in pp.itertuples(index=False):
        d=row._asdict(); md.append(f"| {d['wavelet_configuration']} | {fmt(d['delta_balanced_accuracy_wavelet_minus_mfcc'])} | {fmt(d['delta_balanced_accuracy_ci95_low'])}–{fmt(d['delta_balanced_accuracy_ci95_high'])} | {pct(d['bootstrap_fraction_balanced_accuracy_wavelet_better'])} | {fmt(d['delta_auc_wavelet_minus_mfcc'])} | {fmt(d['delta_auc_ci95_low'])}–{fmt(d['delta_auc_ci95_high'])} | {pct(d['bootstrap_fraction_auc_wavelet_better'])} | {int(d['folds_wavelet_better_balanced_accuracy'])}/5 | {int(d['folds_wavelet_better_auc'])}/5 |")
    md += [
        "",
        "### Interpretação obrigatória da comparação",
        "- Um ΔBA/ΔAUC positivo é uma diferença pontual, não prova por si só superioridade.",
        "- Se o IC95% pareado do delta incluir zero, a evidência é inconclusiva para superioridade nesta avaliação.",
        "- Se o delta for positivo e o IC95% ficar inteiramente acima de zero, há evidência compatível com vantagem nesta coorte e neste protocolo, mas ainda se exige validação externa.",
        "- A escolha do melhor entre sete configurações usando os mesmos resultados OOF cria risco de otimismo por seleção de modelo. A análise é exploratória; uma escolha definitiva deve ser confirmada com nested CV ou conjunto externo independente.",
        "- AUC maior com BA menor (ou o inverso) indica trade-off entre ordenação dos scores e classificação no limiar zero. Interpretar também sensibilidade e especificidade.",
        "",
        "## 8. Resultados por fold",
        "| Cenário | Modelo | Fold | Participantes teste | BA | AUC | Sens. | Esp. | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds.sort_values(["scenario", "configuration", "fold"]).itertuples(index=False):
        md.append(f"| {row.scenario} | {row.configuration} | {row.fold} | {row.test_participants} | {fmt(row.balanced_accuracy)} | {fmt(row.auc)} | {fmt(row.sensitivity)} | {fmt(row.specificity)} | {fmt(row.f1)} |")
    md += [
        "",
        "## 9. Sensibilidade a gravações acima de 60 segundos",
        f"- Regra: excluir somente gravações com duração > {SENSITIVITY_MAX_DURATION_S:.0f} s; nenhum participante deve ser removido.",
        f"- Gravações removidas: {len(excluded)}; participantes mantidos: {metadata['participant_id'].nunique()}.",
        "| Modelo | BA OOF | AUC OOF | Sens. | Esp. | ΔBA vs cenário completo | ΔAUC vs cenário completo |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    sens = summary[summary["scenario"] == "sensibilidade_ate_60s"]
    for row in sens.itertuples(index=False):
        d=row._asdict(); full=primary[primary.configuration==d['configuration']].iloc[0]
        md.append(f"| {d['configuration']} | {fmt(d['balanced_accuracy_oof'])} | {fmt(d['auc_oof'])} | {fmt(d['sensitivity_oof'])} | {fmt(d['specificity_oof'])} | {fmt(d['balanced_accuracy_oof']-full['balanced_accuracy_oof'])} | {fmt(d['auc_oof']-full['auc_oof'])} |")
    md += ["", "### Gravações removidas no cenário de sensibilidade", "| Sample | Participante | Tarefa | Duração (s) | Caminho no ZIP |", "|---|---|---:|---:|---|"]
    if excluded.empty:
        md.append("| — | — | — | — | Nenhuma gravação >60 s |")
    else:
        for row in excluded.sort_values("duration_s", ascending=False).itertuples(index=False):
            path = getattr(row, "audio_path_in_zip", "")
            md.append(f"| {row.sample_id} | {row.participant_id} | {row.task} | {row.duration_s:.3f} | {path} |")
    long30 = metadata[metadata["duration_s"] > 30.0].copy()
    md += [
        "",
        "### Auditoria das gravações acima de 30 segundos (não significa exclusão)",
        "Estas gravações permanecem na análise principal. O objetivo é chamar atenção para casos que merecem inspeção auditiva/manual. A regra de sensibilidade aplicada acima remove apenas >60 s.",
        "| Sample | Participante | Tarefa | Duração (s) | Duração ativa (s) | Janelas | Regiões VAD | Speech ratio |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if long30.empty:
        md.append("| — | — | — | — | — | — | — | — |")
    else:
        for row in long30.sort_values("duration_s", ascending=False).itertuples(index=False):
            active = getattr(row, "active_duration_s", np.nan)
            windows = getattr(row, "window_count", np.nan)
            regions = getattr(row, "speech_regions", np.nan)
            ratio = getattr(row, "speech_ratio", np.nan)
            md.append(f"| {row.sample_id} | {row.participant_id} | {row.task} | {row.duration_s:.3f} | {fmt(active, 3)} | {int(windows) if pd.notna(windows) else 'n/d'} | {int(regions) if pd.notna(regions) else 'n/d'} | {fmt(ratio, 3)} |")
    md += [
        "",
        "## 10. Estabilidade das características selecionadas",
        "Frequência = número de folds (de cinco) em que a característica entrou no SelectKBest. Frequências altas sugerem recorrência dentro deste protocolo, mas cinco folds são uma medida limitada e as features correlacionadas podem substituir umas às outras.",
    ]
    for conf in wavelet_names:
        sub = feature_freq[feature_freq["configuration"] == conf].head(12)
        md += [f"### {conf}", "| Feature | Selecionada em folds |", "|---|---:|"]
        if sub.empty:
            md.append("| Nenhuma informação disponível | — |")
        else:
            for row in sub.itertuples(index=False):
                md.append(f"| `{row.feature}` | {row.selected_in_folds}/5 |")
    md += [
        "",
        "## 11. Resultados históricos anteriores (não são a comparação direta desta execução)",
        "Os valores a seguir vieram das avaliações anteriores registradas no histórico do projeto. São úteis como contexto, mas não devem ser misturados com os resultados novos: podem diferir em extração, pré-processamento, configuração e particionamento. A comparação primária Wavelet vs MFCC é a seção 7, produzida agora com os mesmos folds.",
        "| Representação histórica | BA pooled/OOF | AUC pooled/OOF | BA média folds ± DP | AUC média folds ± DP | Observação |",
        "|---|---:|---:|---:|---:|---|",
    ]
    historical = [
        ("ComParE 2016 balanceado", "0.5781", "0.6850", "0.5932 ± 0.0885", "0.6889 ± 0.0444", "Baseline mais forte na avaliação HSA anterior."),
        ("MFCC balanceado", "0.5402", "0.5999", "0.5457 ± 0.0582", "0.6052 ± 0.0392", "MFCC whole-recording; 149/150 dimensões utilizáveis conforme o pipeline."),
        ("DWT estática balanceada", "0.5769", "0.6029", "0.5753 ± 0.0634", "0.6073 ± 0.0550", "Versão anterior, não temporal."),
        ("WPT estática balanceada", "0.5617", "0.5893", "0.5651 ± 0.0689", "0.5939 ± 0.0718", "Versão anterior, não temporal."),
    ]
    for r in historical:
        md.append("| " + " | ".join(r) + " |")
    md += [
        "",
        "### Coortes pareadas históricas",
        "| Coorte | Modelo | BA média folds ± DP | AUC média folds ± DP |",
        "|---|---|---:|---:|",
        "| Sexo + idade (162; 81/81) | ComParE | 0.5857 ± 0.0679 | 0.6443 ± 0.0690 |",
        "| Sexo + idade (162; 81/81) | MFCC | 0.5059 ± 0.0536 | 0.4811 ± 0.0902 |",
        "| Sexo + idade (162; 81/81) | DWT estática | 0.5298 ± 0.0570 | 0.5251 ± 0.0422 |",
        "| Sexo + idade (162; 81/81) | WPT estática | 0.5551 ± 0.0952 | 0.5495 ± 0.0769 |",
        "| Sexo + idade + ano (136; 68/68) | ComParE | 0.6462 ± 0.0563 | 0.6802 ± 0.0684 |",
        "| Sexo + idade + ano (136; 68/68) | MFCC | 0.5753 ± 0.0810 | 0.6136 ± 0.0685 |",
        "| Sexo + idade + ano (136; 68/68) | DWT estática | 0.5720 ± 0.1126 | 0.6131 ± 0.1522 |",
        "| Sexo + idade + ano (136; 68/68) | WPT estática | 0.5863 ± 0.1116 | 0.6056 ± 0.1297 |",
        "",
        "### Contexto do desbalanceamento",
        "Na avaliação anterior, o ComParE sem `class_weight='balanced'` classificava todos como patológicos: acurácia 0.8005, balanced accuracy 0.5000, sensibilidade 1.0000 e especificidade 0.0000. Isso ilustra por que acurácia isolada é enganosa com 329 patológicos e 82 normais.",
        "Baseline demográfico idade/sexo/ano anterior: accuracy 0.6423, BA 0.5706, sensibilidade 0.6900, especificidade 0.4512, F1 0.7554, AUC 0.5435; média folds BA 0.5731 ± 0.0547 e AUC 0.5581 ± 0.1038. Esse baseline alerta para possíveis efeitos demográficos.",
        "",
        "## 12. Limitações e riscos de interpretação",
        "1. Coorte única HSA, 411 participantes, desbalanceada (82 normais e 329 patológicos); generalização para outros centros da AVFAD não foi demonstrada.",
        "2. MFCC foi extraído do WAV completo sem VAD; Wavelet temporal usa VAD/janelas. O modelo/participantes/folds são pareados, mas o front-end de sinal não é idêntico. Uma comparação totalmente controlada exigiria também extrair MFCC sobre exatamente as mesmas janelas de fala e resumir mean/std/q10/q90.",
        "3. Cada gravação é uma linha de treino; a métrica final é por participante após média dos scores. A seleção f_classif também ocorre em linhas de gravação; pessoas com seis gravações têm peso ligeiramente maior que CCO com cinco. Só uma pessoa tem essa diferença.",
        "4. Os scores SVC são margens de decisão, não probabilidades calibradas. O limiar zero foi mantido fixo; não foi otimizado no teste.",
        "5. TOP_K=128 e C=1 são escolhas fixas e não foram otimizadas. Isso reduz risco de tuning no teste, mas pode não ser a configuração ótima de cada representação.",
        "6. Selecionar o melhor dos sete modelos com base nos mesmos OOF é exploração de modelos e produz viés de seleção. Confirmar qualquer vencedor por nested CV ou base externa.",
        "7. Os intervalos bootstrap são condicionais às predições OOF e ao split observado; não substituem validação externa e não incorporam todas as fontes de incerteza do treinamento.",
        "8. Gravações longas podem refletir conteúdo excedente, repetição, silêncio ou uma particularidade legítima. A análise >60 s é sensibilidade, não uma decisão clínica/forense de invalidade.",
        "9. O VAD identificou toda a duração como ativa em parte dos arquivos longos; esse dado sozinho não prova que todo o conteúdo seja fala CAPE-V válida.",
        "10. As tarefas 004–009 são frases CAPE-V; resultados não devem ser generalizados automaticamente para vogais sustentadas, leitura ou fala espontânea.",
        "",
        "## 13. Próximas análises recomendadas",
        "1. Interpretar a tabela pareada Wavelet−MFCC com ΔBA, ΔAUC, ICs e consistência entre folds.",
        "2. Ouvir/inspecionar MXS008, EMF009 e as gravações longas 004 acima de 30 s; verificar se há fala excedente, repetição, ruído ou anotação inadequada.",
        "3. Se Wavelet parecer superior, executar nested CV para escolha de família/transformada/top-k/C sem reutilizar os mesmos folds para seleção e avaliação.",
        "4. Para comparação acústica mais justa, extrair MFCC temporal nas mesmas janelas VAD de 400 ms e resumir as mesmas quatro estatísticas.",
        "5. Avaliar validação externa por centro, se houver quantidade e classes suficientes; ESSUA isolada não serve para classificação binária porque, nesta coorte, tem apenas participantes normais.",
        "6. Reportar matriz de confusão e erros por classe, além de sensibilidade e especificidade, e verificar influência de sexo, idade e ano de visita.",
        "",
        "## 14. Arquivos gerados",
        f"- `{OUT_SUMMARY.name}` — resumo de métricas OOF e médias/desvios dos folds.",
        f"- `{OUT_FOLDS.name}` — métricas detalhadas por fold.",
        f"- `{OUT_OOF.name}` — predição OOF por participante e configuração.",
        f"- `{OUT_PAIRED.name}` — diferenças pareadas Wavelet−MFCC e IC bootstrap.",
        f"- `{OUT_FEATURES.name}` — frequência de seleção das features Wavelet.",
        f"- `{OUT_ASSIGNMENT.name}` — mapeamento participante-fold.",
        f"- `{OUT_EXCLUDED.name}` — gravações removidas apenas na sensibilidade.",
        f"- `{OUT_REPORT_MD.name}` e `{OUT_REPORT_TXT.name}` — este relatório.",
        f"- `{OUT_CLAUDE.name}` — contexto histórico + resultados desta execução para entregar ao Claude.",
        "",
        "## 15. Nota de reprodutibilidade",
        f"Seed folds={RANDOM_STATE}; seed bootstrap={BOOTSTRAP_SEED}; folds={N_SPLITS}; SVC C={SVC_C}; TOP_K={TOP_K}; threshold de sensibilidade={SENSITIVITY_MAX_DURATION_S} s.",
        "O script não altera as extrações originais nem os caches SQLite. Ele apenas lê os NPZ/CSV e escreve novos resultados de modelagem.",
    ]
    return "\n".join(md) + "\n"


def main():
    started = time.perf_counter()
    say("=" * 78)
    say("MODELAGEM AVFAD — WAVELET TEMPORAL vs MFCC (ComParE como referencia)")
    say("=" * 78)
    X_wave, wave_names, X_mfcc, mfcc_names, X_compare, compare_names, df = load_inputs()
    baseline_meta_path = next((path for path in BASELINE_META_CANDIDATES if path.exists()), None)
    if baseline_meta_path is None:
        raise ModelingError("Metadata ComParE/MFCC desapareceu depois da leitura inicial.")
    progress_dir = prepare_progress_dir(baseline_meta_path)
    say(f"Checkpoint por configuracao: {progress_dir}")
    configs = make_configurations(wave_names, X_wave, X_mfcc, mfcc_names, X_compare, compare_names)
    fold_by_person, fold_by_row, assignments = create_participant_folds(df)
    csv_out(OUT_ASSIGNMENT, assignments)

    duration = df["duration_s"].to_numpy(dtype=np.float64)
    if not np.isfinite(duration).all():
        raise ModelingError("duration_s contem valores ausentes/nao numericos.")
    full_keep = np.ones(len(df), dtype=bool)
    sensitivity_keep = duration <= SENSITIVITY_MAX_DURATION_S
    excluded_cols = [c for c in ["sample_id", "participant_id", "task", "duration_s", "audio_path_in_zip"] if c in df.columns]
    excluded = df.loc[~sensitivity_keep, excluded_cols].copy()
    if df.loc[sensitivity_keep, "participant_id"].nunique() != df["participant_id"].nunique():
        raise ModelingError("A sensibilidade >60 s removeria participante inteiro; revisar antes de continuar.")
    csv_out(OUT_EXCLUDED, excluded)
    say(f"\nCenario principal: {len(df)} gravacoes. Sensibilidade <=60s: {int(sensitivity_keep.sum())}; removidas: {len(excluded)}.")
    say(f"Participantes mantidos nos dois cenarios: {df['participant_id'].nunique()}.")

    all_preds, all_folds, summary_rows, feature_rows = [], [], [], []
    scenarios = [("completo", full_keep), ("sensibilidade_ate_60s", sensitivity_keep)]
    wavelet_configs = ["DWT_db4", "DWT_sym5", "WPT_db4", "WPT_sym5", "db4_DWT_WPT", "sym5_DWT_WPT", "db4_sym5_DWT_WPT"]
    for scenario, keep in scenarios:
        say("\n" + "=" * 78 + f"\nCENARIO: {scenario}\n" + "=" * 78)
        for config_name, (matrix, indices, names) in configs.items():
            cached = load_config_progress(progress_dir, scenario, config_name)
            if cached is not None:
                pred, fold_rows, selected_counts = cached
                say(f"\nCheckpoint recuperado: {scenario} | {config_name} — pulando novo treinamento.")
            else:
                say(f"\nTreinando {config_name}: {len(indices)} features de entrada...")
                pred, fold_rows, selected_counts = evaluate_config(
                    config_name, matrix, indices, names, df, fold_by_row, keep, scenario
                )
                # Salva os resultados imediatamente após os cinco folds; se uma etapa posterior
                # falhar, esta configuracao nao precisara ser treinada novamente.
                save_config_progress(progress_dir, scenario, config_name, pred, fold_rows, selected_counts)
                say(f"Checkpoint salvo: {scenario} | {config_name}")

            all_preds.append(pred)
            all_folds.extend(fold_rows)
            # Seed deterministica derivada do nome da configuracao; evita hash() randomizado.
            seed_text = f"{scenario}::{config_name}"
            stable_seed = BOOTSTRAP_SEED + sum(
                (i + 1) * byte for i, byte in enumerate(seed_text.encode("utf-8"))
            )
            summary_rows.append(
                metric_summary_row(scenario, config_name, len(indices), pred, fold_rows, stable_seed)
            )
            if scenario == "completo":
                for feature, count in selected_counts.items():
                    feature_rows.append({"configuration": config_name, "feature": feature,
                                         "selected_in_folds": int(count), "total_folds": N_SPLITS,
                                         "selection_frequency": count / N_SPLITS})
            # Mantem uma tabela parcial legivel enquanto a modelagem continua.
            csv_out(OUT_PROGRESS_SUMMARY, summary_rows)
            met = metrics_from_scores(pred.y.to_numpy(), pred.decision_score.to_numpy())
            say(f"OOF {config_name}: BA={met['balanced_accuracy']:.4f}; AUC={met['auc']:.4f}; Sens={met['sensitivity']:.4f}; Esp={met['specificity']:.4f}")

    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(all_folds)
    predictions = pd.concat(all_preds, ignore_index=True)
    feature_freq = pd.DataFrame(feature_rows)
    if not feature_freq.empty:
        feature_freq = feature_freq.sort_values(["configuration", "selected_in_folds", "feature"], ascending=[True, False, True])
    paired = paired_comparisons(predictions, folds, wavelet_configs)
    summary = summary.sort_values(["scenario", "balanced_accuracy_oof", "auc_oof"], ascending=[True, False, False])
    folds = folds.sort_values(["scenario", "configuration", "fold"])
    predictions = predictions.sort_values(["scenario", "configuration", "participant_id"])
    csv_out(OUT_SUMMARY, summary)
    csv_out(OUT_FOLDS, folds)
    csv_out(OUT_OOF, predictions)
    csv_out(OUT_PAIRED, paired)
    csv_out(OUT_FEATURES, feature_freq)

    elapsed = time.perf_counter() - started
    report = build_report(summary, folds, paired, feature_freq, assignments, excluded, df, configs, elapsed)
    OUT_REPORT_MD.write_text(report, encoding="utf-8")
    OUT_REPORT_TXT.write_text(report, encoding="utf-8")
    if CONTEXT_FILE.exists():
        context = CONTEXT_FILE.read_text(encoding="utf-8")
    else:
        context = "# Contexto do TCC AVFAD\n\nArquivo de contexto nao encontrado durante a execucao. Use o relatorio completo abaixo.\n"
    OUT_CLAUDE.write_text(context + "\n\n---\n\n# RESULTADOS DA EXECUCAO MAIS RECENTE\n\n" + report, encoding="utf-8")

    say("\n" + "=" * 78)
    say("MODELAGEM CONCLUIDA")
    say("=" * 78)
    say(summary[["scenario", "configuration", "balanced_accuracy_oof", "auc_oof", "sensitivity_oof", "specificity_oof", "f1_oof"]].to_string(index=False))
    say(f"\nRelatorio detalhado: {OUT_REPORT_MD}")
    say(f"Pacote para Claude: {OUT_CLAUDE}")
    say(f"Tempo total: {elapsed/60:.1f} min")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\nModelagem interrompida pelo usuario. Checkpoints por configuracao foram preservados; execute novamente para retomar. Nenhum NPZ/cache de extracao foi alterado.")
        sys.exit(130)
    except Exception as exc:
        say(f"\nERRO NA MODELAGEM: {exc}")
        raise
