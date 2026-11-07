# -*- coding: utf-8 -*-
"""Benchmark exploratorio de classificadores para AVFAD (CAPE-V 004-009).

Compara classificadores diferentes mantendo fixa a representacao de entrada,
pre-processamento e folds por participante. Nao reextrai audios.

Entrada:
  features_wavelet_temporal_avfad_reduzido.npz
  metadata_wavelet_temporal_avfad_reduzido.csv
  features_capev_compare_avfad.npz
  metadata_capev_avfad.csv OU metadata_capev_compare_avfad.csv

Representacoes avaliadas:
  WPT_sym5, sym5_DWT_WPT, WPT_db4, MFCC_baseline, ComParE2016_baseline.

Classificadores (hiperparametros fixos, sem tuning):
  SVC linear (referencia), SVC RBF, Regressao Logistica,
  Random Forest, Extra Trees e HistGradientBoosting.

Protocolo:
  - os mesmos 5 folds estratificados por participante (seed 42);
  - pre-processamento/seleccao de features ajustados somente no treino;
  - ate 128 features por fold via SelectKBest(f_classif);
  - agregacao da margem/score das gravacoes por media para uma predicao por pessoa;
  - class weights/sample weights para lidar com desbalanceamento;
  - bootstrap estratificado por classe para IC e diferencas pareadas vs SVC linear;
  - checkpoint por par representacao/classificador concluido.

IMPORTANTE:
  Este benchmark e exploratorio e usa hiperparametros fixos. A escolha do melhor
  conjunto com base nestes mesmos folds pode gerar otimismo de selecao. Uma
  estimativa final sem esse viés exigiria nested CV ou coorte externa.

Execute em raiz do repositorio com .venv ativado:
  python .\\modelar_classificadores_avfad.py
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import json
import os
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

ROOT = project_root()
WAVELET_NPZ = asset(ROOT, "features_wavelet_temporal_avfad_reduzido.npz")
WAVELET_META = asset(ROOT, "metadata_wavelet_temporal_avfad_reduzido.csv")
BASELINE_NPZ = asset(ROOT, "features_capev_compare_avfad.npz")
BASELINE_META_CANDIDATES = (
    asset(ROOT, "metadata_capev_avfad.csv"),
    asset(ROOT, "metadata_capev_compare_avfad.csv"),
)

N_SPLITS = 5
RANDOM_STATE = 42
TOP_K = 128
N_JOBS = 8
N_TREES = 250
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 20261105
EXPECTED_PARTICIPANTS = 411
EXPECTED_RECORDINGS = 2465
EXPECTED_WAVELET_FEATURES = 4160
EXPECTED_NORMAL = 82
EXPECTED_PATHOLOGICAL = 329

OUT_SUMMARY = asset(ROOT, "resultados_classificadores_AVFAD.csv")
OUT_FOLDS = asset(ROOT, "resultados_classificadores_AVFAD_folds.csv")
OUT_OOF = asset(ROOT, "predicoes_oof_classificadores_AVFAD.csv")
OUT_PAIRED = asset(ROOT, "comparacao_pareada_classificadores_AVFAD.csv")
OUT_FEATURES = asset(ROOT, "frequencia_features_classificadores_AVFAD.csv")
OUT_ASSIGNMENT = asset(ROOT, "folds_participantes_classificadores_AVFAD.csv")
OUT_REPORT_MD = asset(ROOT, "relatorio_classificadores_AVFAD.md")
OUT_REPORT_TXT = asset(ROOT, "relatorio_classificadores_AVFAD.txt")
OUT_CLAUDE = asset(ROOT, "pacote_claude_classificadores_AVFAD.md")
PROGRESS_DIR = asset(ROOT, "checkpoint_classificadores_AVFAD")
PIPELINE_VERSION = "2026-11-05-classifier-benchmark-v1"

REPRESENTATIONS = (
    "WPT_sym5",
    "sym5_DWT_WPT",
    "WPT_db4",
    "MFCC_baseline",
    "ComParE2016_baseline",
)

CLASSIFIER_NAMES = (
    "SVC_linear",
    "SVC_RBF",
    "LogisticRegression",
    "RandomForest",
    "ExtraTrees",
    "HistGradientBoosting",
)

class ModelingError(RuntimeError):
    pass


def say(message: str) -> None:
    print(message, flush=True)


def csv_out(path: Path, data) -> None:
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def stable_seed(text: str) -> int:
    return BOOTSTRAP_SEED + sum((i + 1) * byte for i, byte in enumerate(text.encode("utf-8")))


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


def fast_ba_auc(y: np.ndarray, scores: np.ndarray):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return np.nan, np.nan
    ba = 0.5 * (float(np.mean(pos >= 0.0)) + float(np.mean(neg < 0.0)))
    ranks = rankdata(np.concatenate([pos, neg]), method="average")
    n_pos, n_neg = len(pos), len(neg)
    auc = (float(np.sum(ranks[:n_pos])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return ba, float(auc)


def bootstrap_metric_intervals(y, scores, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    y = np.asarray(y, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = scores[y == 1], scores[y == 0]
    if not len(pos) or not len(neg):
        return {"balanced_accuracy_ci95_low": np.nan, "balanced_accuracy_ci95_high": np.nan,
                "auc_ci95_low": np.nan, "auc_ci95_high": np.nan}
    rng = np.random.default_rng(seed)
    ba_values = np.empty(iterations)
    auc_values = np.empty(iterations)
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
    }


def bootstrap_paired_delta(y, scores_a, scores_b, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    y = np.asarray(y, dtype=np.int8)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    pos_idx, neg_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(pos_idx) or not len(neg_idx):
        raise ModelingError("Bootstrap pareado requer ambas as classes.")
    rng = np.random.default_rng(seed)
    d_ba = np.empty(iterations)
    d_auc = np.empty(iterations)
    for i in range(iterations):
        idx = np.concatenate([
            pos_idx[rng.integers(0, len(pos_idx), size=len(pos_idx))],
            neg_idx[rng.integers(0, len(neg_idx), size=len(neg_idx))],
        ])
        yb = y[idx]
        ba_a, auc_a = fast_ba_auc(yb, scores_a[idx])
        ba_b, auc_b = fast_ba_auc(yb, scores_b[idx])
        d_ba[i] = ba_a - ba_b
        d_auc[i] = auc_a - auc_b
    return {
        "delta_BA_ci95_low": float(np.quantile(d_ba, 0.025)),
        "delta_BA_ci95_high": float(np.quantile(d_ba, 0.975)),
        "delta_AUC_ci95_low": float(np.quantile(d_auc, 0.025)),
        "delta_AUC_ci95_high": float(np.quantile(d_auc, 0.975)),
        "bootstrap_P_delta_BA_gt0": float(np.mean(d_ba > 0)),
        "bootstrap_P_delta_AUC_gt0": float(np.mean(d_auc > 0)),
    }


def load_metadata(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise ModelingError(f"Metadata {description} nao encontrado: {path}")
    df = pd.read_csv(path)
    required = {"sample_id", "participant_id", "task", "y", "duration_s"}
    missing = required - set(df.columns)
    if missing:
        raise ModelingError(f"Metadata {description} sem colunas: {sorted(missing)}")
    df = df.copy()
    for col in ["sample_id", "participant_id"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    df["task"] = df["task"].astype(str).str.strip().str.zfill(3)
    df["y"] = pd.to_numeric(df["y"], errors="raise").astype(int)
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    if df["sample_id"].duplicated().any():
        raise ModelingError(f"Metadata {description} contem sample_id duplicado.")
    return df


def align_by_sample_id(reference: pd.DataFrame, other_meta: pd.DataFrame, matrix: np.ndarray, description: str):
    ref_ids = reference["sample_id"].tolist()
    other_ids = other_meta["sample_id"].tolist()
    if set(ref_ids) != set(other_ids):
        missing = sorted(set(ref_ids) - set(other_ids))[:10]
        extra = sorted(set(other_ids) - set(ref_ids))[:10]
        raise ModelingError(f"Gravacoes Wavelet e {description} diferem. Ausentes em {description}: {missing}; extras: {extra}")
    lookup = {sid: i for i, sid in enumerate(other_ids)}
    return matrix[np.asarray([lookup[sid] for sid in ref_ids], dtype=np.int64)]


def load_inputs():
    for path in (WAVELET_NPZ, WAVELET_META, BASELINE_NPZ):
        if not path.exists():
            raise ModelingError(f"Arquivo necessario nao encontrado: {path}")
    baseline_meta_path = next((p for p in BASELINE_META_CANDIDATES if p.exists()), None)
    if baseline_meta_path is None:
        raise ModelingError("Metadata MFCC/ComParE ausente. Procurei metadata_capev_avfad.csv e metadata_capev_compare_avfad.csv.")
    say(f"Metadata MFCC/ComParE: {baseline_meta_path.name}")

    with np.load(WAVELET_NPZ, allow_pickle=False) as data:
        if not {"wavelet", "feature_names"}.issubset(data.files):
            raise ModelingError("NPZ Wavelet precisa de wavelet e feature_names.")
        X_wave = np.asarray(data["wavelet"], dtype=np.float32)
        wave_names = np.asarray(data["feature_names"]).astype(str)
    wave_meta = load_metadata(WAVELET_META, "Wavelet")
    if X_wave.shape != (len(wave_meta), len(wave_names)) or X_wave.shape[1] != EXPECTED_WAVELET_FEATURES:
        raise ModelingError(f"Dimensoes Wavelet inconsistentes: {X_wave.shape}; metadata={len(wave_meta)}; nomes={len(wave_names)}")

    with np.load(BASELINE_NPZ, allow_pickle=False) as data:
        if not {"acoustic", "acoustic_names"}.issubset(data.files):
            raise ModelingError("NPZ baseline precisa conter acoustic e acoustic_names.")
        acoustic = np.asarray(data["acoustic"], dtype=np.float32)
        acoustic_names = np.asarray(data["acoustic_names"]).astype(str)
        compare = np.asarray(data["compare"], dtype=np.float32) if "compare" in data.files else None
        compare_names = np.asarray(data["compare_names"]).astype(str) if "compare_names" in data.files else None
    baseline_meta = load_metadata(baseline_meta_path, "MFCC/ComParE")
    if acoustic.shape != (len(baseline_meta), len(acoustic_names)):
        raise ModelingError(f"Dimensoes acoustic inconsistentes: {acoustic.shape}; metadata={len(baseline_meta)}; nomes={len(acoustic_names)}")
    mfcc_cols = np.asarray([i for i, n in enumerate(acoustic_names) if n.startswith(("mfcc_", "delta_", "delta2_"))], dtype=np.int64)
    if len(mfcc_cols) < 100:
        raise ModelingError(f"Apenas {len(mfcc_cols)} colunas MFCC encontradas; esperava cerca de 150.")
    acoustic = align_by_sample_id(wave_meta, baseline_meta, acoustic, "MFCC")
    X_mfcc = acoustic[:, mfcc_cols]
    mfcc_names = acoustic_names[mfcc_cols]

    X_compare = None
    if compare is not None and compare_names is not None:
        if compare.shape != (len(baseline_meta), len(compare_names)):
            raise ModelingError("Dimensoes ComParE inconsistentes.")
        X_compare = align_by_sample_id(wave_meta, baseline_meta, compare, "ComParE")

    def sanitize(X):
        X = np.asarray(X, dtype=np.float32).copy()
        X[~np.isfinite(X)] = np.nan
        return X
    X_wave, X_mfcc = sanitize(X_wave), sanitize(X_mfcc)
    if X_compare is not None:
        X_compare = sanitize(X_compare)

    if len(wave_meta) != EXPECTED_RECORDINGS or wave_meta["participant_id"].nunique() != EXPECTED_PARTICIPANTS:
        raise ModelingError(f"Coorte inesperada: {len(wave_meta)} gravações / {wave_meta['participant_id'].nunique()} participantes.")
    people = wave_meta[["participant_id", "y"]].drop_duplicates()
    counts = people["y"].value_counts().to_dict()
    if counts.get(0, 0) != EXPECTED_NORMAL or counts.get(1, 0) != EXPECTED_PATHOLOGICAL:
        raise ModelingError(f"Distribuicao de classes inesperada: {counts}")
    if (wave_meta.groupby("participant_id")["y"].nunique() != 1).any():
        raise ModelingError("Rotulos divergentes nas gravações de um mesmo participante.")
    # A comparação requer que os rótulos e a identidade coincidam por gravação.
    baseline_reordered = baseline_meta.set_index("sample_id").loc[wave_meta["sample_id"]]
    if not np.array_equal(baseline_reordered["y"].to_numpy(dtype=int), wave_meta["y"].to_numpy(dtype=int)):
        raise ModelingError("Rótulos MFCC/ComParE diferem dos rótulos Wavelet após alinhamento.")
    if not np.array_equal(baseline_reordered["participant_id"].to_numpy(), wave_meta["participant_id"].to_numpy()):
        raise ModelingError("participant_id MFCC/ComParE diverge do manifesto Wavelet.")
    return X_wave, wave_names, X_mfcc, mfcc_names, X_compare, compare_names, wave_meta, baseline_meta_path


def make_representations(wave_names, X_wave, X_mfcc, mfcc_names, X_compare, compare_names):
    masks = {
        "DWT_db4": np.array([n.startswith("DWT__db4__") for n in wave_names]),
        "DWT_sym5": np.array([n.startswith("DWT__sym5__") for n in wave_names]),
        "WPT_db4": np.array([n.startswith("WPT__db4__") for n in wave_names]),
        "WPT_sym5": np.array([n.startswith("WPT__sym5__") for n in wave_names]),
    }
    candidates = {
        "WPT_sym5": (X_wave, np.flatnonzero(masks["WPT_sym5"]), wave_names),
        "sym5_DWT_WPT": (X_wave, np.flatnonzero(masks["DWT_sym5"] | masks["WPT_sym5"]), wave_names),
        "WPT_db4": (X_wave, np.flatnonzero(masks["WPT_db4"]), wave_names),
        "MFCC_baseline": (X_mfcc, np.arange(X_mfcc.shape[1]), mfcc_names),
    }
    if X_compare is not None and compare_names is not None:
        candidates["ComParE2016_baseline"] = (X_compare, np.arange(X_compare.shape[1]), compare_names)
    missing = [name for name in REPRESENTATIONS if name not in candidates]
    if missing:
        raise ModelingError(f"Representacoes indisponiveis: {missing}")
    for name in REPRESENTATIONS:
        X, idx, _ = candidates[name]
        if not len(idx):
            raise ModelingError(f"Representacao {name} sem features.")
        say(f"{name}: {len(idx)} features originais")
    return {name: candidates[name] for name in REPRESENTATIONS}


def create_participant_folds(df: pd.DataFrame):
    people = df[["participant_id", "y"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True)
    if people["participant_id"].duplicated().any():
        raise ModelingError("Tabela de participantes não é única.")
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_by_person = {}
    dummy = np.zeros((len(people), 1), dtype=np.uint8)
    for fold, (_, test_idx) in enumerate(splitter.split(dummy, people["y"].to_numpy(), groups=people["participant_id"].to_numpy())):
        for idx in test_idx:
            pid = people.iloc[idx]["participant_id"]
            if pid in fold_by_person:
                raise ModelingError(f"Participante repetido entre folds: {pid}")
            fold_by_person[pid] = fold
    fold_by_row = df["participant_id"].map(fold_by_person).to_numpy(dtype=np.int64)
    if pd.isna(fold_by_row).any():
        raise ModelingError("Há gravação sem fold atribuído.")
    assignments = people.copy()
    assignments["fold"] = assignments["participant_id"].map(fold_by_person) + 1
    assignments["n_recordings"] = assignments["participant_id"].map(df.groupby("participant_id").size())
    for fold in range(N_SPLITS):
        sub = assignments[assignments["fold"] == fold + 1]
        say(f"Fold {fold+1}: {len(sub)} participantes; Normal={(sub.y==0).sum()}; Patológicos={(sub.y==1).sum()}")
    return fold_by_row, assignments


def prepare_fold_data(X, feature_indices, feature_names, df, train_idx, test_idx, representation, fold):
    x_train = X[np.ix_(train_idx, feature_indices)]
    x_test = X[np.ix_(test_idx, feature_indices)]
    y_train = df.iloc[train_idx]["y"].to_numpy(dtype=np.int64)
    nonempty = ~np.isnan(x_train).all(axis=0)
    x_train, x_test = x_train[:, nonempty], x_test[:, nonempty]
    candidate_names = np.asarray(feature_names)[feature_indices][nonempty]
    if x_train.shape[1] == 0:
        raise ModelingError(f"Todas as features ausentes: {representation}, fold {fold}.")
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)
    variance = VarianceThreshold(threshold=0.0)
    x_train = variance.fit_transform(x_train)
    x_test = variance.transform(x_test)
    remaining_names = candidate_names[variance.get_support()]
    if x_train.shape[1] == 0:
        raise ModelingError(f"Nenhuma feature variável: {representation}, fold {fold}.")
    k = min(TOP_K, x_train.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector = SelectKBest(score_func=f_classif, k=k)
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
    selected_names = remaining_names[selector.get_support()]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    x_test = scaler.transform(x_test).astype(np.float32, copy=False)
    return x_train, x_test, y_train, selected_names


def build_classifier(name: str, fold: int):
    seed = RANDOM_STATE * 100 + fold
    if name == "SVC_linear":
        return SVC(kernel="linear", C=1.0, class_weight="balanced", cache_size=512)
    if name == "SVC_RBF":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", cache_size=512)
    if name == "LogisticRegression":
        return LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=3000, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=N_TREES, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=N_JOBS, random_state=seed,
        )
    if name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=N_TREES, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=N_JOBS, random_state=seed,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150, max_leaf_nodes=15,
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )
    raise ModelingError(f"Classificador desconhecido: {name}")


def get_model_scores(model, X):
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
        score_kind = "decision_function; limiar=0"
    elif hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ModelingError("predict_proba não retornou duas classes.")
        scores = probabilities[:, 1] - 0.5
        score_kind = "predict_proba(classe 1)-0.5; limiar=0"
    else:
        raise ModelingError(f"Modelo {type(model).__name__} não fornece score contínuo.")
    if not np.isfinite(scores).all():
        raise ModelingError("Classificador gerou scores não finitos.")
    return scores, score_kind


def progress_paths(representation: str, classifier: str):
    stem = f"{representation}__{classifier}"
    return {
        "pred": PROGRESS_DIR / f"pred_{stem}.csv",
        "folds": PROGRESS_DIR / f"folds_{stem}.csv",
        "selected": PROGRESS_DIR / f"selected_{stem}.csv",
    }


def prepare_progress(baseline_meta_path: Path):
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    paths = [WAVELET_NPZ, WAVELET_META, BASELINE_NPZ, baseline_meta_path]
    inputs = {}
    for p in paths:
        st = p.stat()
        inputs[p.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    signature = {
        "version": PIPELINE_VERSION, "inputs": inputs,
        "representations": REPRESENTATIONS, "classifiers": CLASSIFIER_NAMES,
        "n_splits": N_SPLITS, "random_state": RANDOM_STATE,
        "top_k": TOP_K, "n_trees": N_TREES, "n_jobs": N_JOBS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }
    signature_path = PROGRESS_DIR / "run_signature.json"
    if signature_path.exists():
        try:
            old = json.loads(signature_path.read_text(encoding="utf-8"))
        except Exception:
            old = None
        if old != signature:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_dir = asset(ROOT, f"{PROGRESS_DIR.name}_{stamp}")
            suffix = 1
            while new_dir.exists():
                new_dir = asset(ROOT, f"{PROGRESS_DIR.name}_{stamp}_{suffix}")
                suffix += 1
            new_dir.mkdir(parents=True, exist_ok=False)
            (new_dir / "run_signature.json").write_text(
                json.dumps(signature, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            say(f"Checkpoint antigo preservado; novo diretorio: {new_dir.name}")
            return new_dir
    signature_path.write_text(json.dumps(signature, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return PROGRESS_DIR


def load_checkpoint(progress_dir: Path, representation: str, classifier: str):
    paths = progress_paths(representation, classifier)
    paths = {k: progress_dir / v.name for k, v in paths.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    try:
        pred = pd.read_csv(paths["pred"])
        folds = pd.read_csv(paths["folds"])
        selected = pd.read_csv(paths["selected"])
        if (len(pred) != EXPECTED_PARTICIPANTS
                or pred["participant_id"].nunique() != EXPECTED_PARTICIPANTS
                or pred["representation"].nunique() != 1
                or pred["representation"].iloc[0] != representation
                or pred["classifier"].nunique() != 1
                or pred["classifier"].iloc[0] != classifier
                or len(folds) != N_SPLITS):
            return None
        selected_counts = Counter()
        if not selected.empty:
            selected_counts.update({str(r.feature): int(r.selected_in_folds) for r in selected.itertuples(index=False)})
        return pred, folds.to_dict("records"), selected_counts
    except Exception:
        return None


def save_checkpoint(progress_dir, representation, classifier, pred, fold_rows, selected_counts):
    paths = progress_paths(representation, classifier)
    paths = {k: progress_dir / v.name for k, v in paths.items()}
    atomic_csv(paths["pred"], pred)
    atomic_csv(paths["folds"], pd.DataFrame(fold_rows))
    selected = pd.DataFrame([
        {"feature": k, "selected_in_folds": int(v)} for k, v in selected_counts.items()
    ], columns=["feature", "selected_in_folds"])
    atomic_csv(paths["selected"], selected)


def evaluate_pair(representation, classifier_name, X, indices, names, df, fold_by_row, progress_dir):
    cached = load_checkpoint(progress_dir, representation, classifier_name)
    if cached is not None:
        say(f"  Checkpoint recuperado: {representation} + {classifier_name}")
        return cached
    pred_parts, fold_rows = [], []
    selected_counts = Counter()
    y_all = df["y"].to_numpy(dtype=np.int64)
    for fold in range(1, N_SPLITS + 1):
        train_idx = np.flatnonzero(fold_by_row != fold - 1)
        test_idx = np.flatnonzero(fold_by_row == fold - 1)
        if set(df.iloc[train_idx].participant_id) & set(df.iloc[test_idx].participant_id):
            raise ModelingError(f"Vazamento de participante em {representation}/{classifier_name}/fold {fold}.")
        x_train, x_test, y_train, selected_names = prepare_fold_data(
            X, indices, names, df, train_idx, test_idx, representation, fold
        )
        model = build_classifier(classifier_name, fold)
        if classifier_name == "HistGradientBoosting":
            weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(x_train, y_train, sample_weight=weights)
        else:
            model.fit(x_train, y_train)
        recording_scores, score_kind = get_model_scores(model, x_test)
        test_meta = df.iloc[test_idx][["participant_id", "y"]].copy()
        test_meta["score"] = recording_scores
        people = test_meta.groupby("participant_id", as_index=False).agg(
            y=("y", "first"), score=("score", "mean"), n_recordings=("score", "size")
        )
        people["fold"] = fold
        people["representation"] = representation
        people["classifier"] = classifier_name
        people["label"] = np.where(people.y == 1, "Pathological", "Normal")
        people["prediction"] = (people.score >= 0).astype(np.int8)
        people["correct"] = (people.prediction == people.y).astype(np.int8)
        fold_metrics = metrics_from_scores(people.y.to_numpy(), people.score.to_numpy())
        fold_rows.append({
            "representation": representation, "classifier": classifier_name,
            "fold": fold, "score_kind": score_kind,
            "train_recordings": len(train_idx), "test_recordings": len(test_idx),
            "train_participants": df.iloc[train_idx].participant_id.nunique(),
            "test_participants": people.participant_id.nunique(),
            "input_features": len(indices), "features_after_variance": int(x_train.shape[1]),
            "selected_features": len(selected_names), **fold_metrics,
        })
        pred_parts.append(people)
        selected_counts.update(selected_names.tolist())
        say(f"  {representation} | {classifier_name} | fold {fold}/{N_SPLITS} | BA={fold_metrics['balanced_accuracy']:.3f} | AUC={fold_metrics['auc']:.3f}")
    pred_df = pd.concat(pred_parts, ignore_index=True).sort_values("participant_id").reset_index(drop=True)
    if len(pred_df) != EXPECTED_PARTICIPANTS or pred_df.participant_id.nunique() != EXPECTED_PARTICIPANTS:
        raise ModelingError(f"{representation}/{classifier_name}: esperava 411 predições OOF; obtive {len(pred_df)}.")
    if pred_df.participant_id.duplicated().any():
        raise ModelingError(f"Predições OOF duplicadas em {representation}/{classifier_name}.")
    save_checkpoint(progress_dir, representation, classifier_name, pred_df, fold_rows, selected_counts)
    say(f"  Checkpoint salvo: {representation} + {classifier_name}")
    return pred_df, fold_rows, selected_counts


def summarize_pair(representation, classifier_name, n_features, pred, fold_rows):
    pooled = metrics_from_scores(pred.y.to_numpy(), pred.score.to_numpy())
    ci = bootstrap_metric_intervals(pred.y.to_numpy(), pred.score.to_numpy(), seed=stable_seed(representation + classifier_name))
    fdf = pd.DataFrame(fold_rows)
    result = {
        "representation": representation, "classifier": classifier_name,
        "input_features": int(n_features), "selected_features_mean": float(fdf.selected_features.mean()),
        **{f"{k}_oof": v for k, v in pooled.items()}, **ci,
    }
    for metric in ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "auc", "average_precision", "mcc"]:
        result[f"{metric}_fold_mean"] = float(fdf[metric].mean())
        result[f"{metric}_fold_sd"] = float(fdf[metric].std(ddof=1))
    return result


def make_paired_table(predictions: pd.DataFrame, folds: pd.DataFrame):
    rows = []
    for rep in REPRESENTATIONS:
        ref = predictions[(predictions.representation == rep) & (predictions.classifier == "SVC_linear")].sort_values("participant_id")
        if ref.empty:
            raise ModelingError(f"Faltou SVC_linear de referência para {rep}.")
        for clf in CLASSIFIER_NAMES:
            if clf == "SVC_linear":
                continue
            cur = predictions[(predictions.representation == rep) & (predictions.classifier == clf)].sort_values("participant_id")
            if len(cur) != len(ref) or not np.array_equal(cur.participant_id.to_numpy(), ref.participant_id.to_numpy()):
                raise ModelingError(f"Comparação pareada incompleta: {rep}/{clf}.")
            if not np.array_equal(cur.y.to_numpy(), ref.y.to_numpy()):
                raise ModelingError("Rótulos divergentes na comparação pareada.")
            m_cur = metrics_from_scores(cur.y.to_numpy(), cur.score.to_numpy())
            m_ref = metrics_from_scores(ref.y.to_numpy(), ref.score.to_numpy())
            boot = bootstrap_paired_delta(
                ref.y.to_numpy(), cur.score.to_numpy(), ref.score.to_numpy(),
                seed=stable_seed(rep + "::" + clf),
            )
            row = {"representation": rep, "classifier": clf, "reference_classifier": "SVC_linear", "n_participants": len(ref)}
            for metric in ["balanced_accuracy", "auc", "sensitivity", "specificity", "f1", "mcc"]:
                row[f"{metric}_{clf}"] = m_cur[metric]
                row[f"{metric}_SVC_linear"] = m_ref[metric]
                row[f"delta_{metric}_vs_SVC_linear"] = m_cur[metric] - m_ref[metric]
            row.update(boot)
            fc = folds[(folds.representation == rep) & (folds.classifier == clf)]
            fr = folds[(folds.representation == rep) & (folds.classifier == "SVC_linear")]
            joined = fc.merge(fr, on="fold", suffixes=("_model", "_linear"))
            for metric in ["balanced_accuracy", "auc"]:
                delta = joined[f"{metric}_model"] - joined[f"{metric}_linear"]
                row[f"delta_{metric}_fold_mean"] = float(delta.mean())
                row[f"delta_{metric}_fold_sd"] = float(delta.std(ddof=1))
                row[f"folds_better_{metric}"] = int((delta > 0).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def build_report(summary, folds, paired, selected, assignments, elapsed, baseline_meta_path):
    md = [
        "# Benchmark de classificadores — AVFAD CAPE-V",
        "",
        f"**Gerado em:** {datetime.now().isoformat(timespec='seconds')}  ",
        f"**Python / scikit-learn:** {sys.version.split()[0]} / {sklearn.__version__}  ",
        f"**Duração:** {elapsed/60:.2f} minutos  ",
        f"**Metadata MFCC/ComParE:** `{baseline_meta_path.name}`",
        "",
        "## 1. Objetivo e interpretação",
        "Esta rodada compara famílias de classificadores mantendo fixa a representação de entrada, os cinco folds por participante e o pré-processamento. É um benchmark exploratório com hiperparâmetros predefinidos; não é uma busca exaustiva nem uma estimativa final livre de viés de seleção.",
        "",
        "## 2. Coorte e unidade de avaliação",
        "- Base: AVFAD, centro HSA; 411 participantes (82 normais, 329 patológicos).",
        "- Tarefas: CAPE-V 004–009; 2.465 gravações válidas. CCO tem cinco frases porque CCO004 foi rejeitada por duração de aproximadamente 0,093 s.",
        "- Unidade de divisão e de avaliação: participante. As gravações do mesmo participante ficam sempre no mesmo fold; os scores das frases são agregados pela média para uma única predição por pessoa.",
        "- Folds: StratifiedGroupKFold, cinco folds, shuffle=True, random_state=42. O código reconstrói a mesma regra/fold assignment do benchmark wavelet anterior.",
        "",
        "## 3. Representações selecionadas para esta rodada",
        "Foram escolhidas com base no benchmark inicial de SVC linear: WPT_sym5 (melhor BA/AUC inicial entre as wavelets), sym5_DWT_WPT (segunda wavelet), WPT_db4 (representante db4), MFCC e ComParE 2016 como baselines.",
        "",
        "| Representação | Features de entrada |",
        "|---|---:|",
    ]
    for rep, count in [("WPT_sym5", 1920), ("sym5_DWT_WPT", 2080), ("WPT_db4", 1920), ("MFCC_baseline", 150), ("ComParE2016_baseline", 6373)]:
        md.append(f"| {rep} | {count} |")
    md += [
        "",
        "## 4. Classificadores e parâmetros fixos",
        "| Classificador | Configuração |",
        "|---|---|",
        "| SVC linear | kernel=linear, C=1, class_weight=balanced |",
        "| SVC RBF | kernel=rbf, C=1, gamma=scale, class_weight=balanced |",
        "| Regressão logística | C=1, solver=liblinear, class_weight=balanced, max_iter=3000 |",
        f"| Random Forest | {N_TREES} árvores, max_features=sqrt, min_samples_leaf=2, class_weight=balanced_subsample |",
        f"| Extra Trees | {N_TREES} árvores, max_features=sqrt, min_samples_leaf=2, class_weight=balanced |",
        "| HistGradientBoosting | learning_rate=0,05; max_iter=150; max_leaf_nodes=15; L2=1; early_stopping=False; pesos balanceados no treino |",
        "",
        "SVC linear é a referência de comparação interna. Os demais hiperparâmetros foram fixados previamente; nenhum foi escolhido com base nos folds de teste.",
        "",
        "## 5. Pré-processamento",
        f"Em cada fold, somente no conjunto de treino: imputação pela mediana, remoção de colunas constantes, SelectKBest com ANOVA F (k=min({TOP_K}, número de features disponíveis)) e StandardScaler. O conjunto de teste recebe apenas as transformações ajustadas no treino.",
        "Para modelos com `decision_function`, esse score foi usado diretamente. Para Random Forest, Extra Trees e HistGradientBoosting, foi usado P(classe patológica)-0,5. Os scores das gravações foram promediados por participante; o limiar foi zero. AUC usa o score contínuo.",
        "",
        "## 6. Resultados OOF — todas as gravações",
        "As métricas OOF são calculadas juntando uma única predição fora da amostra para cada participante. BA = balanced accuracy; IC95% por bootstrap estratificado de participantes.",
        "",
        "| Representação | Classificador | BA OOF | IC95% BA | AUC OOF | IC95% AUC | Sens. | Esp. | F1 | MCC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.sort_values(["balanced_accuracy_oof", "auc_oof"], ascending=False).itertuples(index=False):
        md.append(f"| {r.representation} | {r.classifier} | {r.balanced_accuracy_oof:.4f} | [{r.balanced_accuracy_ci95_low:.4f}; {r.balanced_accuracy_ci95_high:.4f}] | {r.auc_oof:.4f} | [{r.auc_ci95_low:.4f}; {r.auc_ci95_high:.4f}] | {r.sensitivity_oof:.4f} | {r.specificity_oof:.4f} | {r.f1_oof:.4f} | {r.mcc_oof:.4f} |")
    md += [
        "",
        "## 7. Média e desvio-padrão entre folds",
        "As médias/desvios abaixo resumem cinco partições; não devem ser confundidas com IC95% das predições OOF agregadas.",
        "",
        "| Representação | Classificador | BA média ± DP | AUC média ± DP |",
        "|---|---|---:|---:|",
    ]
    for r in summary.sort_values(["balanced_accuracy_fold_mean", "auc_fold_mean"], ascending=False).itertuples(index=False):
        md.append(f"| {r.representation} | {r.classifier} | {r.balanced_accuracy_fold_mean:.4f} ± {r.balanced_accuracy_fold_sd:.4f} | {r.auc_fold_mean:.4f} ± {r.auc_fold_sd:.4f} |")
    md += [
        "",
        "## 8. Comparação pareada contra SVC linear",
        "Cada comparação usa as mesmas 411 pessoas e reamostra os mesmos participantes para ambos os classificadores. Delta positivo favorece o classificador comparado. IC bootstrap não equivale a teste de hipótese confirmatório.",
        "",
        "| Representação | Classificador | ΔBA | IC95% ΔBA | ΔAUC | IC95% ΔAUC | Folds BA melhores |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in paired.sort_values("delta_balanced_accuracy_vs_SVC_linear", ascending=False).itertuples(index=False):
        md.append(f"| {r.representation} | {r.classifier} | {r.delta_balanced_accuracy_vs_SVC_linear:+.4f} | [{r.delta_BA_ci95_low:+.4f}; {r.delta_BA_ci95_high:+.4f}] | {r.delta_auc_vs_SVC_linear:+.4f} | [{r.delta_AUC_ci95_low:+.4f}; {r.delta_AUC_ci95_high:+.4f}] | {r.folds_better_balanced_accuracy}/5 |")
    md += [
        "",
        "## 9. Estabilidade de seleção de atributos",
        "A seleção de atributos é independente do classificador, pois ocorre antes do ajuste do estimador. Frequência 5/5 significa que o atributo foi selecionado em todos os folds de treino; frequência baixa indica instabilidade entre partições.",
        "",
    ]
    if selected.empty:
        md.append("Nenhuma frequência de seleção disponível.")
    else:
        for rep in REPRESENTATIONS:
            sub = selected[selected.representation == rep].sort_values(["selected_in_folds", "feature"], ascending=[False, True]).head(10)
            if not sub.empty:
                md.append(f"### {rep} — 10 atributos mais selecionados")
                md.append("| Atributo | Folds selecionados | Frequência |")
                md.append("|---|---:|---:|")
                for r in sub.itertuples(index=False):
                    md.append(f"| `{r.feature}` | {r.selected_in_folds}/5 | {r.selection_frequency:.2f} |")
                md.append("")
    md += [
        "## 10. Resultados por fold",
        "| Representação | Classificador | Fold | N teste | BA | AUC | Sens. | Esp. |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in folds.sort_values(["representation", "classifier", "fold"]).itertuples(index=False):
        md.append(f"| {r.representation} | {r.classifier} | {r.fold} | {r.test_participants} | {r.balanced_accuracy:.4f} | {r.auc:.4f} | {r.sensitivity:.4f} | {r.specificity:.4f} |")
    best = summary.sort_values(["balanced_accuracy_oof", "auc_oof"], ascending=False).iloc[0]
    md += [
        "",
        "## 11. Leitura automática do melhor resultado observado",
        f"A maior BA OOF nesta rodada foi {best['balanced_accuracy_oof']:.4f} para `{best['representation']}` + `{best['classifier']}` (AUC={best['auc_oof']:.4f}, sensibilidade={best['sensitivity_oof']:.4f}, especificidade={best['specificity_oof']:.4f}). Isso identifica o melhor resultado observado neste benchmark exploratório, não uma estimativa confirmatória livre de seleção.",
        "",
        "## 12. Limitações e cuidados de interpretação",
        "1. HSA contém 411 participantes, com 82 normais e 329 patológicos. O desbalanceamento é substancial; por isso BA, sensibilidade e especificidade são essenciais, e acurácia isolada não deve guiar a conclusão.",
        "2. Os seis arquivos por participante são observações repetidas. A separação por participante evita que a mesma pessoa apareça no treino e no teste. A agregação por média produz uma predição por participante.",
        "3. Cada gravação recebe o mesmo peso na média por participante; participantes com cinco frases têm uma observação a menos. CCO004 foi rejeitada por áudio truncado na validação do áudio.",
        "4. As representações Wavelet usam VAD/janelamento temporal; MFCC e ComParE do baseline foram extraídos do WAV completo. A comparação é por participante/fold, mas não isola somente a representação. Extrair MFCC nas mesmas janelas do VAD seria uma comparação mais controlada.",
        "5. Scores de SVC/logística são margens de decisão; scores de árvores são probabilidades centradas em 0,5. Todos são usados com limiar zero, mas suas escalas não são diretamente comparáveis. AUC mede ordenação, e a agregação é feita dentro de cada participante/fold.",
        "6. Hiperparâmetros fixos; não houve GridSearch nem nested CV. A melhor combinação observada foi escolhida entre várias combinações testadas e, portanto, pode apresentar otimismo de seleção.",
        "7. IC bootstrap é condicional às predições OOF já obtidas; não incorpora toda a variabilidade de novos folds, ajuste de hiperparâmetros ou aquisição de uma nova coorte.",
        "8. Random Forest, Extra Trees e HistGradientBoosting podem ter comportamento variável numa amostra relativamente pequena. A estabilidade entre folds e os intervalos devem ser considerados, não apenas o primeiro lugar na tabela.",
        "9. Não é um sistema clínico, nem há validação externa. Resultados são prova de conceito na coorte HSA da AVFAD e nas tarefas CAPE-V 004–009.",
        "10. A escolha das representações Wavelet desta segunda rodada foi orientada pelo primeiro benchmark na mesma base/folds. Esta rodada é continuação exploratória, não confirmação independente da superioridade Wavelet.",
        "",
        "## 13. Arquivos produzidos",
        f"- `{OUT_SUMMARY.name}` — métricas OOF e média/desvio entre folds.",
        f"- `{OUT_FOLDS.name}` — métricas por fold.",
        f"- `{OUT_OOF.name}` — predições OOF por pessoa.",
        f"- `{OUT_PAIRED.name}` — deltas pareados contra SVC linear e IC bootstrap.",
        f"- `{OUT_FEATURES.name}` — frequência de seleção das features por representação.",
        f"- `{OUT_ASSIGNMENT.name}` — atribuição de participantes aos folds.",
        f"- `{OUT_REPORT_MD.name}` e `{OUT_REPORT_TXT.name}` — este relatório.",
        f"- `{OUT_CLAUDE.name}` — pacote de contexto/resultados desta rodada.",
        f"- `{PROGRESS_DIR.name}` — checkpoints retomáveis.",
        "",
        "## 14. Reprodutibilidade",
        f"Folds: StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE}); TOP_K={TOP_K}; árvores={N_TREES}; n_jobs={N_JOBS}; bootstrap={BOOTSTRAP_ITERATIONS}; scikit-learn={sklearn.__version__}.",
    ]
    return "\n".join(md) + "\n"


def main():
    started = time.perf_counter()
    say("=" * 78)
    say("BENCHMARK AVFAD — COMPARACAO DE CLASSIFICADORES")
    say("=" * 78)
    X_wave, wave_names, X_mfcc, mfcc_names, X_compare, compare_names, df, baseline_meta_path = load_inputs()
    say(f"Participantes={df.participant_id.nunique()} | gravações={len(df)} | classes={df.drop_duplicates('participant_id').y.value_counts().to_dict()}")
    representations = make_representations(wave_names, X_wave, X_mfcc, mfcc_names, X_compare, compare_names)
    fold_by_row, assignments = create_participant_folds(df)
    csv_out(OUT_ASSIGNMENT, assignments)

    # O checkpoint tem assinatura própria; não mistura resultados de outros scripts.
    progress_dir = prepare_progress(baseline_meta_path)
    say(f"Checkpoint: {progress_dir}")
    all_pred, all_fold, all_selected = [], [], []
    total_pairs = len(representations) * len(CLASSIFIER_NAMES)
    completed_pairs = 0
    for rep, (matrix, indices, names) in representations.items():
        say("\n" + "-" * 78 + f"\nREPRESENTACAO: {rep} ({len(indices)} features)\n" + "-" * 78)
        for classifier_name in CLASSIFIER_NAMES:
            say(f"\n[{completed_pairs+1}/{total_pairs}] {rep} + {classifier_name}")
            pred, fold_rows, selected_counts = evaluate_pair(
                rep, classifier_name, matrix, indices, names, df, fold_by_row, progress_dir
            )
            all_pred.append(pred)
            all_fold.extend(fold_rows)
            if classifier_name == "SVC_linear":
                for feature, count in selected_counts.items():
                    all_selected.append({
                        "representation": rep, "feature": feature,
                        "selected_in_folds": int(count), "total_folds": N_SPLITS,
                        "selection_frequency": count / N_SPLITS,
                    })
            completed_pairs += 1
            # Grava progresso parcial após cada par concluído.
            partial = pd.concat(all_pred, ignore_index=True)
            partial_folds = pd.DataFrame(all_fold)
            atomic_csv(OUT_OOF, partial)
            atomic_csv(OUT_FOLDS, partial_folds)
            say(f"Progresso: {completed_pairs}/{total_pairs} pares de representação/classificador concluídos.")

    predictions = pd.concat(all_pred, ignore_index=True)
    folds = pd.DataFrame(all_fold)
    selected = pd.DataFrame(all_selected)
    summary_rows = []
    for (rep, clf), group in predictions.groupby(["representation", "classifier"], sort=False):
        frows = folds[(folds.representation == rep) & (folds.classifier == clf)].to_dict("records")
        n_features = len(representations[rep][1])
        summary_rows.append(summarize_pair(rep, clf, n_features, group, frows))
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["balanced_accuracy_oof", "auc_oof"], ascending=False).reset_index(drop=True)
    paired = make_paired_table(predictions, folds)
    csv_out(OUT_SUMMARY, summary)
    csv_out(OUT_FOLDS, folds.sort_values(["representation", "classifier", "fold"]))
    csv_out(OUT_OOF, predictions.sort_values(["representation", "classifier", "participant_id"]))
    csv_out(OUT_PAIRED, paired)
    csv_out(OUT_FEATURES, selected)

    elapsed = time.perf_counter() - started
    report = build_report(summary, folds, paired, selected, assignments, elapsed, baseline_meta_path)
    OUT_REPORT_MD.write_text(report, encoding="utf-8")
    OUT_REPORT_TXT.write_text(report, encoding="utf-8")
    OUT_CLAUDE.write_text(
        "# CONTEXTO DA RODADA DE CLASSIFICADORES — AVFAD\n\n"
        "Esta rodada parte da extração Wavelet temporal já concluída (2.465 gravações, 411 participantes HSA, 4.160 features por gravação) e do baseline MFCC/ComParE extraído em WAV completo.\n\n"
        "A rodada anterior com SVC linear encontrou, no cenário completo: WPT_sym5 BA=0,639262/AUC=0,667025; sym5_DWT_WPT BA=0,620932/AUC=0,659241; WPT_db4 BA=0,570632/AUC=0,622396; ComParE BA=0,564590/AUC=0,631663; MFCC BA=0,515791/AUC=0,570242. Esses números pertencem ao benchmark anterior e devem ser distinguidos dos valores abaixo.\n\n---\n\n" + report,
        encoding="utf-8",
    )

    say("\n" + "=" * 78 + "\nBENCHMARK DE CLASSIFICADORES CONCLUIDO\n" + "=" * 78)
    cols = ["representation", "classifier", "balanced_accuracy_oof", "auc_oof", "sensitivity_oof", "specificity_oof", "f1_oof"]
    say(summary[cols].to_string(index=False))
    say(f"\nRelatorio: {OUT_REPORT_MD}")
    say(f"Pacote Claude: {OUT_CLAUDE}")
    say(f"Tempo: {elapsed/60:.2f} min")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\nInterrompido. Checkpoints dos pares já concluídos foram preservados; execute novamente para retomar.")
        sys.exit(130)
    except Exception as exc:
        say(f"\nERRO NO BENCHMARK: {exc}")
        raise
