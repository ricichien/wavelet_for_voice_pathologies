# -*- coding: utf-8 -*-
"""AVFAD /a/: comparar WPT com os 19 parâmetros acústicos Praat já fornecidos.

Não instala nem chama o aplicativo Praat. Lê as 19 medidas da planilha AVFAD,
associa por File ID e avalia, na coorte HSA pareada por sexo+idade+ano (136),
três entradas: WPT, Praat e WPT+Praat.

Usa exatamente a construção dos pares sintéticos, sementes e estrutura de CV
aninhada do script avaliar_pareamento_avfad.py, para permitir comparação.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import io
import json
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = project_root()
FEATURES_CSV = asset(ROOT, "features_avfad_hsa_002.csv")
REPORT_TXT = asset(ROOT, "resumo_praat_avfad.txt")
FOLDS_CSV = asset(ROOT, "folds_praat_avfad.csv")
PREDICTIONS_CSV = asset(ROOT, "predicoes_praat_avfad.csv")

SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 4
N_JOBS = 4

# Cabeçalhos exatamente como aparecem na aba AVFAD.
PRAAT_SOURCE_COLUMNS = [
    "f0_median (Hz)", "f0_mean (Hz)", "f0_std (Hz)",
    "f0_min (Hz)", "f0_max (Hz)",
    "jitter_local (%)", "jitter_local_abs (s)", "jitter_rap (%)",
    "jitter_ppq5 (%)", "jitter_ddp (%)",
    "shimmer_local (%)", "shimmer_local_dB (dB)",
    "shimmer_apq3 (%)", "shimmer_apq5 (%)", "shimmer_apq11 (%)",
    "shimmer_dda (%)", "autocorrelation_mean", "nhr_mean",
    "hnr_mean (dB)",
]
PRAAT_COLUMNS = [f"praat_{i:02d}" for i in range(len(PRAAT_SOURCE_COLUMNS))]


class SelectColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if not self.columns:
            raise ValueError("Grupo sem características.")
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Colunas ausentes: {missing[:10]}")
        return self

    def transform(self, X):
        return X.loc[:, list(self.columns)].to_numpy(dtype=np.float64)


def find_workbook_bytes():
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
        raise RuntimeError(
            "Esperava uma AVFAD_01_00_00.xlsx nos ZIPs; "
            f"encontrei {len(matches)}."
        )
    zip_path, internal_name = matches[0]
    with zipfile.ZipFile(zip_path, "r") as archive:
        return io.BytesIO(archive.read(internal_name))


def load_praat_table():
    wb = load_workbook(find_workbook_bytes(), read_only=True, data_only=True)
    if "AVFAD" not in wb.sheetnames:
        wb.close()
        raise RuntimeError("Aba AVFAD não encontrada na planilha.")
    ws = wb["AVFAD"]
    iterator = ws.iter_rows(values_only=True)
    headers = list(next(iterator))
    required = ["File ID", *PRAAT_SOURCE_COLUMNS]
    missing = [name for name in required if name not in headers]
    if missing:
        wb.close()
        raise RuntimeError(f"Colunas Praat ausentes: {missing}")
    positions = {name: headers.index(name) for name in required}

    rows = {}
    duplicates = []
    for row in iterator:
        raw_id = row[positions["File ID"]]
        if raw_id is None:
            continue
        pid = str(raw_id).strip().upper()
        if pid in rows:
            duplicates.append(pid)
            continue
        values = {}
        for source, target in zip(PRAAT_SOURCE_COLUMNS, PRAAT_COLUMNS):
            value = row[positions[source]]
            try:
                value = float(value) if value is not None else np.nan
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value):
                value = np.nan
            values[target] = value
        rows[pid] = values
    wb.close()
    if duplicates:
        raise RuntimeError(f"IDs duplicados na aba AVFAD: {duplicates[:10]}")
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("participant_id").reset_index()


def age_band(age):
    return 18 + ((int(age) - 18) // 5) * 5


def build_pair_groups(data):
    """Replica a ordem de pareamento usada na reavaliação anterior."""
    data = data.reset_index(drop=True).copy()
    data["age_band"] = data["age"].map(age_band)
    keys = ["sex", "age_band", "visit_year"]
    if data["visit_year"].isna().any():
        raise ValueError("Há ano de visita ausente na coorte de 136.")
    data["visit_year"] = data["visit_year"].astype(int)

    groups = np.empty(len(data), dtype=object)
    groups[:] = None
    pair_counter = 0
    for _, part in data.groupby(keys, sort=True, dropna=False):
        normals = part.loc[part["y"] == 0].sort_values("participant_id").index.tolist()
        patients = part.loc[part["y"] == 1].sort_values("participant_id").index.tolist()
        if len(normals) != len(patients):
            raise ValueError("A coorte não está balanceada dentro de cada estrato.")
        for normal_idx, patient_idx in zip(normals, patients):
            pair_counter += 1
            group_id = f"PAIR_{pair_counter:03d}"
            groups[normal_idx] = group_id
            groups[patient_idx] = group_id

    check = pd.DataFrame({"pair": groups, "y": data["y"]})
    for pair_id, part in check.groupby("pair"):
        if len(part) != 2 or set(part["y"]) != {0, 1}:
            raise ValueError(f"Par sintético inválido: {pair_id}")
    return data, groups


def make_pipeline():
    return Pipeline([
        ("colunas", SelectColumns()),
        ("imputacao", SimpleImputer(strategy="median")),
        ("selecao", SelectKBest(score_func=f_classif)),
        ("escala", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=4000)),
    ])


def param_grid(feature_cols):
    k_values = sorted({min(k, len(feature_cols)) for k in (10, 20)})
    common = {
        "colunas__columns": [tuple(feature_cols)],
        "selecao__k": k_values,
    }
    grids = []

    p = dict(common)
    p.update({
        "modelo": [LogisticRegression(
            class_weight="balanced", solver="liblinear",
            max_iter=4000, random_state=SEED
        )],
        "modelo__C": [0.1, 1.0, 10.0],
    })
    grids.append(p)

    p = dict(common)
    p.update({
        "modelo": [SVC(
            kernel="rbf", class_weight="balanced",
            probability=False, random_state=SEED
        )],
        "modelo__C": [0.1, 1.0, 10.0],
        "modelo__gamma": ["scale"],
    })
    grids.append(p)

    p = dict(common)
    p.update({
        "modelo": [RandomForestClassifier(
            n_estimators=150, class_weight="balanced_subsample",
            random_state=SEED, n_jobs=1
        )],
        "modelo__max_depth": [None, 6],
        "modelo__min_samples_leaf": [2, 5],
        "modelo__max_features": ["sqrt"],
    })
    grids.append(p)
    return grids


def score_model(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return model.predict(X).astype(float)


def get_metrics(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_pathological": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity_normal": tn / (tn + fp) if (tn + fp) else 0.0,
        "precision_pathological": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_pathological": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def append_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(
            path, mode="a", index=False, header=not path.exists(),
            encoding="utf-8-sig" if not path.exists() else "utf-8"
        )


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    for path in (REPORT_TXT, FOLDS_CSV, PREDICTIONS_CSV):
        if path.exists():
            path.unlink()

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Não encontrei {FEATURES_CSV}")

    print("Carregando características WPT e planilha Praat...")
    df = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
    praat = load_praat_table()
    df["participant_id"] = df["participant_id"].astype(str).str.strip().str.upper()
    df = df.merge(praat, on="participant_id", how="left", validate="one_to_one")

    needed = {"label", "age", "sex", "visit_year", "matched_age_sex_year"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV de características: {sorted(missing)}")

    cohort = df[df["matched_age_sex_year"].astype(int) == 1].copy()
    cohort["label"] = cohort["label"].astype(str).str.strip()
    cohort = cohort[cohort["label"].isin(["Normal", "Pathological"])].copy()
    cohort["y"] = (cohort["label"] == "Pathological").astype(int)
    cohort["age"] = pd.to_numeric(cohort["age"], errors="coerce")
    cohort["visit_year"] = pd.to_numeric(cohort["visit_year"], errors="coerce")
    cohort["sex"] = cohort["sex"].astype(str).str.strip().str.upper()

    wpt_cols = [c for c in cohort.columns if c.startswith("wpt_")]
    if len(wpt_cols) != 128:
        raise ValueError(f"Esperava 128 características WPT; encontrei {len(wpt_cols)}")

    missing_praat = int(cohort[PRAAT_COLUMNS].isna().all(axis=1).sum())
    if missing_praat == len(cohort):
        raise ValueError("Nenhum participante possui os parâmetros Praat associados.")

    groups = {
        "WPT": wpt_cols,
        "PRAAT_19": PRAAT_COLUMNS,
        "WPT_PRAAT": wpt_cols + PRAAT_COLUMNS,
    }
    cohort, pair_groups = build_pair_groups(cohort)
    y = cohort["y"].to_numpy(dtype=int)
    ids = cohort["participant_id"].to_numpy()
    if len(cohort) != 136 or Counter(y) != Counter({0: 68, 1: 68}):
        raise ValueError(
            f"Esperava 136 participantes (68+68), encontrei {len(cohort)} e {Counter(y)}"
        )

    with REPORT_TXT.open("w", encoding="utf-8") as report:
        report.write("AVFAD — PRAAT E FUSÃO COM WPT | TAREFA 002 (/a/)\n")
        report.write("=" * 68 + "\n")
        report.write("Coorte: HSA pareada por sexo + faixa etária de 5 anos + ano.\n")
        report.write("Nested CV: externo StratifiedGroupKFold 5-fold x 2; interno 4-fold.\n")
        report.write("Pares sintéticos são mantidos juntos nos folds.\n")
        report.write("Praat: 19 parâmetros acústicos pré-calculados e fornecidos na planilha AVFAD.\n")
        report.write("Sem idade, sexo, ano, local, duração ou número de segmentos como preditores.\n")
        report.write("Imputação, seleção de características e escala dentro do pipeline.\n")
        report.write(f"Participantes: {len(cohort)} | pares: {len(set(pair_groups))} | Normal=68 | Patológico=68\n")
        report.write(f"Parâmetros Praat totalmente ausentes na linha: {missing_praat}\n")
        report.write("Grupos: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()) + "\n")

        for repeat in range(1, OUTER_REPEATS + 1):
            outer = StratifiedGroupKFold(
                n_splits=OUTER_SPLITS, shuffle=True,
                random_state=SEED + repeat - 1
            )
            for fold, (train_idx, test_idx) in enumerate(
                outer.split(cohort, y, groups=pair_groups), start=1
            ):
                train_groups = pair_groups[train_idx]
                test_groups = pair_groups[test_idx]
                if set(train_groups) & set(test_groups):
                    raise RuntimeError("Par dividido entre treino e teste.")

                X_train = cohort.iloc[train_idx]
                X_test = cohort.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                split_number = (repeat - 1) * OUTER_SPLITS + fold
                inner = StratifiedGroupKFold(
                    n_splits=INNER_SPLITS, shuffle=True,
                    random_state=SEED + 100 + split_number
                )

                fold_rows = []
                prediction_rows = []
                for group_name, feature_cols in groups.items():
                    print(
                        f"Repetição {repeat}/{OUTER_REPEATS} | "
                        f"fold {fold}/{OUTER_SPLITS} | {group_name}",
                        flush=True
                    )
                    search = GridSearchCV(
                        estimator=make_pipeline(),
                        param_grid=param_grid(feature_cols),
                        scoring="balanced_accuracy", cv=inner,
                        refit=True, n_jobs=N_JOBS, error_score="raise",
                        return_train_score=False,
                    )
                    search.fit(X_train, y_train, groups=train_groups)
                    best = search.best_estimator_
                    pred = best.predict(X_test).astype(int)
                    scores = score_model(best, X_test)
                    model_name = type(best.named_steps["modelo"]).__name__

                    fold_rows.append({
                        "cohort": "Pareada_sexo_idade_ano_136",
                        "repeat": repeat, "fold": fold,
                        "feature_set": group_name,
                        "selected_model": model_name,
                        "inner_balanced_accuracy": float(search.best_score_),
                        "selected_k": search.best_params_.get("selecao__k"),
                        "best_params": json.dumps({
                            k: v for k, v in search.best_params_.items()
                            if k not in {"colunas__columns", "modelo"}
                        }, ensure_ascii=False, default=str),
                        "n_train": len(train_idx), "n_test": len(test_idx),
                        **get_metrics(y_test, pred, scores),
                    })
                    for pid, yt, yp, sc in zip(ids[test_idx], y_test, pred, scores):
                        prediction_rows.append({
                            "cohort": "Pareada_sexo_idade_ano_136",
                            "repeat": repeat, "fold": fold,
                            "participant_id": pid, "feature_set": group_name,
                            "y_true": int(yt), "y_pred": int(yp),
                            "score_pathological": float(sc),
                        })

                append_csv(FOLDS_CSV, fold_rows)
                append_csv(PREDICTIONS_CSV, prediction_rows)
                print(f"Concluído: repetição {repeat}, fold {fold}.", flush=True)

        fold_df = pd.read_csv(FOLDS_CSV)
        metrics = [
            "accuracy", "balanced_accuracy", "sensitivity_pathological",
            "specificity_normal", "f1_pathological", "roc_auc",
        ]
        report.write("\nRESULTADOS: média +/- desvio-padrão entre os 10 folds externos\n")
        for feature_set, part in fold_df.groupby("feature_set", sort=False):
            report.write(f"\n{feature_set}\n")
            for metric in metrics:
                vals = pd.to_numeric(part[metric], errors="coerce").dropna()
                report.write(f"  {metric}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}\n")
            report.write(f"  Modelos selecionados: {part['selected_model'].value_counts().to_dict()}\n")

        report.write("\nArquivos gerados:\n")
        report.write(f"  {FOLDS_CSV.name}\n  {PREDICTIONS_CSV.name}\n  {REPORT_TXT.name}\n")

    print("\nConcluído.")
    print(f"Relatório: {REPORT_TXT}")
    print(f"Folds: {FOLDS_CSV}")
    print(f"Predições: {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
