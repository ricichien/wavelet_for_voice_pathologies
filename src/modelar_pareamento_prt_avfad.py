# -*- coding: utf-8 -*-
"""AVFAD /a/: avaliação usando o intervalo anotado no PRT.

Entrada: features_avfad_prt_hsa_002.csv (gerado por extrair_features_prt_avfad.py).
Saídas:
  resumo_pareamento_avfad.txt
  folds_pareamento_avfad.csv
  predicoes_pareamento_avfad.csv

Compara as coortes pareadas por sexo+idade (162) e por sexo+idade+ano (136).
Cada par sintético contém um participante Normal e um Pathological da mesma
faixa de pareamento. Ambos ficam sempre no mesmo fold externo e interno.
O pareamento é por estrato, não um pareamento clínico individual original.

Modelos acústicos: MFCC, DWT e WPT, com seleção/normalização dentro dos folds.
Baseline demográfico separado: idade, sexo e ano; nunca é misturado ao áudio.
"""

from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = project_root()
INPUT_CSV = asset(ROOT, "features_avfad_prt_hsa_002.csv")
REPORT_TXT = asset(ROOT, "resumo_pareamento_prt_avfad.txt")
FOLDS_CSV = asset(ROOT, "folds_pareamento_prt_avfad.csv")
PREDICTIONS_CSV = asset(ROOT, "predicoes_pareamento_prt_avfad.csv")

SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 4
N_JOBS = 4


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


def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required = {
        "participant_id", "label", "age", "sex", "visit_year",
        "matched_age_sex", "matched_age_sex_year"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")

    if df["participant_id"].duplicated().any():
        raise ValueError("Há participant_id duplicado no arquivo de características.")

    df = df.copy()
    df["participant_id"] = df["participant_id"].astype(str).str.strip().str.upper()
    df["label"] = df["label"].astype(str).str.strip()
    if not set(df["label"].dropna().unique()).issubset({"Normal", "Pathological"}):
        raise ValueError("Há rótulos diferentes de Normal/Pathological.")

    df["y"] = (df["label"] == "Pathological").astype(int)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["visit_year"] = pd.to_numeric(df["visit_year"], errors="coerce")
    df["sex"] = df["sex"].astype(str).str.strip().str.upper()
    df["matched_age_sex"] = pd.to_numeric(df["matched_age_sex"], errors="coerce").fillna(0).astype(int)
    df["matched_age_sex_year"] = pd.to_numeric(df["matched_age_sex_year"], errors="coerce").fillna(0).astype(int)

    return df


def age_band(age):
    return 18 + ((int(age) - 18) // 5) * 5


def build_pair_groups(data: pd.DataFrame, include_year: bool):
    """Constrói grupos de dois dentro dos estratos usados no pareamento."""
    data = data.reset_index(drop=True).copy()
    data["age_band"] = data["age"].map(age_band)

    keys = ["sex", "age_band"]
    if include_year:
        if data["visit_year"].isna().any():
            raise ValueError("A coorte pareada por ano contém ano ausente.")
        data["visit_year"] = data["visit_year"].astype(int)
        keys.append("visit_year")

    pair_groups = np.empty(len(data), dtype=object)
    pair_groups[:] = None
    pair_counter = 0
    pair_count = 0

    for stratum, part in data.groupby(keys, sort=True, dropna=False):
        normals = part.loc[part["y"] == 0].sort_values("participant_id").index.tolist()
        patients = part.loc[part["y"] == 1].sort_values("participant_id").index.tolist()

        if len(normals) != len(patients):
            raise ValueError(
                f"Estrato {stratum} não está pareado: "
                f"Normal={len(normals)}, Patológico={len(patients)}."
            )

        for normal_idx, patient_idx in zip(normals, patients):
            pair_counter += 1
            pair_id = f"PAIR_{pair_counter:03d}"
            pair_groups[normal_idx] = pair_id
            pair_groups[patient_idx] = pair_id
            pair_count += 1

    if any(group is None for group in pair_groups):
        raise ValueError("Nem todos os participantes receberam um grupo de pareamento.")

    # Valida que cada grupo tem exatamente duas pessoas, uma por classe.
    check = pd.DataFrame({"pair": pair_groups, "y": data["y"]})
    for pair_id, part in check.groupby("pair"):
        if len(part) != 2 or set(part["y"]) != {0, 1}:
            raise ValueError(f"Grupo de pareamento inválido: {pair_id}")

    return data, pair_groups, pair_count


def feature_groups(df):
    cols = list(df.columns)
    mfcc_all = [
        c for c in cols
        if c.startswith("mfcc_") or c.startswith("delta_") or c.startswith("delta2_")
    ]
    mfcc = [c for c in mfcc_all if c != "mfcc_00_mean"]
    dwt = [c for c in cols if c.startswith("dwt_")]
    wpt = [c for c in cols if c.startswith("wpt_")]

    groups = {"MFCC_sem_nivel": mfcc, "DWT": dwt, "WPT": wpt}
    for name, values in groups.items():
        if not values:
            raise ValueError(f"Não encontrei características para {name}.")
    return groups


def metric_values(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_pathological": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity_normal": tn / (tn + fp) if tn + fp else 0.0,
        "precision_pathological": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_pathological": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def get_score(model, X):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float).reshape(-1)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    return model.predict(X).astype(float)


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


def demographic_pipeline():
    preprocessor = ColumnTransformer([
        ("idade", StandardScaler(), ["age"]),
        ("categoricas", OneHotEncoder(handle_unknown="ignore"), ["sex", "visit_year"]),
    ])
    return Pipeline([
        ("pre", preprocessor),
        ("modelo", LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=3000,
            random_state=SEED
        )),
    ])


def append_rows(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(
            path, mode="a", index=False, header=not path.exists(),
            encoding="utf-8-sig" if not path.exists() else "utf-8"
        )


def evaluate_cohort(name, data, pair_groups, groups, report):
    y = data["y"].to_numpy(dtype=int)
    pair_groups = np.asarray(pair_groups)
    X = data.copy()
    ids = data["participant_id"].to_numpy()

    class_counts = Counter(y)
    n_pairs = len(set(pair_groups))
    if class_counts[0] != class_counts[1] or class_counts[0] != n_pairs:
        raise ValueError(f"Coorte não está balanceada por pares: {class_counts}, pares={n_pairs}")

    report.write(
        f"\nCOORTE {name}: N={len(data)}, pares={n_pairs}, "
        f"Normal={class_counts[0]}, Patológico={class_counts[1]}\n"
    )

    fold_rows = []
    prediction_rows = []
    split_number = 0

    for repeat in range(1, OUTER_REPEATS + 1):
        outer = StratifiedGroupKFold(
            n_splits=OUTER_SPLITS,
            shuffle=True,
            random_state=SEED + repeat - 1,
        )

        for fold, (train_idx, test_idx) in enumerate(
            outer.split(X, y, groups=pair_groups), start=1
        ):
            split_number += 1
            prediction_start = len(prediction_rows)
            train_groups = pair_groups[train_idx]
            test_groups = pair_groups[test_idx]
            if set(train_groups) & set(test_groups):
                raise RuntimeError("Um par foi dividido entre treino e teste.")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            inner = StratifiedGroupKFold(
                n_splits=INNER_SPLITS,
                shuffle=True,
                random_state=SEED + 100 + split_number,
            )

            # Baseline majoritária.
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(np.zeros((len(y_train), 1)), y_train)
            dummy_pred = dummy.predict(np.zeros((len(y_test), 1))).astype(int)
            dummy_score = dummy_pred.astype(float)
            fold_rows.append({
                "cohort": name, "repeat": repeat, "fold": fold,
                "feature_set": "BASELINE_majoritaria", "selected_model": "Dummy_majority",
                "n_train": len(train_idx), "n_test": len(test_idx),
                **metric_values(y_test, dummy_pred, dummy_score),
            })

            # Baseline separado de confundidores demográficos (não é modelo acústico).
            demo_train = X_train[["age", "sex", "visit_year"]].copy()
            demo_test = X_test[["age", "sex", "visit_year"]].copy()
            demo_train["sex"] = demo_train["sex"].astype(str)
            demo_test["sex"] = demo_test["sex"].astype(str)
            demo_train["visit_year"] = demo_train["visit_year"].astype(int).astype(str)
            demo_test["visit_year"] = demo_test["visit_year"].astype(int).astype(str)
            demo = demographic_pipeline()
            demo.fit(demo_train, y_train)
            demo_pred = demo.predict(demo_test).astype(int)
            demo_score = get_score(demo, demo_test)
            fold_rows.append({
                "cohort": name, "repeat": repeat, "fold": fold,
                "feature_set": "BASELINE_demografico", "selected_model": "LogisticRegression_demografia",
                "n_train": len(train_idx), "n_test": len(test_idx),
                **metric_values(y_test, demo_pred, demo_score),
            })
            for pid, yt, yp, sc in zip(ids[test_idx], y_test, demo_pred, demo_score):
                prediction_rows.append({
                    "cohort": name, "repeat": repeat, "fold": fold,
                    "participant_id": pid, "feature_set": "BASELINE_demografico",
                    "y_true": int(yt), "y_pred": int(yp), "score_pathological": float(sc),
                })

            for group_name, feature_cols in groups.items():
                print(
                    f"{name} | repetição {repeat}/{OUTER_REPEATS} | "
                    f"fold {fold}/{OUTER_SPLITS} | {group_name}",
                    flush=True,
                )
                search = GridSearchCV(
                    estimator=make_pipeline(),
                    param_grid=param_grid(feature_cols),
                    scoring="balanced_accuracy",
                    cv=inner,
                    refit=True,
                    n_jobs=N_JOBS,
                    error_score="raise",
                    return_train_score=False,
                )
                search.fit(X_train, y_train, groups=train_groups)
                best = search.best_estimator_
                pred = best.predict(X_test).astype(int)
                score = get_score(best, X_test)
                model_name = type(best.named_steps["modelo"]).__name__

                fold_rows.append({
                    "cohort": name, "repeat": repeat, "fold": fold,
                    "feature_set": group_name, "selected_model": model_name,
                    "inner_balanced_accuracy": float(search.best_score_),
                    "n_train": len(train_idx), "n_test": len(test_idx),
                    "selected_k": search.best_params_.get("selecao__k"),
                    "best_params": json.dumps({
                        k: v for k, v in search.best_params_.items()
                        if k not in {"colunas__columns", "modelo"}
                    }, ensure_ascii=False, default=str),
                    **metric_values(y_test, pred, score),
                })
                for pid, yt, yp, sc in zip(ids[test_idx], y_test, pred, score):
                    prediction_rows.append({
                        "cohort": name, "repeat": repeat, "fold": fold,
                        "participant_id": pid, "feature_set": group_name,
                        "y_true": int(yt), "y_pred": int(yp),
                        "score_pathological": float(sc),
                    })

            append_rows(FOLDS_CSV, fold_rows[-(2 + len(groups)):])
            append_rows(PREDICTIONS_CSV, prediction_rows[prediction_start:])
            print(f"  Concluído {name}: fold {split_number}/{OUTER_SPLITS * OUTER_REPEATS}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    report.write("\nResumo: média +/- desvio-padrão entre os 10 folds externos.\n")
    metrics = ["accuracy", "balanced_accuracy", "sensitivity_pathological", "specificity_normal", "f1_pathological", "roc_auc"]
    for feature_set, part in fold_df.groupby("feature_set", sort=False):
        report.write(f"\n{feature_set}\n")
        for metric in metrics:
            vals = pd.to_numeric(part[metric], errors="coerce").dropna()
            if len(vals):
                report.write(f"  {metric}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}\n")
        selected = part["selected_model"].value_counts().to_dict()
        report.write(f"  Modelos: {selected}\n")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    for path in (REPORT_TXT, FOLDS_CSV, PREDICTIONS_CSV):
        if path.exists():
            path.unlink()

    df = load_data()
    all_groups = feature_groups(df)

    # Duas coortes: uma controlando sexo/idade; outra controlando também ano.
    cohorts = [
        ("Pareada_sexo_idade_162", df[df["matched_age_sex"] == 1].copy(), False),
        ("Pareada_sexo_idade_ano_136", df[df["matched_age_sex_year"] == 1].copy(), True),
    ]

    with REPORT_TXT.open("w", encoding="utf-8") as report:
        report.write("AVFAD — REAVALIAÇÃO COM SEGMENTOS /a/ ANOTADOS NO PRT\n")
        report.write("=" * 68 + "\n")
        report.write("Nested CV: externo StratifiedGroupKFold 5-fold x 2; interno 4-fold.\n")
        report.write("Os dois participantes de cada par sintético ficam no mesmo fold.\n")
        report.write("Pareamento é por estrato demográfico, não par clínico original.\n")
        report.write("Modelos acústicos: MFCC sem MFCC0 médio, DWT e WPT extraídos do intervalo /a/ anotado no PRT.\n")
        report.write("Baseline demográfico separado: idade, sexo e ano.\n")
        report.write("Normalização, imputação e seleção de características dentro do pipeline.\n")
        report.write("\nCaracterísticas: " + ", ".join(f"{k}={len(v)}" for k, v in all_groups.items()) + "\n")

        # Documenta a única possível divergência entre o Sound anotado no PRT
        # e o WAV externo. As características são extraídas do Sound do PRT,
        # pois é esse objeto que o TextGrid efetivamente anota.
        if {"prt_same_shape_rate_as_wav", "prt_waveform_matches_wav"}.issubset(df.columns):
            same_shape = pd.to_numeric(df["prt_same_shape_rate_as_wav"], errors="coerce")
            same_wave = pd.to_numeric(df["prt_waveform_matches_wav"], errors="coerce")
            mismatch_mask = (same_shape != 1) | (same_wave != 1)
            mismatch_rows = df.loc[mismatch_mask].copy()
            report.write("\nINTEGRIDADE DO SOUND EMBUTIDO NO PRT\n")
            report.write(
                f"Sound com mesma forma/taxa do WAV: {int((same_shape == 1).sum())}/{len(df)}\n"
            )
            report.write(
                f"Forma de onda correspondente ao WAV (tol. 1e-5): {int((same_wave == 1).sum())}/{len(df)}\n"
            )
            report.write(f"Divergências documentadas: {len(mismatch_rows)}\n")
            if len(mismatch_rows):
                report.write("Participantes com divergência (não substituídos automaticamente):\n")
                for _, r in mismatch_rows.iterrows():
                    report.write(
                        f"  {r['participant_id']} | classe={r['label']} | "
                        f"coorte 162={int(r['matched_age_sex'])} | "
                        f"coorte 136={int(r['matched_age_sex_year'])} | "
                        f"duração /a/={float(r['annotation_duration_s']):.4f}s\n"
                    )
            report.write(
                "As características usam o Sound do próprio PRT, que é a fonte alinhada ao TextGrid.\n"
            )
        else:
            report.write("\nAVISO: colunas de comparação PRT/WAV ausentes no CSV de características.\n")

        if "annotation_duration_s" in df.columns:
            ann_duration = pd.to_numeric(df["annotation_duration_s"], errors="coerce")
            report.write("\nDURAÇÃO DOS INTERVALOS ANOTADOS /a/\n")
            report.write(
                f"Mínima={ann_duration.min():.4f}s | mediana={ann_duration.median():.4f}s | "
                f"máxima={ann_duration.max():.4f}s\n"
            )

        for name, data, include_year in cohorts:
            if data.empty:
                raise ValueError(f"Coorte vazia: {name}")
            data, pair_groups, pair_count = build_pair_groups(data, include_year)
            report.write(f"\n{ name }: participantes={len(data)}, pares={pair_count}\n")
            evaluate_cohort(name, data, pair_groups, all_groups, report)

        report.write("\nArquivos gerados:\n")
        report.write(f"  {FOLDS_CSV.name}\n  {PREDICTIONS_CSV.name}\n  {REPORT_TXT.name}\n")

    print("\nConcluído.")
    print(f"Relatório: {REPORT_TXT}")
    print(f"Folds: {FOLDS_CSV}")
    print(f"Predições: {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
