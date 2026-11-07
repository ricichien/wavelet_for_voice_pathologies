# -*- coding: utf-8 -*-
"""
AVFAD - avaliação inicial com validação cruzada aninhada.

Entrada:
    raiz do repositorio\\features_avfad_hsa_002.csv

Saídas:
    resultados_folds_avfad.csv
    resumo_resultados_avfad.csv
    predicoes_oof_avfad.csv
    resumo_modelagem_avfad.txt

Metodologia:
- Uma linha por participante (não trata segmentos como pessoas independentes).
- Três coortes: HSA completa, pareada por sexo/idade e pareada por
  sexo/idade/ano.
- Comparação de MFCC, DWT, WPT e combinações.
- Repeated Stratified 5-fold externo (2 repetições).
- GridSearchCV 4-fold interno para selecionar classificador e hiperparâmetros.
- Seleção de características e normalização acontecem dentro dos folds.
- Métrica de seleção: balanced accuracy.
- Idade, sexo, ano, duração, número de segmentos e local NÃO entram como
  características preditoras.

Observação:
As avaliações dos diferentes conjuntos de características são estimativas
por nested CV, adequadas para comparação exploratória. A escolha posterior do
melhor resultado entre várias famílias ainda deve ser interpretada como uma
comparação múltipla, não como um teste externo independente.
"""

from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import csv
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
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
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = project_root()
INPUT_CSV = asset(ROOT, "features_avfad_hsa_002.csv")
FOLDS_CSV = asset(ROOT, "resultados_folds_avfad.csv")
SUMMARY_CSV = asset(ROOT, "resumo_resultados_avfad.csv")
PREDICTIONS_CSV = asset(ROOT, "predicoes_oof_avfad.csv")
REPORT_TXT = asset(ROOT, "resumo_modelagem_avfad.txt")

RANDOM_SEED = 42
OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 4
N_JOBS = 4  # conservador para máquinas com 16 GB de RAM

# Começamos pela coorte metodologicamente mais controlada (136 pessoas).
# Depois, para executar análises secundárias, altere esta lista para incluir:
# "Pareada_sexo_idade_162" e/ou "HSA_completa_411".
COHORTS_TO_RUN = ["Pareada_sexo_idade_ano_136"]


class SelectColumns(BaseEstimator, TransformerMixin):
    """Seleciona colunas nomeadas de um DataFrame dentro do pipeline sklearn."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is None or len(self.columns) == 0:
            raise ValueError("Nenhuma coluna de característica foi selecionada.")
        missing = [col for col in self.columns if col not in X.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no CSV: {missing[:10]}")
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):
        return X.loc[:, list(self.columns)].to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.columns), dtype=object)


def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Não encontrei {INPUT_CSV}. Coloque o script na pasta AVFAD "
            "e confirme que features_avfad_hsa_002.csv está lá."
        )

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required = {
        "participant_id", "label", "matched_age_sex", "matched_age_sex_year"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    if df["participant_id"].duplicated().any():
        duplicated = df.loc[df["participant_id"].duplicated(), "participant_id"].tolist()
        raise ValueError(f"Há participantes duplicados no CSV: {duplicated[:10]}")

    labels = set(df["label"].dropna().unique())
    if labels != {"Normal", "Pathological"}:
        raise ValueError(f"Rótulos inesperados: {labels}")

    df = df.copy()
    df["y"] = (df["label"] == "Pathological").astype(int)
    df["matched_age_sex"] = pd.to_numeric(df["matched_age_sex"], errors="coerce").fillna(0).astype(int)
    df["matched_age_sex_year"] = pd.to_numeric(df["matched_age_sex_year"], errors="coerce").fillna(0).astype(int)

    return df


def build_feature_groups(df):
    cols = list(df.columns)

    mfcc_all = [
        c for c in cols
        if c.startswith("mfcc_") or c.startswith("delta_") or c.startswith("delta2_")
    ]
    # O MFCC de ordem zero médio é particularmente sensível ao nível absoluto
    # de gravação; será isolado em uma análise de sensibilidade.
    mfcc_no_level = [c for c in mfcc_all if c != "mfcc_00_mean"]
    dwt = [c for c in cols if c.startswith("dwt_")]
    wpt = [c for c in cols if c.startswith("wpt_")]
    signal = [c for c in cols if c.startswith("signal_")]
    level_cols = [c for c in ("mfcc_00_mean", "signal_rms", "signal_rms_db") if c in cols]
    zcr = [c for c in ("signal_zcr",) if c in cols]

    groups = {
        "MFCC_sem_nivel": mfcc_no_level,
        "DWT": dwt,
        "WPT": wpt,
        "MFCC_DWT": mfcc_no_level + dwt,
        "MFCC_WPT": mfcc_no_level + wpt,
        "DWT_WPT": dwt + wpt,
        "TODAS_sem_nivel": mfcc_no_level + dwt + wpt + zcr,
        "TODAS_com_nivel": mfcc_all + dwt + wpt + signal,
        "NIVEL_amplitude": level_cols,
    }

    # Remove duplicatas preservando a ordem e verifica os grupos.
    cleaned = {}
    for name, feature_cols in groups.items():
        feature_cols = list(dict.fromkeys(feature_cols))
        if not feature_cols:
            raise ValueError(f"O grupo {name} ficou sem características.")
        cleaned[name] = feature_cols

    expected_families = {
        "MFCC": len(mfcc_all),
        "DWT": len(dwt),
        "WPT": len(wpt),
        "sinal": len(signal),
    }
    if not all(expected_families.values()):
        raise ValueError(f"Não encontrei todas as famílias de características: {expected_families}")

    return cleaned, expected_families


def make_param_grid(feature_cols):
    n_features = len(feature_cols)
    k_values = sorted({min(k, n_features) for k in (5, 10, 20, 30)})

    common = {
        "colunas__columns": [tuple(feature_cols)],
        "selecao__k": k_values,
    }

    grids = []

    # Regressão logística: baseline linear regularizado.
    p = dict(common)
    p.update({
        "modelo": [LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=4000,
            random_state=RANDOM_SEED,
        )],
        "modelo__C": [0.1, 1.0, 10.0],
    })
    grids.append(p)

    # SVM RBF: captura fronteiras não lineares; escala ajustada dentro do fold.
    p = dict(common)
    p.update({
        "modelo": [SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            random_state=RANDOM_SEED,
        )],
        "modelo__C": [0.1, 1.0, 10.0],
        "modelo__gamma": ["scale"],
    })
    grids.append(p)

    # Random Forest com árvores moderadas e folhas mínimas para conter overfitting.
    p = dict(common)
    p.update({
        "modelo": [RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            random_state=RANDOM_SEED,
            n_jobs=1,
        )],
        "modelo__max_depth": [None, 6],
        "modelo__min_samples_leaf": [2, 5],
        "modelo__max_features": ["sqrt"],
    })
    grids.append(p)

    return grids


def make_pipeline():
    return Pipeline([
        ("colunas", SelectColumns()),
        ("imputacao", SimpleImputer(strategy="median")),
        ("selecao", SelectKBest(score_func=f_classif)),
        ("escala", StandardScaler()),
        ("modelo", LogisticRegression()),
    ])


def get_continuous_score(estimator, X):
    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(X)
        return np.asarray(score, dtype=float).reshape(-1)
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)
        return np.asarray(probabilities[:, 1], dtype=float)
    return estimator.predict(X).astype(float)


def calculate_metrics(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_pathological": sensitivity,
        "specificity_normal": specificity,
        "precision_pathological": precision_score(y_true, y_pred, zero_division=0),
        "f1_pathological": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, score) if len(np.unique(y_true)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def append_csv(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(
        path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig" if write_header else "utf-8",
    )


def run_cohort(name, data, feature_groups, fold_rows, prediction_rows, report):
    y = data["y"].to_numpy(dtype=int)
    ids = data["participant_id"].astype(str).to_numpy()
    X = data.copy()

    # Apenas os identificadores e as variáveis de seleção permanecem no DataFrame.
    # O SelectColumns do pipeline garante que somente as colunas acústicas indicadas
    # em cada grupo cheguem ao classificador.
    class_counts = Counter(y)
    if min(class_counts.values()) < OUTER_SPLITS:
        raise ValueError(f"Amostra {name} tem poucos participantes em uma classe: {class_counts}")

    outer = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_SEED,
    )
    total_folds = OUTER_SPLITS * OUTER_REPEATS

    report.write(
        f"\nCOORTE: {name} | N={len(data)} | Normal={class_counts[0]} | "
        f"Patológico={class_counts[1]}\n"
    )

    for split_no, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        repeat_no = (split_no - 1) // OUTER_SPLITS + 1
        fold_no = (split_no - 1) % OUTER_SPLITS + 1
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        inner = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED + split_no,
        )

        this_fold_rows = []
        this_prediction_rows = []

        # Baseline sem áudio: classificador que sempre prevê a classe majoritária.
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(y_train), 1)), y_train)
        dummy_pred = dummy.predict(np.zeros((len(y_test), 1)))
        dummy_score = dummy_pred.astype(float)
        dummy_metrics = calculate_metrics(y_test, dummy_pred, dummy_score)
        this_fold_rows.append({
            "cohort": name,
            "repeat": repeat_no,
            "fold": fold_no,
            "feature_set": "BASELINE_majoritaria",
            "selected_model": "Dummy_majority",
            "inner_balanced_accuracy": np.nan,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "selected_k": np.nan,
            "best_params": "{}",
            **dummy_metrics,
        })

        for group_index, (group_name, feature_cols) in enumerate(feature_groups.items(), start=1):
            print(
                f"  {name}: repetição {repeat_no}/{OUTER_REPEATS}, "
                f"fold {fold_no}/{OUTER_SPLITS}, "
                f"grupo {group_index}/{len(feature_groups)} ({group_name})",
                flush=True,
            )

            search = GridSearchCV(
                estimator=make_pipeline(),
                param_grid=make_param_grid(feature_cols),
                scoring="balanced_accuracy",
                cv=inner,
                refit=True,
                n_jobs=N_JOBS,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_train, y_train)

            best = search.best_estimator_
            y_pred = best.predict(X_test).astype(int)
            score = get_continuous_score(best, X_test)
            metrics = calculate_metrics(y_test, y_pred, score)

            best_k = search.best_params_.get("selecao__k")
            best_model = best.named_steps["modelo"]
            model_name = type(best_model).__name__
            params_to_save = {
                key: value
                for key, value in search.best_params_.items()
                if key not in {"colunas__columns", "modelo"}
            }

            this_fold_rows.append({
                "cohort": name,
                "repeat": repeat_no,
                "fold": fold_no,
                "feature_set": group_name,
                "selected_model": model_name,
                "inner_balanced_accuracy": float(search.best_score_),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "selected_k": best_k,
                "best_params": json.dumps(params_to_save, ensure_ascii=False, default=str),
                **metrics,
            })

            for pid, true_label, predicted_label, continuous in zip(ids[test_idx], y_test, y_pred, score):
                this_prediction_rows.append({
                    "cohort": name,
                    "repeat": repeat_no,
                    "fold": fold_no,
                    "participant_id": pid,
                    "y_true": int(true_label),
                    "y_pred": int(predicted_label),
                    "score_pathological": float(continuous),
                    "feature_set": group_name,
                    "selected_model": model_name,
                })

        # Salva ao final de cada fold externo; uma interrupção não perde folds concluídos.
        append_csv(FOLDS_CSV, this_fold_rows)
        append_csv(PREDICTIONS_CSV, this_prediction_rows)
        fold_rows.extend(this_fold_rows)
        prediction_rows.extend(this_prediction_rows)

        print(
            f"  Concluído {name}: fold externo {split_no}/{total_folds}",
            flush=True,
        )


def build_summary(fold_df):
    metric_cols = [
        "accuracy", "balanced_accuracy", "sensitivity_pathological",
        "specificity_normal", "precision_pathological", "f1_pathological", "roc_auc"
    ]
    rows = []

    for (cohort, feature_set), group in fold_df.groupby(["cohort", "feature_set"], sort=False):
        row = {
            "cohort": cohort,
            "feature_set": feature_set,
            "n_outer_folds": len(group),
            "selected_models": "; ".join(
                f"{model}={count}"
                for model, count in group["selected_model"].value_counts().items()
            ),
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    # Começar uma execução limpa, evitando misturar resultados de execuções anteriores.
    for path in (FOLDS_CSV, SUMMARY_CSV, PREDICTIONS_CSV, REPORT_TXT):
        if path.exists():
            path.unlink()

    print("Lendo características...", flush=True)
    df = load_data()
    feature_groups, family_counts = build_feature_groups(df)

    all_cohorts = {
        "HSA_completa_411": df.copy(),
        "Pareada_sexo_idade_162": df[df["matched_age_sex"] == 1].copy(),
        "Pareada_sexo_idade_ano_136": df[df["matched_age_sex_year"] == 1].copy(),
    }
    unknown = [name for name in COHORTS_TO_RUN if name not in all_cohorts]
    if unknown:
        raise ValueError(f"Nomes de coorte inválidos em COHORTS_TO_RUN: {unknown}")
    cohorts = {name: all_cohorts[name] for name in COHORTS_TO_RUN}

    fold_rows = []
    prediction_rows = []

    with REPORT_TXT.open("w", encoding="utf-8") as report:
        report.write("MODELAGEM AVFAD — TAREFA 002 (/a/)\n")
        report.write("=" * 65 + "\n")
        report.write("Validação cruzada aninhada: externo 5-fold x 2 repetições; interno 4-fold.\n")
        report.write("Seleção interna por balanced accuracy.\n")
        report.write("Somente características acústicas entram nos modelos.\n")
        report.write("Idade, sexo, ano, duração, local e número de segmentos não são preditores.\n")
        report.write(f"Coortes executadas: {', '.join(COHORTS_TO_RUN)}\n")
        report.write("\nFamílias de características disponíveis:\n")
        for name, count in family_counts.items():
            report.write(f"  {name}: {count}\n")
        report.write("\nGrupos avaliados:\n")
        for name, columns in feature_groups.items():
            report.write(f"  {name}: {len(columns)} características\n")

        for cohort_name, cohort_data in cohorts.items():
            run_cohort(
                cohort_name,
                cohort_data,
                feature_groups,
                fold_rows,
                prediction_rows,
                report,
            )

        fold_df = pd.DataFrame(fold_rows)
        summary_df = build_summary(fold_df)
        summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

        report.write("\n\nRESUMO POR COORTE E GRUPO DE CARACTERÍSTICAS\n")
        report.write("Médias e desvios-padrão entre os folds externos.\n")
        for _, row in summary_df.iterrows():
            report.write(f"\n{row['cohort']} | {row['feature_set']}\n")
            for metric in (
                "accuracy", "balanced_accuracy", "sensitivity_pathological",
                "specificity_normal", "f1_pathological", "roc_auc"
            ):
                mean = row.get(f"{metric}_mean", np.nan)
                std = row.get(f"{metric}_std", np.nan)
                if pd.notna(mean):
                    report.write(f"  {metric}: {mean:.4f} +/- {std:.4f}\n")
            report.write(f"  Modelos selecionados: {row['selected_models']}\n")

        report.write("\nArquivos gerados:\n")
        report.write(f"  {FOLDS_CSV.name}\n")
        report.write(f"  {SUMMARY_CSV.name}\n")
        report.write(f"  {PREDICTIONS_CSV.name}\n")
        report.write(f"  {REPORT_TXT.name}\n")

    print("\nMODELAGEM CONCLUÍDA.", flush=True)
    print(f"Relatório: {REPORT_TXT}")
    print(f"Resumo CSV: {SUMMARY_CSV}")
    print(f"Detalhes por fold: {FOLDS_CSV}")
    print(f"Predições fora da amostra: {PREDICTIONS_CSV}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário. Folds já concluídos foram salvos nos CSVs.")
        raise
    except Exception as exc:
        print(f"ERRO: {exc}")
        raise
