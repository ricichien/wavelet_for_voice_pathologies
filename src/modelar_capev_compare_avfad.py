# -*- coding: utf-8 -*-
"""Modelagem CAPE-V AVFAD: ComParE 2016 vs MFCC/DWT/WPT.

Entrada (gerada por extrair_features_capev_compare_avfad.py):
  features_capev_compare_avfad.npz
  metadata_capev_avfad.csv

Unidade de avaliacao principal: PARTICIPANTE.
Cada arquivo/frase e usado como amostra de treinamento, mas todas as frases
 do mesmo participante permanecem no mesmo fold. As metricas por gravacao
 sao reportadas apenas como comparacao secundaria. Para a metrica principal,
os escores das frases de cada participante sao agregados pela media.

Cohortes:
  HSA_411: todos os participantes HSA (82 Normais / 329 Patologicos).
  Pareada_sexo_idade_162: coorte pareada por sexo e faixa etaria de 5 anos.
  Pareada_sexo_idade_ano_136: coorte pareada tambem por ano.

Classificador de referencia: SVC(C=1, kernel='poly', degree=1), seguindo
 a implementacao publica dos autores (kernel de grau 1, linearmente separavel).
 Escalonamento e imputacao sao ajustados exclusivamente no treino de cada fold.

Este experimento NAO e uma reproducao exata das metricas publicadas: usa HSA,
validacao estratificada por participante e uma coorte controlada por local.
"""

from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import csv
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


ROOT = project_root()
NPZ_PATH = asset(ROOT, "features_capev_compare_avfad.npz")
META_PATH = asset(ROOT, "metadata_capev_avfad.csv")
REPORT_PATH = asset(ROOT, "resumo_modelagem_capev_compare_avfad.txt")
FOLDS_PATH = asset(ROOT, "folds_modelagem_capev_compare_avfad.csv")
RECORD_PRED_PATH = asset(ROOT, "predicoes_gravacao_capev_compare_avfad.csv")
PERSON_PRED_PATH = asset(ROOT, "predicoes_participante_capev_compare_avfad.csv")
TASK_METRICS_PATH = asset(ROOT, "metricas_por_frase_capev_compare_avfad.csv")

SEED = 42
N_SPLITS = 5
TASKS = ("004", "005", "006", "007", "008", "009")


class ExperimentError(RuntimeError):
    pass


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_progress(report_lines, folds_out, record_preds_out, person_preds_out, task_metrics_out):
    """Persiste resultados ja concluidos para evitar perder uma rodada longa."""
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    write_csv(FOLDS_PATH, folds_out)
    write_csv(RECORD_PRED_PATH, record_preds_out)
    write_csv(PERSON_PRED_PATH, person_preds_out)
    write_csv(TASK_METRICS_PATH, task_metrics_out)


def load_inputs():
    if not NPZ_PATH.exists() or not META_PATH.exists():
        raise ExperimentError(
            "Nao encontrei os arquivos de extracao. Esperados:\n"
            f"  {NPZ_PATH}\n  {META_PATH}"
        )

    meta = pd.read_csv(META_PATH, encoding="utf-8-sig")
    required = {
        "row_index", "sample_id", "participant_id", "task", "label", "y",
        "age", "sex", "visit_year", "visit_place",
        "matched_age_sex", "matched_age_sex_year",
    }
    missing = required - set(meta.columns)
    if missing:
        raise ExperimentError(f"Colunas ausentes em metadata: {sorted(missing)}")

    meta = meta.sort_values("row_index").reset_index(drop=True)
    if meta["row_index"].duplicated().any():
        raise ExperimentError("row_index duplicado nos metadados.")
    if meta["row_index"].isna().any():
        raise ExperimentError("row_index ausente nos metadados.")

    with np.load(NPZ_PATH, allow_pickle=False) as data:
        compare = data["compare"].astype(np.float64, copy=False)
        acoustic = data["acoustic"].astype(np.float64, copy=False)
        compare_names = data["compare_names"].astype(str).tolist()
        acoustic_names = data["acoustic_names"].astype(str).tolist()

    if len(meta) != compare.shape[0] or len(meta) != acoustic.shape[0]:
        raise ExperimentError(
            "Quantidade de linhas dos metadados difere das matrizes NPZ: "
            f"meta={len(meta)}, ComParE={compare.shape[0]}, proprio={acoustic.shape[0]}"
        )
    expected_rows = np.arange(len(meta))
    if not np.array_equal(meta["row_index"].to_numpy(dtype=int), expected_rows):
        raise ExperimentError("row_index nao e sequencial/alinhado as matrizes NPZ.")
    if compare.shape[1] != 6373:
        raise ExperimentError(f"ComParE tem {compare.shape[1]} colunas; esperado 6373.")
    if acoustic.shape[1] != 295:
        raise ExperimentError(f"Matriz propria tem {acoustic.shape[1]} colunas; esperado 295.")
    if len(compare_names) != compare.shape[1] or len(acoustic_names) != acoustic.shape[1]:
        raise ExperimentError("Dimensoes dos nomes de caracteristicas inconsistentes.")

    # Trata infinitos como ausentes; imputacao acontece dentro de cada fold.
    compare[~np.isfinite(compare)] = np.nan
    acoustic[~np.isfinite(acoustic)] = np.nan

    meta["participant_id"] = meta["participant_id"].astype(str).str.strip().str.upper()
    meta["task"] = meta["task"].astype(str).str.strip().str.zfill(3)
    meta["label"] = meta["label"].astype(str).str.strip()
    meta["y"] = pd.to_numeric(meta["y"], errors="raise").astype(int)
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    meta["visit_year"] = pd.to_numeric(meta["visit_year"], errors="coerce")
    meta["sex"] = meta["sex"].astype(str).str.strip().str.upper()
    meta["matched_age_sex"] = pd.to_numeric(meta["matched_age_sex"], errors="coerce").fillna(0).astype(int)
    meta["matched_age_sex_year"] = pd.to_numeric(meta["matched_age_sex_year"], errors="coerce").fillna(0).astype(int)

    if set(meta["label"].unique()) - {"Normal", "Pathological"}:
        raise ExperimentError("Rotulos inesperados nos metadados.")
    if not np.array_equal(meta["y"].to_numpy(), (meta["label"] == "Pathological").astype(int).to_numpy()):
        raise ExperimentError("A coluna y nao corresponde a label.")
    if meta.duplicated(["participant_id", "task"]).any():
        raise ExperimentError("Ha mais de uma linha para participante/tarefa.")
    if not set(meta["task"].unique()).issubset(set(TASKS)):
        raise ExperimentError("Existem tarefas fora de 004-009.")

    # Cada participante deve ter um unico rotulo e um unico valor demografico.
    for col in ("label", "y", "age", "sex", "visit_year", "visit_place",
                "matched_age_sex", "matched_age_sex_year"):
        nunique = meta.groupby("participant_id")[col].nunique(dropna=False)
        if (nunique > 1).any():
            raise ExperimentError(f"Metadados variam entre gravacoes para {col}.")

    # Feature groups are selected by the saved feature names, never by guessed offsets.
    compare_idx = np.arange(compare.shape[1])
    mfcc_idx = np.array([
        i for i, name in enumerate(acoustic_names)
        if name.startswith(("mfcc_", "delta_", "delta2_")) and name != "mfcc_00_mean"
    ], dtype=int)
    dwt_idx = np.array([i for i, name in enumerate(acoustic_names) if name.startswith("dwt_")], dtype=int)
    wpt_idx = np.array([i for i, name in enumerate(acoustic_names) if name.startswith("wpt_")], dtype=int)

    if len(mfcc_idx) != 149 or len(dwt_idx) != 14 or len(wpt_idx) != 128:
        raise ExperimentError(
            "Dimensoes dos grupos proprios inesperadas: "
            f"MFCC={len(mfcc_idx)}, DWT={len(dwt_idx)}, WPT={len(wpt_idx)}"
        )

    features = {
        "ComParE_2016": compare[:, compare_idx],
        "MFCC_sem_nivel": acoustic[:, mfcc_idx],
        "DWT": acoustic[:, dwt_idx],
        "WPT": acoustic[:, wpt_idx],
    }
    feature_dims = {name: matrix.shape[1] for name, matrix in features.items()}
    return meta, features, feature_dims


def age_band(value):
    if pd.isna(value):
        return None
    age = int(value)
    return 18 + ((age - 18) // 5) * 5


def build_pair_map(person_meta: pd.DataFrame, include_year: bool):
    """Recria pareamento por estrato, um Normal com um Patologico."""
    data = person_meta.drop_duplicates("participant_id").copy()
    data["age_band"] = data["age"].map(age_band)
    keys = ["sex", "age_band"]
    if include_year:
        keys.append("visit_year")

    pair_map = {}
    pair_number = 0
    for stratum, part in data.groupby(keys, sort=True, dropna=False):
        normals = sorted(part.loc[part["y"] == 0, "participant_id"].tolist())
        patients = sorted(part.loc[part["y"] == 1, "participant_id"].tolist())
        if len(normals) != len(patients):
            raise ExperimentError(
                f"Coorte pareada com contagens diferentes no estrato {stratum}: "
                f"Normal={len(normals)}, Patologico={len(patients)}"
            )
        for normal_id, patient_id in zip(normals, patients):
            pair_number += 1
            pair_id = f"PAIR_{pair_number:03d}"
            pair_map[normal_id] = pair_id
            pair_map[patient_id] = pair_id

    if len(pair_map) != len(data):
        missing = sorted(set(data["participant_id"]) - set(pair_map))
        raise ExperimentError(f"Nem todos os IDs receberam par: {missing[:10]}")
    return pair_map, pair_number


def make_model(class_weight):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=1.0,
            kernel="poly",
            degree=1,
            gamma="scale",
            coef0=0.0,
            class_weight=class_weight,
            probability=False,
            cache_size=1000,
        )),
    ])


def metric_dict(y_true, y_pred, scores):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = np.asarray(scores, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_pathological": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity_normal": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision_pathological": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_pathological": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def aggregate_person_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    grouped = pred_df.groupby("participant_id", sort=True).agg(
        y_true=("y_true", "first"),
        score=("score", "mean"),
        n_recordings=("task", "count"),
        tasks=("task", lambda values: ",".join(sorted(values.astype(str).unique()))),
        fold=("fold", "first"),
        repeat=("repeat", "first"),
    ).reset_index()
    if (pred_df.groupby("participant_id")["y_true"].nunique() > 1).any():
        raise ExperimentError("Um participante recebeu rotulos divergentes nas previsoes.")
    grouped["y_pred"] = (grouped["score"] >= 0.0).astype(int)
    return grouped


def evaluate_demographic_baseline(person_meta: pd.DataFrame, report):
    data = person_meta.drop_duplicates("participant_id").copy().reset_index(drop=True)
    y = data["y"].to_numpy(dtype=int)
    groups = data["participant_id"].to_numpy()
    X = data[["age", "sex", "visit_year"]].copy()
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = []
    folds = []

    pre = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), ["age", "visit_year"]),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), ["sex"]),
    ])

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = Pipeline([
            ("pre", pre),
            ("lr", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED)),
        ])
        model.fit(X.iloc[train_idx], y[train_idx])
        scores = model.decision_function(X.iloc[test_idx])
        preds = (scores >= 0).astype(int)
        fold_metric = metric_dict(y[test_idx], preds, scores)
        folds.append({"cohort": "HSA_411", "model": "Baseline_demografico", "level": "participante", "fold": fold, **fold_metric})
        for idx, score, pred in zip(test_idx, scores, preds):
            oof.append({
                "participant_id": data.iloc[idx]["participant_id"],
                "y_true": int(y[idx]), "score": float(score), "y_pred": int(pred),
                "fold": fold, "repeat": 1,
            })

    all_oof = pd.DataFrame(oof)
    pooled = metric_dict(all_oof["y_true"], all_oof["y_pred"], all_oof["score"])
    folds_df = pd.DataFrame(folds)
    report.write("\nBASELINE DEMOGRAFICO — HSA_411\n")
    report.write("Preditores: idade, sexo e ano; modelo separado dos atributos de audio.\n")
    report.write("Pooled OOF (uma previsao por participante):\n")
    for key, value in pooled.items():
        report.write(f"  {key}: {value}\n")
    report.write("Media +/- DP entre folds:\n")
    for metric in ("accuracy", "balanced_accuracy", "sensitivity_pathological", "specificity_normal", "f1_pathological", "roc_auc"):
        report.write(f"  {metric}: {folds_df[metric].mean():.4f} +/- {folds_df[metric].std(ddof=0):.4f}\n")
    return folds, oof


def evaluate_one(name, data, Xmat, feature_name, class_weight, group_col, repeat, folds_out, record_preds_out, person_preds_out, task_metrics_out):
    y = data["y"].to_numpy(dtype=int)
    groups = data[group_col].astype(str).to_numpy()
    X = Xmat
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat - 1
    )
    record_predictions = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        if train_groups & test_groups:
            raise ExperimentError(f"Vazamento de grupos em {name}/{feature_name}/fold {fold}.")
        if len(np.unique(y[train_idx])) != 2 or len(np.unique(y[test_idx])) != 2:
            raise ExperimentError(f"Fold sem as duas classes em {name}/{feature_name}/fold {fold}.")

        model = make_model(class_weight)
        print(f"  Fold {fold}/{N_SPLITS}: treino={len(train_idx)} gravacoes, teste={len(test_idx)} gravacoes")
        model.fit(X[train_idx], y[train_idx])
        scores = model.decision_function(X[test_idx])
        preds = (scores >= 0.0).astype(int)

        test_meta = data.iloc[test_idx].reset_index(drop=True)
        fold_frame = pd.DataFrame({
            "participant_id": test_meta["participant_id"].to_numpy(),
            "task": test_meta["task"].to_numpy(),
            "y_true": y[test_idx],
            "score": scores,
            "y_pred": preds,
            "fold": fold,
            "repeat": repeat,
        })
        record_predictions.append(fold_frame)

        rec_metric = metric_dict(y[test_idx], preds, scores)
        folds_out.append({
            "cohort": name, "model": feature_name,
            "class_weight": str(class_weight), "level": "gravacao",
            "repeat": repeat, "fold": fold, **rec_metric,
        })

        person_frame = aggregate_person_predictions(fold_frame)
        person_metric = metric_dict(
            person_frame["y_true"], person_frame["y_pred"], person_frame["score"]
        )
        folds_out.append({
            "cohort": name, "model": feature_name,
            "class_weight": str(class_weight), "level": "participante",
            "repeat": repeat, "fold": fold, **person_metric,
        })

    pred_frame = pd.concat(record_predictions, ignore_index=True)
    # Out-of-fold rows: each recording appears exactly once for this repeat.
    if pred_frame.duplicated(["participant_id", "task"]).any():
        raise ExperimentError(f"Gravacao repetida nas previsoes OOF de {name}/{feature_name}.")
    if pred_frame[["score", "y_pred"]].isna().any().any():
        raise ExperimentError("Previsoes contem valores ausentes.")

    person_frame = aggregate_person_predictions(pred_frame)
    record_metric = metric_dict(pred_frame["y_true"], pred_frame["y_pred"], pred_frame["score"])
    person_metric = metric_dict(person_frame["y_true"], person_frame["y_pred"], person_frame["score"])

    for row in pred_frame.to_dict("records"):
        record_preds_out.append({
            "cohort": name, "model": feature_name, "class_weight": str(class_weight), **row
        })
    for row in person_frame.to_dict("records"):
        person_preds_out.append({
            "cohort": name, "model": feature_name, "class_weight": str(class_weight), **row
        })

    # Desempenho por frase: uma amostra por pessoa em cada tarefa.
    for task, part in pred_frame.groupby("task", sort=True):
        if part["participant_id"].nunique() < 10 or part["y_true"].nunique() < 2:
            continue
        m = metric_dict(part["y_true"], part["y_pred"], part["score"])
        task_metrics_out.append({
            "cohort": name, "model": feature_name,
            "class_weight": str(class_weight), "task": task, **m,
        })

    return record_metric, person_metric


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    print("Carregando matrizes e metadados...")
    meta, features, feature_dims = load_inputs()

    people = meta.drop_duplicates("participant_id").copy()
    # Consistencia de IDs e contagens.
    people_counts = meta.groupby("participant_id").size()
    if len(people) != 411:
        raise ExperimentError(f"Esperava 411 participantes HSA, encontrei {len(people)}.")
    if (people["visit_place"].str.upper() != "HSA").any():
        raise ExperimentError("A extracao contem participante fora da HSA.")
    missing_audio_people = sorted(
    people_counts.loc[people_counts < 6].index.astype(str).tolist()
    )
    if any(count > 6 or count < 1 for count in people_counts):
        raise ExperimentError("Quantidade de frases por participante fora de 1-6.")
    if len(meta) != 2465:
        raise ExperimentError(f"Esperava 2465 gravacoes processadas; encontrei {len(meta)}.")

    report_lines = []
    report_lines.append("MODELAGEM CAPE-V AVFAD 004-009 — ComParE 2016 vs MFCC/DWT/WPT")
    report_lines.append("=" * 78)
    report_lines.append("Unidade de separacao: participante; as frases de uma pessoa nunca cruzam treino/teste.")
    report_lines.append("Validacao: StratifiedGroupKFold 5-fold, uma repeticao, seed=42.")
    report_lines.append("Classificador: SVC C=1, kernel polinomial grau 1 (linear), sem busca de hiperparametros.")
    report_lines.append("Escala e imputacao calculadas apenas no treino de cada fold.")
    report_lines.append("Metricas principais: uma previsao por participante, media dos escores das frases disponiveis.")
    report_lines.append("Metricas por gravacao e por frase sao secundarias (amostras correlacionadas por pessoa).")
    report_lines.append("")
    report_lines.append(f"Participantes HSA: {len(people)} | gravacoes: {len(meta)}")
    label_counts = Counter(people["label"])
    report_lines.append(f"Classes: Normal={label_counts['Normal']} | Pathological={label_counts['Pathological']}")
    report_lines.append(f"Frases por participante: min={people_counts.min()} | mediana={people_counts.median():.0f} | max={people_counts.max()}")
    report_lines.append(f"Participantes com frase ausente: {len(missing_audio_people)} ({', '.join(missing_audio_people) if missing_audio_people else 'nenhum'})")
    report_lines.append("Dimensoes: " + ", ".join(f"{k}={v}" for k, v in feature_dims.items()))
    report_lines.append("")
    report_lines.append("IMPORTANTE: HSA e um subconjunto localmente controlado, mas desbalanceado (82/329).")
    report_lines.append("Os resultados HSA nao sao comparacao direta com os 86,84% publicados para AVFAD completa.")
    report_lines.append("A classe positiva e Pathological (1); limiar da funcao de decisao = 0.")

    folds_out = []
    record_preds_out = []
    person_preds_out = []
    task_metrics_out = []

    # Build cohorts at participant level, then subset all recordings for those people.
    cohorts = []
    full_ids = set(people["participant_id"])
    cohorts.append(("HSA_411", full_ids, "participant_id", None))

    matched162 = people.loc[people["matched_age_sex"] == 1]
    pairmap162, n_pairs162 = build_pair_map(matched162, include_year=False)
    cohorts.append(("Pareada_sexo_idade_162", set(matched162["participant_id"]), "pair_id", pairmap162))

    matched136 = people.loc[people["matched_age_sex_year"] == 1]
    pairmap136, n_pairs136 = build_pair_map(matched136, include_year=True)
    cohorts.append(("Pareada_sexo_idade_ano_136", set(matched136["participant_id"]), "pair_id", pairmap136))

    # Main cohort: fixed original ComParE SVM both without and with class weighting.
    # Acoustic feature groups use balanced class weights due to HSA's 82/329 imbalance.
    for cohort_name, ids, group_kind, pairmap in cohorts:
        data = meta.loc[meta["participant_id"].isin(ids)].copy().reset_index(drop=True)
        if group_kind == "participant_id":
            data["cv_group"] = data["participant_id"]
        else:
            data["cv_group"] = data["participant_id"].map(pairmap)
            if data["cv_group"].isna().any():
                raise ExperimentError(f"Grupo de pareamento ausente em {cohort_name}.")

        persons = data.drop_duplicates("participant_id")
        counts = Counter(persons["label"])
        report_lines.append("")
        report_lines.append(
            f"COORTE {cohort_name}: participantes={len(persons)}, gravacoes={len(data)}, "
            f"Normal={counts['Normal']}, Pathological={counts['Pathological']}"
        )
        if cohort_name == "Pareada_sexo_idade_162":
            report_lines.append(f"Pares sinteticos preservados: {n_pairs162}")
        elif cohort_name == "Pareada_sexo_idade_ano_136":
            report_lines.append(f"Pares sinteticos preservados: {n_pairs136}")

        # Ensure matrices preserve metadata row order (meta is already row_index-sorted).
        row_indices = data["row_index"].to_numpy(dtype=int)

        if cohort_name == "HSA_411":
            runs = [
                ("ComParE_2016_sem_peso", "ComParE_2016", None),
                ("ComParE_2016_peso_balanceado", "ComParE_2016", "balanced"),
                ("MFCC_sem_nivel", "MFCC_sem_nivel", "balanced"),
                ("DWT", "DWT", "balanced"),
                ("WPT", "WPT", "balanced"),
            ]
        else:
            runs = [
                ("ComParE_2016", "ComParE_2016", None),
                ("MFCC_sem_nivel", "MFCC_sem_nivel", None),
                ("DWT", "DWT", None),
                ("WPT", "WPT", None),
            ]

        for model_label, feature_key, class_weight in runs:
            print(f"Avaliando {cohort_name} | {model_label}...")
            record_metric, person_metric = evaluate_one(
                cohort_name, data, features[feature_key][row_indices], model_label,
                class_weight, "cv_group", 1,
                folds_out, record_preds_out, person_preds_out, task_metrics_out,
            )
            report_lines.append(f"\n{model_label} | class_weight={class_weight}")
            report_lines.append("  Gravacao (OOF pooled, secundaria):")
            for key, value in record_metric.items():
                report_lines.append(f"    {key}: {value}")
            report_lines.append("  Participante (OOF pooled, PRINCIPAL):")
            for key, value in person_metric.items():
                report_lines.append(f"    {key}: {value}")

        # Fold-wise summary for participant-level metric on this cohort.
        cohort_fold_df = pd.DataFrame([
            row for row in folds_out
            if row["cohort"] == cohort_name and row["level"] == "participante"
        ])
        report_lines.append("\nMedia +/- DP entre folds (participante):")
        for model_name, part in cohort_fold_df.groupby("model", sort=False):
            report_lines.append(f"  {model_name}")
            for metric in ("accuracy", "balanced_accuracy", "sensitivity_pathological", "specificity_normal", "f1_pathological", "roc_auc"):
                report_lines.append(
                    f"    {metric}: {part[metric].mean():.4f} +/- {part[metric].std(ddof=0):.4f}"
                )
        save_progress(report_lines, folds_out, record_preds_out, person_preds_out, task_metrics_out)

    # Separate demographic baseline on one row per person, HSA only.
    report_lines.append("")
    report_lines.append("BASELINE DEMOGRAFICO — HSA_411")
    report_lines.append("Preditores: idade, sexo e ano; sem caracteristicas de audio.")
    demog_folds, demog_oof = evaluate_demographic_baseline(people, report_writer_proxy(report_lines))
    folds_out.extend(demog_folds)
    person_preds_out.extend([
        {"cohort": "HSA_411", "model": "Baseline_demografico", "class_weight": "balanced", **r}
        for r in demog_oof
    ])
    save_progress(report_lines, folds_out, record_preds_out, person_preds_out, task_metrics_out)

    # Task-level metrics for the pooled model are already accumulated.
    report_lines.append("")
    report_lines.append("METRICAS POR FRASE")
    report_lines.append("Cada frase tem uma previsao por participante; resultados sao secundarios.")
    if task_metrics_out:
        task_df = pd.DataFrame(task_metrics_out)
        for (cohort, model, cw), part in task_df.groupby(["cohort", "model", "class_weight"], sort=False):
            report_lines.append(f"\n{cohort} | {model} | class_weight={cw}")
            for _, row in part.sort_values("task").iterrows():
                report_lines.append(
                    f"  Tarefa {row['task']}: N={int(row['n'])}, "
                    f"BA={row['balanced_accuracy']:.4f}, AUC={row['roc_auc']:.4f}, "
                    f"Sens={row['sensitivity_pathological']:.4f}, Esp={row['specificity_normal']:.4f}"
                )

    report_lines.extend([
        "",
        "INTERPRETACAO",
        "1) Resultados por participante sao a referencia principal; cada pessoa conta uma vez.",
        "2) Resultados por gravacao sao secundarios e nao representam amostras independentes.",
        "3) HSA_411 controla o local de gravacao, mas tem desbalanceamento 82/329; por isso ha comparacao sem/com peso de classe para ComParE.",
        "4) Coortes pareadas sao analises de sensibilidade, nao substituem a amostra HSA completa.",
        "5) Este desenho nao replica os folds originais dos autores; e uma avaliacao agrupada por participante.",
        "6) O participante com arquivo ausente participa com as cinco frases disponiveis; nenhum participante foi descartado.",
        "",
        "ARQUIVOS GERADOS:",
        f"  {FOLDS_PATH.name}",
        f"  {RECORD_PRED_PATH.name}",
        f"  {PERSON_PRED_PATH.name}",
        f"  {TASK_METRICS_PATH.name}",
    ])

    save_progress(report_lines, folds_out, record_preds_out, person_preds_out, task_metrics_out)

    print("\nConcluido.")
    print(f"Relatorio: {REPORT_PATH}")
    print(f"Dobras: {FOLDS_PATH}")
    print(f"Previsoes por gravacao: {RECORD_PRED_PATH}")
    print(f"Previsoes por participante: {PERSON_PRED_PATH}")
    print(f"Metricas por frase: {TASK_METRICS_PATH}")


def report_writer_proxy(lines):
    """Adapta lista de linhas a uma interface write() simples."""
    class _Writer:
        def write(self, text):
            lines.extend(str(text).splitlines())
    return _Writer()


if __name__ == "__main__":
    main()
