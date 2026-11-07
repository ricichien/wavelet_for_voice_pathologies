# -*- coding: utf-8 -*-
"""Experimento 14: abla busca e limiar na WPT sym5 sem reextrair audio.

Use --verificar para leitura, --autoteste para testes sinteticos, ou --executar
para registrar protocolo e executar a CV aninhada em uma pasta NOVA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


VERSION = "1.0"
DEFAULT_SEED = 20261105
DEFAULT_SEEDS = (DEFAULT_SEED, 101, 102, 103, 104, 105)
OUTER_FOLDS = 5
INNER_FOLDS = 3
GRID = ((64, 0.1), (64, 1.0), (128, 0.1), (128, 1.0))
ARMS = ("A_WPT_fixa", "B_busca_restrita_zero", "C_busca_restrita_limiar")
QUANTILES = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
DEFAULT_OUTPUT = Path("results/experiments/resultados_14_ablacao_busca_limiar_wpt")
LOG_PATH = None


def say(message):
    print(message, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} | {message}\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_text(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.17g")
    os.replace(temporary, path)


def freeze_text(path, value):
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"Documento congelado difere do protocolo atual: {path}")
    else:
        atomic_text(path, value)


def persons(meta):
    return meta[["participant_id", "y"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True)


def validate_metadata(meta, real=True):
    need = {"sample_id", "participant_id", "task", "y"}
    if not need.issubset(meta.columns):
        raise ValueError(f"Metadata sem colunas {need - set(meta.columns)}")
    meta = meta.copy()
    if meta[list(need)].isna().any().any():
        raise ValueError("Metadata com identificador, tarefa ou rotulo ausente")
    for col in ("sample_id", "participant_id"):
        meta[col] = meta[col].astype(str).str.strip().str.upper()
        if (meta[col] == "").any():
            raise ValueError("Identificador vazio")
    meta["task"] = meta["task"].astype(str).str.zfill(3)
    yy = pd.to_numeric(meta["y"], errors="raise")
    if not set(yy).issubset({0, 1}):
        raise ValueError("Rotulos devem ser 0/1")
    meta["y"] = yy.astype(int)
    if meta.sample_id.duplicated().any() or meta.duplicated(["participant_id", "task"]).any():
        raise ValueError("Gravacao ou participante/tarefa duplicado")
    if (meta.groupby("participant_id").y.nunique() != 1).any():
        raise ValueError("Rotulos divergentes por pessoa")
    p = persons(meta)
    if real:
        if (len(meta), len(p), p.y.value_counts().to_dict()) != (2465, 411, {1: 329, 0: 82}):
            raise ValueError("Entrada nao corresponde a coorte HSA auditada 2465/411/82/329")
        if set(meta.task) != {"004", "005", "006", "007", "008", "009"}:
            raise ValueError("Tarefas diferentes das seis CAPE-V")
        if meta.groupby("participant_id").size().value_counts().to_dict() != {6: 410, 5: 1}:
            raise ValueError("Numero de frases por pessoa inesperado")
    return meta


def read_inputs(root):
    wp = root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz"
    mp = root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv"
    meta = validate_metadata(pd.read_csv(mp))
    with np.load(wp, allow_pickle=False) as data:
        names = np.asarray(data["feature_names"]).astype(str)
        indices = np.flatnonzero(np.char.startswith(names, "WPT__sym5__"))
        matrix = data["wavelet"]
        if matrix.shape != (2465, 4160) or len(indices) != 1920 or len(set(names)) != len(names):
            raise ValueError("Dimensao ou nomes do NPZ Wavelet inesperados")
        embedded_ids = "sample_id" in data.files
        if embedded_ids and not np.array_equal(np.asarray(data["sample_id"]).astype(str), meta.sample_id):
            raise ValueError("Ordem de IDs do NPZ difere da metadata")
        x = np.asarray(matrix[:, indices], dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    return x, meta, names[indices], (wp, mp), embedded_ids


def row_splits(meta, count, seed):
    p = persons(meta)
    cv = StratifiedKFold(n_splits=count, shuffle=True, random_state=seed)
    for tr, te in cv.split(p.participant_id, p.y):
        train_ids = set(p.iloc[tr].participant_id)
        test_ids = set(p.iloc[te].participant_id)
        if train_ids & test_ids:
            raise AssertionError("Participante em treino e validacao")
        train = np.flatnonzero(meta.participant_id.isin(train_ids).to_numpy())
        test = np.flatnonzero(meta.participant_id.isin(test_ids).to_numpy())
        yield train, test


def pipeline(k, c):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("remove_constants", VarianceThreshold()),
        ("select", SelectKBest(f_classif, k=k)),
        ("scale", StandardScaler()),
        ("classifier", SVC(kernel="linear", C=c, class_weight="balanced", cache_size=300)),
    ])


def fit_model(x, meta, k, c):
    clf = pipeline(k, c)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        clf.fit(x, meta.y.to_numpy())
    return clf


def aggregate(meta, scores):
    frame = meta[["participant_id", "y"]].copy()
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise ValueError("Scores invalidos")
    frame["score"] = scores
    return frame.groupby("participant_id", as_index=False).agg(
        y=("y", "first"), score=("score", "mean"), n_gravacoes=("score", "size")
    ).sort_values("participant_id").reset_index(drop=True)


def chosen_names(clf, names):
    original_indices = np.flatnonzero(clf.named_steps["remove_constants"].get_support())
    selected = original_indices[clf.named_steps["select"].get_support()]
    return [(int(i), str(names[i])) for i in selected]


def select_threshold(y, score):
    thresholds = np.unique(np.r_[0.0, np.quantile(score, QUANTILES)])
    results = [(float(balanced_accuracy_score(y, score >= t)), float(t)) for t in thresholds]
    peak = max(v for v, _ in results)
    ba, threshold = sorted((v for v in results if v[0] >= peak - 0.003), key=lambda v: (abs(v[1]), -v[0]))[0]
    return threshold, ba


def evaluate_candidates(x, meta, names, seed, fold):
    summary, predictions, selected = [], [], []
    for k, c in GRID:
        current = []
        for inner, (tr, va) in enumerate(row_splits(meta, INNER_FOLDS, seed + 100 + fold), 1):
            clf = fit_model(x[tr], meta.iloc[tr], k, c)
            pred = aggregate(meta.iloc[va], clf.decision_function(x[va]))
            pred["inner_fold"] = inner
            current.append(pred)
            for index, name in chosen_names(clf, names):
                selected.append(dict(stage="inner", inner_fold=inner, configuracao="candidato", k=k, C=c,
                                     indice_atributo=index, atributo=name))
        oof = pd.concat(current, ignore_index=True).sort_values("participant_id").reset_index(drop=True)
        if oof.participant_id.duplicated().any() or set(oof.participant_id) != set(meta.participant_id):
            raise AssertionError("OOF interno incompleto ou duplicado")
        ba = balanced_accuracy_score(oof.y, oof.score >= 0)
        summary.append(dict(k=k, C=c, BA_interna_zero=float(ba), n_participantes=len(oof)))
        oof["k"], oof["C"] = k, c
        predictions.append(oof)
        say(f"  seed={seed} fold={fold} candidato k={k} C={c:g} BA interna zero={ba:.4f}")
    candidates = pd.DataFrame(summary).sort_values(["BA_interna_zero", "k", "C"], ascending=[False, True, True], kind="stable").reset_index(drop=True)
    all_oof = pd.concat(predictions, ignore_index=True)
    best = candidates.iloc[0]
    chosen = all_oof[(all_oof.k == int(best.k)) & (all_oof.C == float(best.C))].copy()
    threshold, threshold_ba = select_threshold(chosen.y.to_numpy(), chosen.score.to_numpy())
    candidates["selecionado"] = (candidates.k == int(best.k)) & (candidates.C == float(best.C))
    return candidates, all_oof, pd.DataFrame(selected), int(best.k), float(best.C), threshold, threshold_ba


def run_fold(x, meta, names, train, test, seed, fold):
    if set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise AssertionError("Vazamento no fold externo")
    started = time.perf_counter()
    m = meta.iloc[train].reset_index(drop=True)
    candidates, inner_oof, features, k, c, threshold, threshold_ba = evaluate_candidates(x[train], m, names, seed, fold)
    outputs, outer_features = [], []
    fitted = {}
    for arm, kk, cc, tt in ((ARMS[0], 128, 1.0, 0.0), (ARMS[1], k, c, 0.0), (ARMS[2], k, c, threshold)):
        key = (kk, cc)
        if key not in fitted:
            fitted[key] = fit_model(x[train], m, kk, cc)
        clf = fitted[key]
        pred = aggregate(meta.iloc[test], clf.decision_function(x[test]))
        pred["configuracao"], pred["k"], pred["C"], pred["limiar"] = arm, kk, cc, tt
        pred["predicao"] = (pred.score >= tt).astype(int)
        outputs.append(pred)
        for index, name in chosen_names(clf, names):
            outer_features.append(dict(stage="outer", inner_fold=0, configuracao=arm, k=kk, C=cc,
                                       indice_atributo=index, atributo=name))
    pred = pd.concat(outputs, ignore_index=True)
    features = pd.concat([features, pd.DataFrame(outer_features)], ignore_index=True)
    for frame in (pred, candidates, inner_oof, features):
        frame["seed"], frame["outer_fold"] = seed, fold
    choice = dict(seed=seed, outer_fold=fold, k=k, C=c, limiar_C=threshold,
                  BA_interna_C=threshold_ba, segundos=time.perf_counter() - started)
    return {"predicoes.csv": pred, "candidatos.csv": candidates, "oof_interno.csv": inner_oof,
            "atributos_selecionados.csv": features, "escolha.csv": pd.DataFrame([choice])}


def expected_fold_tables(meta, seeds):
    external, internal = [], []
    for seed in seeds:
        for fold, (tr, te) in enumerate(row_splits(meta, OUTER_FOLDS, seed), 1):
            for pid, y in persons(meta.iloc[te]).itertuples(index=False, name=None):
                external.append(dict(seed=seed, outer_fold=fold, participant_id=pid, y=y))
            train_meta = meta.iloc[tr].reset_index(drop=True)
            for inner, (_, va) in enumerate(row_splits(train_meta, INNER_FOLDS, seed + 100 + fold), 1):
                for pid, y in persons(train_meta.iloc[va]).itertuples(index=False, name=None):
                    internal.append(dict(seed=seed, outer_fold=fold, inner_fold=inner, participant_id=pid, y=y))
    return pd.DataFrame(external), pd.DataFrame(internal)


def check_fold_outputs(data, meta, names, train, test, seed, fold):
    pred, candidates, oof, features, choice = (data[n] for n in ("predicoes.csv", "candidatos.csv", "oof_interno.csv", "atributos_selecionados.csv", "escolha.csv"))
    expected_test = persons(meta.iloc[test]).set_index("participant_id").y.to_dict()
    expected_train = persons(meta.iloc[train]).set_index("participant_id").y.to_dict()
    n_recordings = meta.iloc[test].groupby("participant_id").size().to_dict()
    if set(pred.configuracao) != set(ARMS):
        raise ValueError("Checkpoint nao possui exatamente os tres bracos")
    for arm in ARMS:
        sub = pred[pred.configuracao == arm]
        if sub.participant_id.duplicated().any() or sub.set_index("participant_id").y.to_dict() != expected_test:
            raise ValueError("IDs/rotulos inesperados nas predicoes externas")
        if sub.set_index("participant_id").n_gravacoes.to_dict() != n_recordings:
            raise ValueError("Contagem de frases incorreta")
    for frame in (pred, candidates, oof, features, choice):
        if not (frame.seed == seed).all() or not (frame.outer_fold == fold).all():
            raise ValueError("Seed/fold incorreto no checkpoint")
    if not np.isfinite(pred[["score", "limiar"]].to_numpy()).all() or not np.array_equal(pred.predicao, (pred.score >= pred.limiar).astype(int)):
        raise ValueError("Score ou decisao externa inconsistente")
    a = pred[pred.configuracao == ARMS[0]]
    if not ((a.k == 128) & (a.C == 1) & (a.limiar == 0)).all():
        raise ValueError("Referencia fixa alterada")
    b = pred[pred.configuracao == ARMS[1]].sort_values("participant_id")
    d = pred[pred.configuracao == ARMS[2]].sort_values("participant_id")
    if not np.array_equal(b.score.to_numpy(), d.score.to_numpy()) or not (b.limiar == 0).all():
        raise ValueError("B/C devem compartilhar scores e B usa zero")
    if len(candidates) != len(GRID) or set(zip(candidates.k, candidates.C)) != set(GRID):
        raise ValueError("Grade interna incompleta")
    sorted_candidates = candidates.sort_values(["BA_interna_zero", "k", "C"], ascending=[False, True, True], kind="stable")
    best = sorted_candidates.iloc[0]
    if not ((b.k == best.k) & (b.C == best.C)).all() or not ((d.k == best.k) & (d.C == best.C)).all():
        raise ValueError("Selecao difere do maximo interno predefinido")
    inner_mapping = {}
    mt = meta.iloc[train].reset_index(drop=True)
    for inn, (_, va) in enumerate(row_splits(mt, INNER_FOLDS, seed + 100 + fold), 1):
        inner_mapping.update({pid: inn for pid in mt.iloc[va].participant_id.unique()})
    for kk, cc in GRID:
        sub = oof[(oof.k == kk) & (oof.C == cc)]
        if sub.participant_id.duplicated().any() or sub.set_index("participant_id").y.to_dict() != expected_train:
            raise ValueError("OOF interno incompleto ou com rotulos incorretos")
        if sub.set_index("participant_id").inner_fold.to_dict() != inner_mapping:
            raise ValueError("Folds internos do checkpoint incorretos")
        saved_ba = candidates[(candidates.k == kk) & (candidates.C == cc)].BA_interna_zero.iloc[0]
        if not np.isclose(saved_ba, balanced_accuracy_score(sub.y, sub.score >= 0), atol=1e-12, rtol=0):
            raise ValueError("BA de candidato nao corresponde aos scores internos")
    best_oof = oof[(oof.k == best.k) & (oof.C == best.C)]
    threshold, inner_ba = select_threshold(best_oof.y.to_numpy(), best_oof.score.to_numpy())
    if not np.allclose(d.limiar, threshold, rtol=0, atol=1e-12):
        raise ValueError("Limiar C nao corresponde ao OOF interno de B")
    if len(choice) != 1 or choice.k.iloc[0] != best.k or choice.C.iloc[0] != best.C or not np.isclose(choice.limiar_C.iloc[0], threshold, rtol=0, atol=1e-12) or not np.isclose(choice.BA_interna_C.iloc[0], inner_ba, rtol=0, atol=1e-12):
        raise ValueError("Escolha salva nao corresponde aos candidatos/OOF")
    expected_groups = {("inner", inn, "candidato", kk, cc) for inn in range(1, INNER_FOLDS + 1) for kk, cc in GRID}
    expected_groups.update({("outer", 0, ARMS[0], 128, 1.0), ("outer", 0, ARMS[1], int(best.k), float(best.C)), ("outer", 0, ARMS[2], int(best.k), float(best.C))})
    if set(map(tuple, features[["stage", "inner_fold", "configuracao", "k", "C"]].drop_duplicates().to_numpy())) != expected_groups:
        raise ValueError("Faltam grupos de atributos selecionados")
    if not np.isfinite(oof.score).all() or not features.atributo.isin(names).all():
        raise ValueError("OOF ou atributos selecionados invalidos")
    for _, group in features.groupby(["stage", "inner_fold", "configuracao", "k", "C"]):
        if len(group) != int(group.k.iloc[0]) or group.atributo.duplicated().any():
            raise ValueError("Quantidade de atributos selecionados incorreta")
        for row in group.itertuples():
            if names[int(row.indice_atributo)] != row.atributo:
                raise ValueError("Indice de atributo inconsistente")


def metrics(frame):
    y, pred, score = frame.y.to_numpy(), frame.predicao.to_numpy(), frame.score.to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n_participantes=len(frame), BA=float(balanced_accuracy_score(y, pred)),
                sensibilidade=float(tp / (tp + fn)), especificidade=float(tn / (tn + fp)),
                acuracia=float(np.mean(y == pred)), AUC_concatenada_exploratoria=float(roc_auc_score(y, score)),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def paired_intervals(raw, comparisons, iterations=5000):
    """Bootstrap estratificado dos participantes, conservando todas as seeds no cluster."""
    pivot = raw.pivot(index=["participant_id", "y", "seed"], columns="configuracao", values="predicao").reset_index()
    rng = np.random.default_rng(DEFAULT_SEED)
    rows = []
    for alternative, reference, label in comparisons:
        diff = (pivot[alternative].to_numpy() == pivot.y.to_numpy()).astype(float) - (pivot[reference].to_numpy() == pivot.y.to_numpy()).astype(float)
        work = pivot[["participant_id", "y"]].copy()
        work["delta_acerto"] = diff
        # Uma linha por pessoa: media de acerto nas mesmas seeds, sem multiplicar N.
        per_person = work.groupby(["participant_id", "y"]).delta_acerto.mean().reset_index()
        a = per_person.loc[per_person.y == 0, "delta_acerto"].to_numpy()
        b = per_person.loc[per_person.y == 1, "delta_acerto"].to_numpy()
        boot = (a[rng.integers(0, len(a), (iterations, len(a)))].mean(axis=1) + b[rng.integers(0, len(b), (iterations, len(b)))].mean(axis=1)) / 2
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append(dict(comparacao=label, delta_BA=float((a.mean() + b.mean()) / 2),
                         IC95_inferior=float(lo), IC95_superior=float(hi), n_participantes=len(per_person),
                         n_seeds=pivot.seed.nunique(), bootstrap_iteracoes=iterations))
    return pd.DataFrame(rows)


def consolidate(out, raw, candidates, choices, features, seeds):
    fold_rows = []
    for (seed, fold, arm), group in raw.groupby(["seed", "outer_fold", "configuracao"]):
        mm = metrics(group)
        mm["AUC_fold"] = mm.pop("AUC_concatenada_exploratoria")
        mm["peso_pares_AUC"] = int((group.y == 0).sum() * (group.y == 1).sum())
        fold_rows.append(dict(seed=seed, outer_fold=fold, configuracao=arm, **mm))
    fold_metrics = pd.DataFrame(fold_rows)
    seed_rows = []
    for (seed, arm), group in raw.groupby(["seed", "configuracao"]):
        if group.participant_id.duplicated().any() or len(group) != raw.participant_id.nunique():
            raise ValueError("OOF completo nao possui uma predicao por participante/seed/braco")
        folds = fold_metrics[(fold_metrics.seed == seed) & (fold_metrics.configuracao == arm)]
        seed_rows.append(dict(seed=seed, configuracao=arm,
                              AUC_media_folds_ponderada_pares=float(np.average(folds.AUC_fold, weights=folds.peso_pares_AUC)),
                              **metrics(group)))
    seed_metrics = pd.DataFrame(seed_rows)
    comparisons = ((ARMS[1], ARMS[0], "B_menos_A_primaria"), (ARMS[2], ARMS[1], "C_menos_B_secundaria"), (ARMS[2], ARMS[0], "C_menos_A_exploratoria"))
    scopes = []
    if DEFAULT_SEED in seeds:
        scopes.append((f"principal_seed_{DEFAULT_SEED}", (DEFAULT_SEED,)))
    stability = tuple(s for s in seeds if s != DEFAULT_SEED)
    if stability:
        scopes.append(("robustez_seeds_" + "_".join(map(str, stability)), stability))
    summaries, intervals_list = [], []
    for scope, chosen_seeds in scopes:
        scope_metrics = seed_metrics[seed_metrics.seed.isin(chosen_seeds)]
        ss = scope_metrics.groupby("configuracao").agg(
            BA_media=("BA", "mean"), BA_min=("BA", "min"), BA_max=("BA", "max"), BA_DP_seeds=("BA", "std"),
            sensibilidade_media=("sensibilidade", "mean"), especificidade_media=("especificidade", "mean"),
            AUC_media_folds=("AUC_media_folds_ponderada_pares", "mean"),
            AUC_concatenada_exploratoria=("AUC_concatenada_exploratoria", "mean"),
        ).reset_index()
        ss.insert(0, "escopo", scope)
        ss["n_seeds"] = len(chosen_seeds)
        summaries.append(ss)
        interval = paired_intervals(raw[raw.seed.isin(chosen_seeds)], comparisons)
        interval.insert(0, "escopo", scope)
        intervals_list.append(interval)
    summary = pd.concat(summaries, ignore_index=True)
    intervals = pd.concat(intervals_list, ignore_index=True)
    per_seed = []
    indexed = seed_metrics.set_index(["seed", "configuracao"])
    for alternative, reference, label in comparisons:
        for seed in seeds:
            per_seed.append(dict(comparacao=label, seed=seed, delta_BA=float(indexed.loc[(seed, alternative), "BA"] - indexed.loc[(seed, reference), "BA"])))
    artifacts = {"predicoes_oof_participante.csv": raw, "metricas_por_fold.csv": fold_metrics,
                 "metricas_por_seed.csv": seed_metrics, "resumo_final.csv": summary,
                 "comparacoes_pareadas.csv": intervals, "diferencas_por_seed.csv": pd.DataFrame(per_seed),
                 "candidatos_todos_folds.csv": candidates, "escolhas_por_fold.csv": choices,
                 "atributos_selecionados_todos_folds.csv": features}
    for filename, frame in artifacts.items():
        atomic_csv(out / filename, frame)
    msg = ["EXPERIMENTO 14 - ABLACAO DE BUSCA E LIMIAR WPT SYM5", "",
           "Estado: execucao completa; resultado de desenvolvimento na AVFAD, sem conclusao final.",
           f"Seeds predefinidas: {list(seeds)}. Mesmos {raw.participant_id.nunique()} participantes por seed.",
           f"Analise principal: seed {DEFAULT_SEED} (mesmos folds do13); robustez: demais seeds, reportadas separadamente.",
           "A: SVC linear, k128/C1/limiar0; B: busca k64/128 e C0.1/1 por BA interna com limiar0;",
           "C: mesmos hiperparametros e scores de B, apenas limiar ajustado no OOF interno.", "",
           "BA media de OOF por participante; AUC principal abaixo e media dos folds ponderada pelo numero de pares normal/patologico.",
           "AUC de scores concatenados e apenas exploratoria: margens de folds distintos nao sao calibradas.", ""]
    for row in summary.itertuples(index=False):
        msg.append(f"{row.escopo} | {row.configuracao}: BA={row.BA_media:.6f}; intervalo de seeds=[{row.BA_min:.6f},{row.BA_max:.6f}]; sens={row.sensibilidade_media:.6f}; esp={row.especificidade_media:.6f}; AUC_media_folds={row.AUC_media_folds:.6f}")
    msg.extend(["", "COMPARACOES PAREADAS (IC95% bootstrap estratificado por pessoa)"])
    for row in intervals.itertuples(index=False):
        msg.append(f"{row.escopo} | {row.comparacao}: delta BA={row.delta_BA:+.6f}; IC95%=[{row.IC95_inferior:+.6f},{row.IC95_superior:+.6f}]")
    msg.extend(["", "LIMITES E INTERPRETACAO",
                "- Os mesmos participantes ja foram usados no desenvolvimento de experimentos anteriores; nao ha teste independente.",
                "- IC condicional a estas predicoes, preservando as seeds de cada pessoa no mesmo cluster; nao refaz treinamento/selecao.",
                "- Seeds nao sao novos participantes e o DP entre seeds nao e intervalo de confianca.",
                "- Comparacao primaria B-A; C-B isola a politica de limiar mantendo identicos modelos e scores externos.",
                "- Todas as variantes sao reportadas. Nao escolher melhor seed nem mudar protocolo a partir de resultados externos.",
                "- ANOVA e treinamento usam gravacoes correlacionadas, embora os folds e as metricas sejam por participante.",
                "- Balanceamento por classe de gravacao; a unica pessoa com cinco frases tem peso ligeiramente menor.",
                "- NPZ antigo sem IDs internos: vinculo com metadata depende da proveniencia e da auditoria do cache.",
                "- Nao ha ajuste de pre-processamento, reextracao, exclusao por resultado ou instalacao de dependencias.", "",
                "Consultar PROTOCOLO.txt, assinatura_execucao.json, folds_participantes_*.csv e checkpoints por seed/fold para auditoria."])
    atomic_text(out / "RELATORIO_RESULTADOS.txt", "\n".join(msg) + "\n")
    say("\n" + "\n".join(msg))


def identity(paths, seeds, embedded_ids):
    source = Path(__file__).resolve()
    config = dict(version=VERSION, seeds=list(seeds), primary_seed=DEFAULT_SEED,
                  stability_seeds=[s for s in seeds if s != DEFAULT_SEED], outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS,
                  inner_seed_policy="seed+100+outer_fold", grid=[list(v) for v in GRID], arms=list(ARMS),
                  tie_break="max BA interna zero, menor k, menor C", threshold_quantiles=list(QUANTILES),
                  threshold_tolerance=0.003, fixed_k=128, fixed_C=1.0, class_weight="balanced",
                  selection_unit="gravacao", evaluation_unit="participante", aggregate="media decision_function",
                  bootstrap_iterations=5000, embedded_sample_ids=embedded_ids)
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl"):
        environment[name] = importlib.metadata.version(name)
    inputs = {str(p.resolve()): {"sha256": sha256(p), "size_bytes": p.stat().st_size} for p in paths}
    return dict(configuration=config, input_files=inputs, code={str(source): sha256(source)}, environment=environment)


def protocol(ident):
    seeds = ident["configuration"]["seeds"]
    return f"""EXPERIMENTO 14 - PROTOCOLO CONGELADO ANTES DO TREINAMENTO

Hipotese: a busca conjunta ampla e o limiar aprendido em margens internas podem
ser instaveis nesta coorte. Uma busca pequena linear com limiar zero pode reduzir
essa instabilidade. Resultado positivo nao e garantido.

Coorte: AVFAD HSA, 411 participantes (82 normais/329 patologicos), 2465 gravacoes
CAPE-V 004-009; 410 pessoas com seis frases, uma com cinco. Nenhuma exclusao nova.
Entrada: WPT sym5 temporal L5+L6, 1920 atributos por gravacao, matriz ja extraida.

Seeds externas predefinidas: {seeds}; 5 folds estratificados numa tabela de pessoas.
Analise principal: seed20261105, identica a particao do13; analise secundaria de
estabilidade: seeds101..105, se presentes. Separar os dois escopos no resumo e IC.
Todas as seeds sao declaradas antes de qualquer treinamento; nenhuma e escolhida
apos observar seu resultado. As cinco repeticoes reutilizam os mesmos individuos.
Inner: 3 folds na tabela de pessoas do treino externo; seed externa+100+numero fold.
As frases da mesma pessoa nunca atravessam treino/validacao/teste.
Imputacao mediana, VarianceThreshold, ANOVA SelectKBest e StandardScaler sao
ajustados apenas no treino de cada fold interno; refit somente no treino externo.
Estimador unico: SVC linear, class_weight=balanced. Sem SMOTE ou calibracao.

A_WPT_fixa: k=128, C=1, limiar=0.
B_busca_restrita_zero: k em {{64,128}}, C em {{0.1,1}}, quatro candidatos.
Escolha pela maior BA OOF interna por PARTICIPANTE com limiar zero;
desempate exato por menor k, depois menor C. Nenhuma AUC concatenada e usada na escolha.
C_busca_restrita_limiar: MESMOS hiperparametros/modelo/scores externos de B;
apenas limiar escolhido no OOF interno do candidato vencedor de B.
Limiar: zero e quantis {QUANTILES}; tolerancia BA de 0.003 ao pico, preferencia
por menor modulo do limiar; empate seguinte pela maior BA, conforme experimento13.

Agregacao: media das margens decision_function de todas as frases disponiveis da
pessoa, depois limiar. Uma predicao por pessoa/seed/braco. Metricas internas iguais.
Comparacao PRIMARIA: B menos A em BA. SECUNDARIA: C menos B; C-A exploratoria.
Relatar BA, sensibilidade, especificidade, matriz de confusao por fold/seed,
distribuicao entre seeds e AUC por fold (media ponderada por pares normal/patologico).
AUC concatenada entre folds e apenas exploratoria, sem alegacao de calibracao.
IC95% por 5000 bootstraps estratificados pareados de participantes, preservando
todas as seeds de cada individuo no mesmo cluster. IC condicional aos OOF;
nao refaz selecao/treinamento, nao elimina adaptacao historica a esta coorte.

Registrar candidatos, OOF internos, atributos selecionados, particoes, escolhas,
hashes SHA256 de conteudo dos dois insumos e deste script e versoes do ambiente.
Checkpoints so sao retomados apos verificar assinatura, hashes de artefatos e
consistencia de IDs, rotulos, particoes, scores, limiares e politica de selecao.
Nenhum resultado anterior sera sobrescrito; diretorio exclusivo desta execucao.

Esta e uma avaliacao de DESENVOLVIMENTO apos exploracao previa dos mesmos 411
individuos. Novas seeds nao sao teste independente. Nao escolher seed favoravel.
SVD localizada em pasta externa, com inventario parcial documentado em
documentacao_tg/04_INVENTARIO_SVD.txt. Confirmacao depende de completar a
proveniencia dos audios patologicos e auditar coorte, tarefas e rotulos.
Nao ha conclusao final nem promessa de atingir BA especifica.
"""


def prepare(out, ident, meta, seeds):
    signature_file = out / "assinatura_execucao.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf-8")) != ident:
            raise RuntimeError("Dados/codigo/ambiente/config diferem da execucao existente; use nova pasta")
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError("Pasta nao vazia sem assinatura: nao sobrescrever")
        atomic_text(signature_file, json_text(ident))
        atomic_text(out / "REGISTRO_INICIO.txt", "Protocolo registrado em UTC: " + datetime.now(timezone.utc).isoformat() + "\n")
    freeze_text(out / "PROTOCOLO.txt", protocol(ident))
    external, internal = expected_fold_tables(meta, seeds)
    # Csv com comparacao byte a byte reproduzivel antes dos fits.
    for name, frame in (("folds_participantes_externos.csv", external), ("folds_participantes_internos.csv", internal)):
        value = frame.to_csv(index=False, lineterminator="\n")
        freeze_text(out / name, value)


def execute(x, meta, names, out, ident, seeds):
    global LOG_PATH
    prepare(out, ident, meta, seeds)
    LOG_PATH = out / "EXECUCAO.log.txt"
    run_signature = hashlib.sha256(json_text(ident).encode("utf-8")).hexdigest()
    all_data = []
    for seed in seeds:
        for fold, (tr, te) in enumerate(row_splits(meta, OUTER_FOLDS, seed), 1):
            directory = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            manifest_file = directory / "COMPLETO.json"
            if manifest_file.exists():
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
                if manifest["run_signature"] != run_signature:
                    raise ValueError("Assinatura do checkpoint incorreta")
                files = manifest["artifacts"]
                expected = {"predicoes.csv", "candidatos.csv", "oof_interno.csv", "atributos_selecionados.csv", "escolha.csv"}
                if set(files) != expected or any(sha256(directory / n) != h for n, h in files.items()):
                    raise ValueError(f"Checkpoint alterado ou incompleto: {directory}")
                data = {name: pd.read_csv(directory / name, float_precision="round_trip") for name in files}
                check_fold_outputs(data, meta, names, tr, te, seed, fold)
                say(f"Checkpoint validado seed={seed} fold={fold}; nenhum ajuste repetido")
            else:
                if directory.exists() and any(directory.iterdir()):
                    raise RuntimeError(f"Checkpoint parcial preservado; revisar antes de retomar: {directory}")
                say(f"INICIO seed={seed} fold={fold}/{OUTER_FOLDS}")
                data = run_fold(x, meta, names, tr, te, seed, fold)
                check_fold_outputs(data, meta, names, tr, te, seed, fold)
                for name, frame in data.items():
                    atomic_csv(directory / name, frame)
                manifest = dict(run_signature=run_signature, completed_utc=datetime.now(timezone.utc).isoformat(),
                                artifacts={name: sha256(directory / name) for name in data})
                atomic_text(manifest_file, json_text(manifest))
                say(f"CONCLUIDO seed={seed} fold={fold}; segundos={data['escolha.csv'].segundos.iloc[0]:.2f}")
            all_data.append(data)
    combined = {name: pd.concat([item[name] for item in all_data], ignore_index=True) for name in all_data[0]}
    consolidate(out, combined["predicoes.csv"], combined["candidatos.csv"], combined["escolha.csv"], combined["atributos_selecionados.csv"], seeds)


def selftest():
    global LOG_PATH
    rng = np.random.default_rng(230)
    meta = pd.DataFrame([dict(participant_id=f"S{pid:03d}", sample_id=f"S{pid:03d}_{task:03d}",
                              task=f"{task:03d}", y=pid % 2) for pid in range(36) for task in range(3)])
    meta = validate_metadata(meta, real=False)
    x = rng.normal(size=(len(meta), 160)).astype(np.float32)
    x[:, :8] += meta.y.to_numpy()[:, None] * 0.4
    # Ausencia e constante exercitam imputacao e remocao sem impedir k128.
    x[::11, 12] = np.nan
    x[:, -1] = 1
    names = np.array([f"sintetico_{i}" for i in range(x.shape[1])])
    tr, te = next(row_splits(meta, OUTER_FOLDS, DEFAULT_SEED))
    data = run_fold(x, meta, names, tr, te, DEFAULT_SEED, 1)
    check_fold_outputs(data, meta, names, tr, te, DEFAULT_SEED, 1)
    altered = x.copy()
    altered[te] = 1e8 * rng.normal(size=altered[te].shape)
    again = run_fold(altered, meta, names, tr, te, DEFAULT_SEED, 1)
    # Contaminacao extrema de todos os testes NAO altera candidatos/OOF internos,
    # atributos escolhidos nem limiar. Modelos podem, corretamente, prever diferente.
    for filename in ("candidatos.csv", "oof_interno.csv", "atributos_selecionados.csv"):
        pd.testing.assert_frame_equal(data[filename], again[filename])
    assert data["escolha.csv"].limiar_C.iloc[0] == again["escolha.csv"].limiar_C.iloc[0]
    # Estatisticas do scaler/imputer e suporte dependem apenas do treino.
    model = fit_model(x[tr], meta.iloc[tr], 64, 0.1)
    center = model.named_steps["scale"].mean_.copy()
    model.decision_function(altered[te])
    np.testing.assert_array_equal(center, model.named_steps["scale"].mean_)
    with TemporaryDirectory(prefix="avfad_exp14_test_") as temporary:
        directory = Path(temporary)
        for name, frame in data.items():
            atomic_csv(directory / name, frame)
        loaded = {n: pd.read_csv(directory / n, float_precision="round_trip") for n in data}
        check_fold_outputs(loaded, meta, names, tr, te, DEFAULT_SEED, 1)
        # Corrupcao de um rotulo e rejeitada antes de qualquer reutilizacao.
        corrupt = {n: f.copy() for n, f in loaded.items()}
        corrupt["predicoes.csv"].loc[0, "y"] = 1 - corrupt["predicoes.csv"].loc[0, "y"]
        try:
            check_fold_outputs(corrupt, meta, names, tr, te, DEFAULT_SEED, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Checkpoint com rotulo corrompido foi aceito")
    # Identidade de seeds: duplicar as mesmas predicoes em nova seed nao deve
    # estreitar artificialmente o IC de participantes.
    pred = data["predicoes.csv"]
    duplicate = pred.assign(seed=999)
    comparison = [(ARMS[1], ARMS[0], "B-A")]
    once = paired_intervals(pred, comparison, iterations=500)
    twice = paired_intervals(pd.concat([pred, duplicate]), comparison, iterations=500)
    pd.testing.assert_frame_equal(once.drop(columns="n_seeds"), twice.drop(columns="n_seeds"))
    # Exercita protocolo, todos os folds, consolidacao TXT/CSV e retomada real
    # dos checkpoints, usando apenas a pequena coorte sintetica temporaria.
    with TemporaryDirectory(prefix="avfad_exp14_fulltest_") as temporary:
        saved_log = LOG_PATH
        try:
            destination = Path(temporary) / "resultado_sintetico"
            ident = identity([], (DEFAULT_SEED,), True)
            execute(x, meta, names, destination, ident, (DEFAULT_SEED,))
            before = {str(p.relative_to(destination)): sha256(p) for p in destination.glob("checkpoints/*/*") if p.is_file()}
            execute(x, meta, names, destination, ident, (DEFAULT_SEED,))
            after = {str(p.relative_to(destination)): sha256(p) for p in destination.glob("checkpoints/*/*") if p.is_file()}
            assert before == after, "Retomada alterou checkpoints completos"
            summary = pd.read_csv(destination / "resumo_final.csv")
            assert set(summary.configuracao) == set(ARMS) and summary.BA_media.between(0, 1).all()
            assert (destination / "RELATORIO_RESULTADOS.txt").is_file()
        finally:
            LOG_PATH = saved_log
    say("AUTOTESTE OK: isolamento por pessoa/teste, B=C scores, roundtrip CSV, corrupcao rejeitada, bootstrap por cluster.")


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--saida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                        help="Seeds congeladas; padrao: 20261105 principal, 101 102 103 104 105 robustez")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true", help="Somente leitura; nao grava protocolo")
    action.add_argument("--executar", action="store_true", help="Registra protocolo e executa/retoma apenas nesta pasta")
    action.add_argument("--autoteste", action="store_true", help="Apenas sintetico e temporario, sem AVFAD")
    args = parser.parse_args()
    if args.autoteste:
        selftest()
        return
    seeds = tuple(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)) or any(s < 0 or s > 2**32 - 1000 for s in seeds):
        parser.error("Seeds devem ser unicas e validas para sklearn/NumPy")
    root = args.dados.resolve()
    out = (args.saida if args.saida.is_absolute() else root / args.saida).resolve()
    # Resultados permanecem sob experiments, fora de todos os insumos existentes.
    experiments = (root / "results/experiments").resolve()
    if experiments not in out.parents:
        parser.error("--saida deve ser uma subpasta de results/experiments")
    x, meta, names, paths, embedded = read_inputs(root)
    ident = identity(paths, seeds, embedded)
    say(f"ENTRADAS OK: {x.shape}, {len(persons(meta))} pessoas; seeds={seeds}; NPZ com IDs={embedded}")
    if not args.executar:
        say("VERIFICACAO OK: nenhum arquivo gravado, nenhum modelo treinado.")
        return
    execute(x, meta, names, out, ident, seeds)


if __name__ == "__main__":
    try:
        cli()
    except Exception:
        if LOG_PATH is not None:
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
        raise
