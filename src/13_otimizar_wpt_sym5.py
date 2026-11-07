# -*- coding: utf-8 -*-
"""Experimento 13 — otimizacao WPT sym5 temporal na AVFAD.

Sem leitura dos ZIPs AVFAD e sem reextracao de WAV.

Protocolo: 5 folds EXTERNOS estratificados por participante; 3 internos para
selecionar SVC linear, RBF ou regressao logistica, k, C e limiar. As 6 frases de
cada pessoa permanecem no mesmo fold; as metricas agregam scores por pessoa.

Rodar (raiz raiz do repositorio):
  .\\.venv\\Scripts\\python.exe .\\src\\13_otimizar_wpt_sym5.py --verificar
  .\\.venv\\Scripts\\python.exe .\\src\\13_otimizar_wpt_sym5.py --executar

Saida: results/experiments/resultados_13_otimizacao_wpt_sym5/
Os folds concluidos sao retomados sem refazer treinamento.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Preespecificado. Nao editar com base nos resultados externos.
SEED = 20261105
OUTER_FOLDS = 5
INNER_FOLDS = 3
FAMILIES = ("SVC_linear", "SVC_RBF", "LogisticRegression")
TOP_K = (64, 128, 256, 512)
C_VALUES = (0.1, 1.0)
THRESHOLD_QUANTILES = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
N_EXPECTED = 2465
PEOPLE_EXPECTED = 411
N_NORMAL = 82
N_PATHO = 329
N_FEATURES = 1920
BASELINE = {"modelo": "SVC_linear", "k": 128, "C": 1.0, "limiar": 0.0}
RESULTS_REL = Path("results") / "experiments" / "resultados_13_otimizacao_wpt_sym5"


def say(message: str):
    print(message, flush=True)


def stable_csv(df: pd.DataFrame, filename: Path):
    """Gravacao atomica para evitar checkpoints truncados."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(filename.name + ".part")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, filename)


def stable_json(data: dict, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(filename.name + ".part")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, filename)


def read_inputs(root: Path, synthetic: bool = False):
    if synthetic:
        rng = np.random.default_rng(17)
        n_people = 48
        meta = pd.DataFrame([
            {"sample_id": f"S{p:03d}_{t}", "participant_id": f"S{p:03d}",
             "task": str(t).zfill(3), "y": int(p % 4 != 0)}
            for p in range(n_people) for t in range(3)
        ])
        X = rng.normal(0, 1, (len(meta), N_FEATURES)).astype("float32")
        X[meta["y"].to_numpy() == 1, :8] += 0.45
        return X, meta, {"source": "synthetic"}
    npz = root / "data" / "processed" / "features_wavelet_temporal_avfad_reduzido.npz"
    csv = root / "data" / "processed" / "metadata_wavelet_temporal_avfad_reduzido.csv"
    for p in (npz, csv):
        if not p.is_file():
            raise FileNotFoundError(f"Arquivo exigido ausente: {p}")
    meta = pd.read_csv(csv)
    missing = {"sample_id", "participant_id", "task", "y"} - set(meta.columns)
    if missing:
        raise ValueError(f"Metadados sem colunas: {sorted(missing)}")
    for col in ("sample_id", "participant_id"):
        if meta[col].isna().any():
            raise ValueError(f"{col} com dados ausentes")
        meta[col] = meta[col].astype(str).str.strip().str.upper()
    meta["task"] = meta["task"].astype(str).str.strip().str.zfill(3)
    meta["y"] = pd.to_numeric(meta["y"], errors="raise").astype("int8")
    if meta["sample_id"].duplicated().any():
        raise ValueError("sample_id duplicado no manifesto")
    if not set(meta["y"]).issubset({0, 1}):
        raise ValueError("y deve ser 0/1")
    if meta.groupby("participant_id")["y"].nunique().max() != 1:
        raise ValueError("Participante com rotulos divergentes")
    persons = meta[["participant_id", "y"]].drop_duplicates()
    counts = persons["y"].value_counts().to_dict()
    if (len(meta), len(persons), counts.get(0), counts.get(1)) != (N_EXPECTED, PEOPLE_EXPECTED, N_NORMAL, N_PATHO):
        raise ValueError(f"Coorte inesperada: {len(meta)} gravacoes, {len(persons)} pessoas, classes={counts}")
    with np.load(npz, allow_pickle=False) as data:
        if not {"wavelet", "feature_names"}.issubset(data.files):
            raise ValueError("NPZ nao contem 'wavelet' e 'feature_names'")
        names = np.asarray(data["feature_names"]).astype(str)
        idx = np.flatnonzero(np.char.startswith(names, "WPT__sym5__"))
        if len(idx) != N_FEATURES:
            raise ValueError(f"WPT__sym5__: {len(idx)} colunas; esperado {N_FEATURES}")
        array = data["wavelet"]
        if array.ndim != 2 or array.shape != (len(meta), len(names)):
            raise ValueError(f"NPZ/metadata: {array.shape} vs ({len(meta)}, {len(names)})")
        if "sample_id" in data.files:
            sample_ids = np.asarray(data["sample_id"]).astype(str)
            if list(sample_ids) != meta["sample_id"].tolist():
                raise ValueError("Ordem do NPZ diferente da metadata; nao avancar")
        # O extrator da AVFAD grava a matriz na mesma ordem do manifesto.
        # Se o NPZ nao contiver sample_id, esta correspondencia exige a
        # proveniencia do pipeline; nao pode ser auditada so com o NPZ.
        X = np.asarray(array[:, idx], dtype="float32")
    if not np.isfinite(X).all():
        X[~np.isfinite(X)] = np.nan
        say("AVISO: valores nao finitos encontrados; serao imputados por fold")
    files = {}
    for p in (npz, csv):
        st = p.stat()
        files[p.name] = {"size_bytes": st.st_size, "modified_ns": st.st_mtime_ns}
    ident = {
        "source_files": files,
        "participant_ids_sha256": hashlib.sha256(("|".join(persons["participant_id"])).encode()).hexdigest(),
        "samples_sha256": hashlib.sha256(("|".join(meta["sample_id"])).encode()).hexdigest(),
        "features_sha256": hashlib.sha256(("|".join(names[idx])).encode()).hexdigest(),
        "seed": SEED, "outer_folds": OUTER_FOLDS, "inner_folds": INNER_FOLDS,
        "models": list(FAMILIES), "top_k": list(TOP_K), "C_values": list(C_VALUES),
        "threshold_quantiles": list(THRESHOLD_QUANTILES), "baseline": BASELINE,
    }
    return X, meta, ident


def person_table(meta: pd.DataFrame):
    return meta[["participant_id", "y"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True)


def row_splits(meta: pd.DataFrame, n_splits: int, seed: int):
    """Indices de GRAVACOES produzidos a partir de folds por PARTICIPANTE."""
    people = person_table(meta)
    ids = meta["participant_id"].to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for a, b in splitter.split(np.zeros(len(people)), people["y"].to_numpy()):
        train_p = set(people.iloc[a]["participant_id"])
        test_p = set(people.iloc[b]["participant_id"])
        assert not (train_p & test_p)
        it = np.flatnonzero(np.isin(ids, list(train_p)))
        iv = np.flatnonzero(np.isin(ids, list(test_p)))
        assert len(set(ids[it]) & set(ids[iv])) == 0
        yield it, iv


def pipeline_for(model: str, k: int, c: float):
    common = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("remove_constants", VarianceThreshold()),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("scale", StandardScaler()),
    ]
    if model == "SVC_linear":
        est = SVC(kernel="linear", C=c, class_weight="balanced", cache_size=300)
    elif model == "SVC_RBF":
        est = SVC(kernel="rbf", C=c, gamma="scale", class_weight="balanced", cache_size=300)
    elif model == "LogisticRegression":
        est = LogisticRegression(C=c, max_iter=3000, class_weight="balanced", solver="lbfgs")
    else:
        raise ValueError(f"Modelo desconhecido: {model}")
    return Pipeline(common + [("classifier", est)])


def aggregate(meta_test: pd.DataFrame, raw_scores: np.ndarray):
    if len(meta_test) != len(raw_scores):
        raise ValueError("Metadados nao correspondem aos scores")
    rows = meta_test[["participant_id", "y"]].copy()
    rows["score"] = np.asarray(raw_scores, dtype=float)
    people = rows.groupby("participant_id", as_index=False).agg(
        y=("y", "first"), score=("score", "mean"), n_gravacoes=("score", "size")
    )
    if len(people) != meta_test["participant_id"].nunique():
        raise AssertionError("Numero de pessoas apos agregacao")
    return people.sort_values("participant_id").reset_index(drop=True)


def metricas(y, score, threshold):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "BA": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, score)),
        "acuracia": float(accuracy_score(y, pred)),
        "sensibilidade": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "especificidade": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def threshold_from_inner(y, scores):
    """Escolhe limiar pela BA OOF INTERNA; prefere 0 em empates quase exatos."""
    pool = np.unique(np.concatenate([[0.0], np.quantile(scores, THRESHOLD_QUANTILES)]))
    candidates = [(float(balanced_accuracy_score(y, scores >= t)), float(t)) for t in pool]
    peak = max(ba for ba, _ in candidates)
    almost = [(ba, t) for ba, t in candidates if ba >= peak - 0.003]
    # Prefere limiar mais proximo de zero se praticamente empatar.
    best = sorted(almost, key=lambda x: (abs(x[1]), -x[0]))[0]
    return best[1], best[0]


def evaluate_candidates(X, meta, train_rows, fold_number):
    """Busca SO no treino externo; cada candidato e avaliado em pessoas OOF internas."""
    m = meta.iloc[train_rows].reset_index(drop=True)
    xt = X[train_rows]
    internal = list(row_splits(m, INNER_FOLDS, SEED + 100 + fold_number))
    rows = []
    n_candidates = len(FAMILIES) * len(TOP_K) * len(C_VALUES)
    step = 0
    for model in FAMILIES:
        for k in TOP_K:
            for c in C_VALUES:
                step += 1
                preds = []
                for fit_idx, val_idx in internal:
                    clf = pipeline_for(model, k, c)
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        clf.fit(xt[fit_idx], m["y"].to_numpy()[fit_idx])
                    scores = clf.decision_function(xt[val_idx])
                    preds.append(aggregate(m.iloc[val_idx].reset_index(drop=True), scores))
                oof = pd.concat(preds, ignore_index=True)
                oof = oof.sort_values("participant_id").reset_index(drop=True)
                if oof["participant_id"].duplicated().any() or len(oof) != m["participant_id"].nunique():
                    raise AssertionError("OOF interno duplicado/incompleto")
                threshold, ba = threshold_from_inner(oof["y"].to_numpy(), oof["score"].to_numpy())
                auc = float(roc_auc_score(oof["y"], oof["score"]))
                rows.append({"outer_fold": fold_number, "modelo": model, "k": k,
                             "C": c, "limiar_interno": threshold, "BA_interna": ba, "AUC_interna": auc})
                say(f"  candidato {step:02d}/{n_candidates} | {model:18} k={k:3d} C={c:>4} BA_interna={ba:.4f}")
    candidates = pd.DataFrame(rows)
    candidates = candidates.sort_values(
        ["BA_interna", "AUC_interna", "k", "modelo"],
        ascending=[False, False, True, True], kind="stable"
    ).reset_index(drop=True)
    return candidates, candidates.iloc[0].to_dict()


def evaluate_outer_fold(X, meta, outer_train, outer_test, fold):
    if len(set(meta.iloc[outer_train]["participant_id"]) & set(meta.iloc[outer_test]["participant_id"])):
        raise AssertionError("Vazamento de participantes entre folds")
    say(f"\n===== FOLD EXTERNO {fold}/{OUTER_FOLDS} =====")
    candidates, choice = evaluate_candidates(X, meta, outer_train, fold)
    outputs = []
    for tag, model, k, c, threshold in (
        ("WPT_fixa", BASELINE["modelo"], BASELINE["k"], BASELINE["C"], BASELINE["limiar"]),
        ("WPT_otimizada", choice["modelo"], int(choice["k"]), float(choice["C"]), float(choice["limiar_interno"])),
    ):
        cls = pipeline_for(model, k, c)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            cls.fit(X[outer_train], meta.iloc[outer_train]["y"])
        probs = cls.decision_function(X[outer_test])
        people = aggregate(meta.iloc[outer_test].reset_index(drop=True), probs)
        people["outer_fold"] = fold
        people["configuracao"] = tag
        people["classificador"] = model
        people["k"] = k
        people["C"] = c
        people["limiar"] = threshold
        people["predicao"] = (people["score"] >= threshold).astype(int)
        mm = metricas(people["y"], people["score"], threshold)
        say(f"  -> {tag} | BA={mm['BA']:.4f} | AUC={mm['AUC']:.4f} | esp={mm['especificidade']:.4f}")
        outputs.append(people)
    return candidates, pd.concat(outputs, ignore_index=True)


def bootstrap_delta(pair, iterations=2000):
    y = pair["y"].to_numpy()
    sf = pair["score_fixo"].to_numpy()
    so = pair["score_otimizado"].to_numpy()
    tf = pair["limiar_fixo"].to_numpy()
    to = pair["limiar_otimizado"].to_numpy()
    i0 = np.flatnonzero(y == 0)
    i1 = np.flatnonzero(y == 1)
    rng = np.random.default_rng(20261105)
    deltas = []
    for _ in range(iterations):
        take = np.concatenate((rng.choice(i0, len(i0), replace=True), rng.choice(i1, len(i1), replace=True)))
        yy = y[take]
        ba_o = balanced_accuracy_score(yy, (so[take] >= to[take]).astype(int))
        ba_f = balanced_accuracy_score(yy, (sf[take] >= tf[take]).astype(int))
        deltas.append(ba_o - ba_f)
    return np.quantile(deltas, [0.025, 0.975]).tolist()


def consolidate(out: Path):
    files = [out / f"predicoes_outer_{i:02d}.csv" for i in range(1, OUTER_FOLDS + 1)]
    if not all(p.is_file() for p in files):
        return
    raw = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    for tag in ("WPT_fixa", "WPT_otimizada"):
        sub = raw[raw["configuracao"] == tag]
        if len(sub) != PEOPLE_EXPECTED or sub["participant_id"].nunique() != PEOPLE_EXPECTED:
            raise ValueError(f"OOF externo incompleto/duplicado em {tag}: {len(sub)}")
    fold_metrics = []
    for (tag, fold), sub in raw.groupby(["configuracao", "outer_fold"]):
        mm = metricas(sub["y"], sub["score"], sub["limiar"].iloc[0])
        fold_metrics.append({"configuracao": tag, "outer_fold": fold, "participantes": len(sub), **mm})
    stable_csv(pd.DataFrame(fold_metrics), out / "metricas_por_fold.csv")
    stable_csv(raw, out / "predicoes_oof_participante.csv")
    summary = []
    for tag, sub in raw.groupby("configuracao"):
        # Threshold difere por fold; portanto predicao ja vem consolidada por pessoa.
        y = sub["y"].to_numpy(dtype=int)
        pred = sub["predicao"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        summary.append({"configuracao": tag, "n_participantes": len(sub),
                        "BA_OOF": balanced_accuracy_score(y, pred),
                        "AUC_OOF": roc_auc_score(y, sub["score"]),
                        "acuracia_OOF": accuracy_score(y, pred),
                        "sensibilidade": tp / (tp + fn), "especificidade": tn / (tn + fp),
                        "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    summary = pd.DataFrame(summary).sort_values("configuracao")
    stable_csv(summary, out / "resumo_final.csv")
    a = raw[raw.configuracao == "WPT_fixa"].sort_values("participant_id").reset_index(drop=True)
    b = raw[raw.configuracao == "WPT_otimizada"].sort_values("participant_id").reset_index(drop=True)
    assert np.array_equal(a.participant_id, b.participant_id) and np.array_equal(a.y, b.y)
    pair = pd.DataFrame({"y": a.y, "score_fixo": a.score, "score_otimizado": b.score,
                         "limiar_fixo": a.limiar, "limiar_otimizado": b.limiar})
    lo, hi = bootstrap_delta(pair)
    by = summary.set_index("configuracao")
    delta = by.loc["WPT_otimizada", "BA_OOF"] - by.loc["WPT_fixa", "BA_OOF"]
    report = f"""# Experimento 13 — WPT sym5, otimizacao aninhada por participante

**Estado:** concluido em 5 folds externos, cada um com 3 folds internos.\n
- Coorte: 411 participantes HSA (82 normais, 329 patologicos), 2465 gravacoes CAPE-V.
- Unica representacao: WPT sym5 temporal (1920 atributos por gravacao).
- Treino: imputacao, remoção de constantes, SelectKBest, StandardScaler dentro de cada fold.
- Busca interna preespecificada: SVC linear/RBF e LogisticRegression; k=64,128,256,512; C=0.1,1; limiar ajustado apenas por OOF interno.
- Comparacao primaria: WPT otimizada vs WPT fixa (SVC linear, k128, C1, limiar 0) **nas mesmas divisoes externas**.

| Configuracao | BA OOF | AUC OOF | Sens. | Esp. |\n|---|---:|---:|---:|---:|\n"""
    for _, row in summary.iterrows():
        report += f"| {row['configuracao']} | {row['BA_OOF']:.4f} | {row['AUC_OOF']:.4f} | {row['sensibilidade']:.4f} | {row['especificidade']:.4f} |\n"
    report += f"""\nDelta de BA OOF (otimizada - fixa): **{delta:+.4f}**.
IC95% bootstrap estratificado, pareado por participante, condicional a esses folds: **[{lo:+.4f}, {hi:+.4f}]**.

**Limitacoes:** nao ha teste externo independente; bootstrap nao refaz selecao de hiperparametros. AUC concatenada utiliza margens dos modelos ajustados em diferentes folds e nao deve ser confundida com calibração global. A selecao interna nao demonstra validacao clinica. Nao comparar diretamente com o Experimento 12: mudaram as particoes.

Arquivos: `resumo_final.csv`, `metricas_por_fold.csv`, `predicoes_oof_participante.csv`, `candidatos_outer_*.csv`, `predicoes_outer_*.csv` e `assinatura_execucao.json`.\n"""
    (out / "RELATORIO_EXPERIMENTO_13.md").write_text(report, encoding="utf-8")
    say("\n" + report)


def main():
    ap = argparse.ArgumentParser(description="Otimizar apenas WPT sym5 temporal sem tocar nos ZIPs")
    ap.add_argument("--dados", type=Path, default=Path(__file__).resolve().parents[1],
                    help="Raiz AVFAD (default: pai da pasta src)")
    action = ap.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true", help="Valida entradas; sem escrever")
    action.add_argument("--executar", action="store_true", help="Inicia/retoma folds e grava apenas resultados")
    action.add_argument("--autoteste", action="store_true", help="Testa fluxo em dados sinteticos; sem ler AVFAD")
    args = ap.parse_args()
    root = args.dados.resolve()
    X, meta, ident = read_inputs(root, synthetic=args.autoteste)
    p = person_table(meta)
    say(f"Dados: {len(meta)} gravacoes, {len(p)} pessoas, features WPT sym5={X.shape[1]}")
    say(f"Modelos={FAMILIES}; k={TOP_K}; C={C_VALUES}; limiar=OOF interno")
    if args.autoteste:
        # Autoteste pequeno em dados artificiais; nenhuma saida definitiva.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            train, test = next(row_splits(meta, 3, SEED))
            say(f"TESTE SINTETICO: treino={len(train)} teste={len(test)}")
            c, out = evaluate_outer_fold(X, meta, train, test, 1)
            assert len(c) == len(FAMILIES) * len(TOP_K) * len(C_VALUES)
            assert out.groupby("configuracao")["participant_id"].nunique().min() > 0
            assert all(c.BA_interna.between(0, 1))
            say("AUTOTESTE OK; sem arquivos criados no projeto AVFAD")
        return
    if not args.executar:
        say("VERIFICACAO OK. Nenhum resultado novo foi gravado.")
        say("Para executar: .\\.venv\\Scripts\\python.exe .\\src\\13_otimizar_wpt_sym5.py --executar")
        return
    out = root / RESULTS_REL
    sigfile = out / "assinatura_execucao.json"
    if sigfile.exists():
        prev = json.loads(sigfile.read_text(encoding="utf-8"))
        if prev != ident:
            raise RuntimeError("Checkpoint existente usa dados/config diferentes. Nao sobrescrever. Confira assinatura_execucao.json")
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError("Pasta de resultados existe, mas nao possui assinatura; nao sobrescrever")
        stable_json(ident, sigfile)
    outer = list(row_splits(meta, OUTER_FOLDS, SEED))
    for fold, (train, test) in enumerate(outer, 1):
        predfile = out / f"predicoes_outer_{fold:02d}.csv"
        candfile = out / f"candidatos_outer_{fold:02d}.csv"
        if predfile.exists() and candfile.exists():
            existing = pd.read_csv(predfile)
            valid_ids = set(meta.iloc[test]["participant_id"])
            if (set(existing["participant_id"]) != valid_ids or
                set(existing["configuracao"]) != {"WPT_fixa", "WPT_otimizada"} or
                len(existing) != 2 * len(valid_ids)):
                raise RuntimeError(f"Checkpoint fold {fold} falhou na integridade; nao sobrescrever")
            say(f"Fold externo {fold}/{OUTER_FOLDS}: checkpoint confirmado; pulando")
            continue
        if predfile.exists() != candfile.exists():
            raise RuntimeError(f"Fold {fold}: checkpoint parcial, revisar antes de retomar")
        candidates, result = evaluate_outer_fold(X, meta, train, test, fold)
        stable_csv(candidates, candfile)
        stable_csv(result, predfile)
    consolidate(out)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise
