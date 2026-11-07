# -*- coding: utf-8 -*-
"""Exp15: WPT compacta, ranking por pessoa e CV aninhada sem consultar teste."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from wavelet_participant_selection import ParticipantMeanANOVA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp13_for_exp15", HERE / "13_otimizar_wpt_sym5.py")
EXP13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP13)
ARMS = ("A_REFERENCIA", "D_ANOVA_GRAVACAO", "E_ANOVA_PESSOA")
DEFAULT_SEEDS = (20261105, 101, 102, 103, 104, 105)
GRID = tuple((rep, k, c) for rep in ("L5", "L6", "L5_L6")
             for k in (32, 64, 128) for c in (0.1, 1.0))
VERSION = "15.1"


def stamp():
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def csv_write(path, frame):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def txt_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, message):
        line = f"[{stamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def prepare(root, seeds, protocol):
    x, meta, _ = EXP13.read_inputs(root)
    with np.load(root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz", allow_pickle=False) as z:
        names = z["feature_names"].astype(str)
        names = names[np.char.startswith(names, "WPT__sym5__")]
    masks = {"L5": np.flatnonzero(np.char.find(names, "__L5__") >= 0),
             "L6": np.flatnonzero(np.char.find(names, "__L6__") >= 0),
             "L5_L6": np.arange(len(names))}
    if {key: len(val) for key, val in masks.items()} != {"L5": 640, "L6": 1280, "L5_L6": 1920}:
        raise ValueError("Dimensoes WPT fora do protocolo")
    files = [root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz",
             root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv",
             Path(__file__), HERE / "13_otimizar_wpt_sym5.py",
             HERE / "wavelet_participant_selection.py", protocol]
    environment = {"python": sys.version, "executable": sys.executable,
                   "platform": platform.platform(), "numpy": np.__version__,
                   "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    identity = {"version": VERSION, "files": {str(p.resolve()): sha(p) for p in files},
                "environment": environment, "seeds": list(seeds), "outer_folds": 5,
                "inner_folds": 3, "grid": GRID, "arms": ARMS,
                "selection": "BA_interna_zero; k,C,dimensao,nome em empates", "bootstrap": 2000}
    # JSON roundtrip normaliza tuplas para comparar com assinaturas persistidas.
    identity = json.loads(json.dumps(identity, sort_keys=True))
    return x, meta, names, masks, identity


def make_model(kind, k, c):
    model = EXP13.pipeline_for("SVC_linear", k, c)
    if kind == "E_ANOVA_PESSOA":
        model.set_params(select=ParticipantMeanANOVA(k=k))
    return model


def fit_model(model, x, meta, kind):
    params = {"select__groups": meta.participant_id.to_numpy()} if kind == "E_ANOVA_PESSOA" else {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(x, meta.y.to_numpy(), **params)
    return model


def part(meta, raw):
    result = EXP13.aggregate(meta.reset_index(drop=True), raw)
    result["predicao"] = (result.score >= 0).astype(int)
    return result


def saved_selection(model, names):
    keep = model.named_steps["remove_constants"].get_support()
    selected = model.named_steps["select"].get_support()
    return np.asarray(names)[keep][selected]


def validate_predictions(frame, expected_ids, expected_y, seed, fold):
    if set(frame.arm) != set(ARMS) or set(frame.seed) != {seed} or set(frame.fold) != {fold}:
        raise ValueError("Bracos/seed/fold invalidos")
    for arm, sub in frame.groupby("arm"):
        if len(sub) != len(expected_ids) or sub.participant_id.duplicated().any() or set(sub.participant_id) != expected_ids:
            raise ValueError(f"OOF inesperado: {arm}")
        labels = sub.participant_id.map(expected_y).to_numpy()
        if not np.array_equal(labels, sub.y.to_numpy()) or not np.isfinite(sub.score).all():
            raise ValueError("Rotulos/scores invalidos")
        if not np.array_equal((sub.score.to_numpy() >= 0).astype(int), sub.predicao.to_numpy()):
            raise ValueError("Predicao inconsistente")


def fold_manifest(meta, seeds):
    rows = []
    for seed in seeds:
        for fold, (tr, te) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            train = meta.iloc[tr].reset_index(drop=True)
            a = set(train.participant_id)
            b = set(meta.iloc[te].participant_id)
            if a & b:
                raise ValueError("Sobreposicao externa")
            for role, ids in (("treino", a), ("teste", b)):
                rows.extend({"seed": seed, "outer_fold": fold, "level": "outer", "inner_fold": 0,
                             "role": role, "participant_id": pid} for pid in sorted(ids))
            for inner, (itr, ite) in enumerate(EXP13.row_splits(train, 3, seed + 100 + fold), 1):
                a = set(train.iloc[itr].participant_id)
                b = set(train.iloc[ite].participant_id)
                if a & b:
                    raise ValueError("Sobreposicao interna")
                for role, ids in (("treino", a), ("validacao", b)):
                    rows.extend({"seed": seed, "outer_fold": fold, "level": "inner", "inner_fold": inner,
                                 "role": role, "participant_id": pid} for pid in sorted(ids))
    return pd.DataFrame(rows)


def evaluate_fold(x, meta, names, masks, seed, fold, tr, te, log, grid=GRID, baseline=None):
    train = meta.iloc[tr].reset_index(drop=True)
    test = meta.iloc[te].reset_index(drop=True)
    if set(train.participant_id) & set(test.participant_id):
        raise ValueError("Vazamento externo")
    inner = list(EXP13.row_splits(train, 3, seed + 100 + fold))
    candidates, inner_rows, choices = [], [], {}
    for arm in ARMS[1:]:
        local = []
        for rep, k, c in grid:
            cols = masks[rep]
            predictions = []
            for inner_fold, (itr, ite) in enumerate(inner, 1):
                fit_meta = train.iloc[itr].reset_index(drop=True)
                val_meta = train.iloc[ite].reset_index(drop=True)
                if set(fit_meta.participant_id) & set(val_meta.participant_id):
                    raise ValueError("Vazamento interno")
                model = fit_model(make_model(arm, k, c), x[tr[itr]][:, cols], fit_meta, arm)
                output = part(val_meta, model.decision_function(x[tr[ite]][:, cols]))
                output["inner_fold"] = inner_fold
                predictions.append(output)
            oof = pd.concat(predictions, ignore_index=True)
            if oof.participant_id.duplicated().any() or set(oof.participant_id) != set(train.participant_id):
                raise ValueError("OOF interno incompleto")
            ba = balanced_accuracy_score(oof.y, oof.predicao)
            record = {"arm": arm, "representation": rep, "k": k, "C": c,
                      "input_features": len(cols), "BA_inner": float(ba), "seed": seed, "fold": fold}
            local.append(record)
            for key, value in record.items():
                oof[key] = value
            inner_rows.append(oof)
            log(f"seed={seed} fold={fold} {arm} {rep} k={k} C={c:g} BA_inner={ba:.4f}")
        ranked = sorted(local, key=lambda r: (-r["BA_inner"], r["k"], r["C"], r["input_features"], r["representation"]))
        choices[arm] = ranked[0]
        candidates.extend(local)
    choices[ARMS[0]] = {"representation": "L5_L6", "k": 128, "C": 1.0}
    outputs, features = [], []
    for arm in ARMS:
        choice = choices[arm]
        cols = masks[choice["representation"]]
        model = baseline if arm == ARMS[0] and baseline is not None else fit_model(
            make_model(arm, choice["k"], choice["C"]), x[tr][:, cols], train, arm)
        output = part(test, model.decision_function(x[te][:, cols]))
        output["arm"] = arm
        for key in ("representation", "k", "C"):
            output[key] = choice[key]
        output["seed"], output["fold"], output["limiar"] = seed, fold, 0.0
        outputs.append(output)
        features.extend({"seed": seed, "fold": fold, "arm": arm, "feature": name}
                        for name in saved_selection(model, names[cols]))
        log(f"RESULTADO seed={seed} fold={fold} {arm} BA={balanced_accuracy_score(output.y, output.predicao):.4f}")
    result = pd.concat(outputs, ignore_index=True)
    validate_predictions(result, set(test.participant_id), meta.drop_duplicates("participant_id").set_index("participant_id").y,
                         seed, fold)
    return {"predicoes.csv": result, "candidatos.csv": pd.DataFrame(candidates),
            "oof_interno.csv": pd.concat(inner_rows, ignore_index=True),
            "atributos.csv": pd.DataFrame(features)}


def metric(frame):
    tn, fp, fn, tp = confusion_matrix(frame.y, frame.predicao, labels=[0, 1]).ravel()
    return {"n_participantes": len(frame), "BA": (tn / (tn + fp) + tp / (tp + fn)) / 2,
            "sensibilidade": tp / (tp + fn), "especificidade": tn / (tn + fp),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "AUC": roc_auc_score(frame.y, frame.score)}


def bootstrap_pairs(pred, seeds, a, b, iterations=2000):
    # BA = media das taxas de acerto por classe. Reamostrar o mesmo ID em todas
    # as repeticoes produz IC da media de BA, preservando dependencia pessoa-seed.
    group = pred[pred.seed.isin(seeds)]
    ids = sorted(group.participant_id.unique())
    labels = group.drop_duplicates("participant_id").set_index("participant_id").loc[ids].y.to_numpy()
    differences = []
    for seed in seeds:
        left = group[(group.seed == seed) & (group.arm == a)].set_index("participant_id").loc[ids]
        right = group[(group.seed == seed) & (group.arm == b)].set_index("participant_id").loc[ids]
        differences.append((left.predicao.to_numpy() == labels).astype(float)
                           - (right.predicao.to_numpy() == labels).astype(float))
    delta = np.mean(differences, axis=0)
    i0, i1 = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    rng = np.random.default_rng(15052026)
    sampled = (delta[rng.choice(i0, (iterations, len(i0)), replace=True)].mean(axis=1)
               + delta[rng.choice(i1, (iterations, len(i1)), replace=True)].mean(axis=1)) / 2
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {"delta_BA": float((delta[i0].mean() + delta[i1].mean()) / 2),
            "IC95_low": float(low), "IC95_high": float(high)}


def check_reference(output, prior, seed, fold):
    if prior is None or seed != 20261105:
        return
    expected = prior[prior.outer_fold == fold].set_index("participant_id").sort_index()
    current = output.set_index("participant_id").sort_index()
    pd.testing.assert_index_equal(expected.index, current.index)
    for col in ("y", "predicao", "n_gravacoes"):
        np.testing.assert_array_equal(expected[col], current[col])
    np.testing.assert_allclose(expected.score, current.score, rtol=1e-9, atol=1e-9)


def consolidate(out, meta, seeds, log, root):
    directories = [out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" for seed in seeds for fold in range(1, 6)]
    pred = pd.concat([pd.read_csv(p / "predicoes.csv") for p in directories], ignore_index=True)
    selection = pd.concat([pd.read_csv(p / "atributos.csv") for p in directories], ignore_index=True)
    expected_ids = set(meta.participant_id)
    for (seed, arm), sub in pred.groupby(["seed", "arm"]):
        if len(sub) != len(expected_ids) or sub.participant_id.duplicated().any() or set(sub.participant_id) != expected_ids:
            raise ValueError(f"OOF completo invalido: {seed} {arm}")
    folds = []
    for (seed, arm, fold), sub in pred.groupby(["seed", "arm", "fold"]):
        folds.append({"seed": int(seed), "arm": arm, "fold": int(fold), **metric(sub)})
    fm = pd.DataFrame(folds)
    summaries = []
    for (seed, arm), sub in pred.groupby(["seed", "arm"]):
        row = metric(sub)
        row["AUC_OOF_exploratoria"] = row.pop("AUC")
        sf = fm[(fm.seed == seed) & (fm.arm == arm)]
        row["AUC_media_por_fold"] = np.average(sf.AUC, weights=(sf.TN + sf.FP) * (sf.TP + sf.FN))
        summaries.append({"seed": int(seed), "arm": arm, **row})
    summary = pd.DataFrame(summaries)
    comparisons = []
    sets = [("primaria_seed20261105", [20261105])] if 20261105 in seeds else []
    secondary = [s for s in seeds if s != 20261105]
    if secondary:
        sets.append(("estabilidade_secundaria", secondary))
    for label, selected_seeds in sets:
        for a, b, designation in ((ARMS[2], ARMS[1], "primaria_E-D"), (ARMS[2], ARMS[0], "secundaria_E-A"),
                                  (ARMS[1], ARMS[0], "secundaria_D-A")):
            comparisons.append({"conjunto": label, "comparacao": designation, "seeds": str(selected_seeds),
                                **bootstrap_pairs(pred, selected_seeds, a, b)})
    csv_write(out / "predicoes_oof_participante.csv", pred)
    csv_write(out / "metricas_por_fold.csv", fm)
    csv_write(out / "metricas_por_seed.csv", summary)
    csv_write(out / "comparacoes_pareadas.csv", pd.DataFrame(comparisons))
    csv_write(out / "atributos_selecionados.csv", selection)
    # Estabilidade das selecoes externas: folds sobrepostos, descritiva somente.
    stability = []
    for arm in ARMS:
        groups = [set(s.feature) for _, s in selection[selection.arm == arm].groupby(["seed", "fold"])]
        jaccard = [len(a & b) / len(a | b) for i, a in enumerate(groups) for b in groups[i + 1:] if a | b]
        stability.append({"arm": arm, "jaccard_medio_descritivo": float(np.mean(jaccard)) if jaccard else None,
                          "n_modelos_externos": len(groups)})
    csv_write(out / "estabilidade_selecao.csv", pd.DataFrame(stability))
    old = root / "results/experiments/resultados_13_otimizacao_wpt_sym5/predicoes_oof_participante.csv"
    reproduction = "Exp13 nao comparado (seed primaria ausente ou arquivo nao localizado)."
    if 20261105 in seeds and old.exists():
        prior = pd.read_csv(old)
        prior = prior[prior.configuracao == "WPT_fixa"].set_index("participant_id").sort_index()
        current = pred[(pred.seed == 20261105) & (pred.arm == ARMS[0])].set_index("participant_id").sort_index()
        pd.testing.assert_index_equal(prior.index, current.index)
        for col in ("y", "predicao"):
            np.testing.assert_array_equal(prior[col], current[col])
        np.testing.assert_allclose(prior.score, current.score, rtol=1e-9, atol=1e-9)
        np.testing.assert_array_equal(prior.outer_fold, current.fold)
        reproduction = "Controle A reproduziu IDs, folds, rotulos, decisoes e scores do Exp13 (tolerancia 1e-9)."
    report = ["EXPERIMENTO 15 - RESULTADOS DE DESENVOLVIMENTO", f"Consolidado: {stamp()}", "",
              "Estado: todos os folds das sementes solicitadas foram concluidos.",
              "Protocolo de seis seeds completo." if list(seeds) == list(DEFAULT_SEEDS)
              else "Configuracao parcial/diferente do protocolo de seis seeds; nao chamar de ciclo completo.",
              reproduction, "",
              "BA por participante. Uma predicao por pessoa e repeticao.",
              "AUC_media_por_fold pondera pares normal-patologico; OOF concatenada e exploratoria.", ""]
    for row in summary.itertuples():
        report.append(f"seed={row.seed} {row.arm}: BA={row.BA:.6f}; sens={row.sensibilidade:.6f}; "
                      f"esp={row.especificidade:.6f}; AUC_fold={row.AUC_media_por_fold:.6f}; "
                      f"TN/FP/FN/TP={row.TN}/{row.FP}/{row.FN}/{row.TP}")
    report += ["", "COMPARACOES PAREADAS (delta primeiro braco menos segundo)"]
    for row in comparisons:
        report.append(f"{row['conjunto']} {row['comparacao']}: deltaBA={row['delta_BA']:+.6f}; "
                      f"IC95 condicional=[{row['IC95_low']:+.6f}, {row['IC95_high']:+.6f}]")
    if secondary:
        report += ["", "ESTABILIDADE SECUNDARIA (mesmas pessoas; DP nao e IC)"]
        for arm, group in summary[summary.seed.isin(secondary)].groupby("arm"):
            report.append(f"{arm}: BA media={group.BA.mean():.6f}; min={group.BA.min():.6f}; "
                          f"max={group.BA.max():.6f}; DP={group.BA.std():.6f}")
    report += ["", "LIMITACOES E LEITURA", "AVFAD reutilizada: desenvolvimento exploratorio, sem teste externo independente.",
               "Bootstrap agrupa pessoa, mantem suas repeticoes e nao refaz selecao nem treinamento.",
               "Nao escolher melhor seed/braco externo para anunciar estimativa imparcial.",
               "E-D avalia politica de ranking dentro de buscas equivalentes, nao transformadas diferentes.",
               "ANOVA por pessoa altera apenas ranking; escala/classificador continuam por gravacao.",
               "As mesmas frases e dados Wavelet sao mantidos. Nenhum audio foi removido ou reextraido.",
               "ANOVA pessoa usa medias float64 para ranking; transform preserva dtype original.",
               "Empates de atributos: pessoa menor indice; SelectKBest padrao para gravacao.",
               "Jaccard de selecao e descritivo: treinos externos se sobrepoem.",
               "Ver PROTOCOLO.txt, assinatura_execucao.json, particoes_participantes.csv e checkpoints.",
               "Este relatorio nao constitui conclusao final do TG.\n"]
    txt_write(out / "RELATORIO_RESULTADOS.txt", "\n".join(report))
    log(reproduction)
    log("Consolidacao concluida; relatorio TXT e tabelas CSV gravados.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path, default=ROOT / "results/experiments/resultados_15_wpt_compacta_participante")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true")
    action.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) != len(args.seeds) or any(s < 0 for s in seeds):
        parser.error("Seeds devem ser distintas e nao negativas")
    out = args.saida.resolve()
    protocol = out / "PROTOCOLO.txt"
    if not protocol.exists():
        raise FileNotFoundError("Elaborar PROTOCOLO.txt antes de executar (nao copiar resultados para protocolo).")
    root = args.dados.resolve()
    x, meta, names, masks, identity = prepare(root, seeds, protocol)
    reference_path = root / "results/experiments/resultados_13_otimizacao_wpt_sym5/predicoes_oof_participante.csv"
    reference = None
    if reference_path.exists():
        reference = pd.read_csv(reference_path)
        reference = reference[reference.configuracao == "WPT_fixa"].copy()
        identity["reference_exp13_sha256"] = sha(reference_path)
    manifest = fold_manifest(meta, seeds)
    print(f"Verificado: X={x.shape}, pessoas={meta.participant_id.nunique()}, seeds={seeds}; folds disjuntos.", flush=True)
    if not args.executar:
        print("Nenhum treino realizado; nenhum arquivo gravado.", flush=True)
        return
    sig = out / "assinatura_execucao.json"
    if sig.exists():
        if json.loads(sig.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Identidade diverge; nao reutilizar resultados. Usar pasta/protocolo novo.")
    else:
        allowed = {"PROTOCOLO.txt"}
        if any(p.name not in allowed for p in out.iterdir()):
            raise RuntimeError("Pasta sem assinatura contem outros arquivos; revisar antes de continuar")
        json_write(sig, identity)
    log = Logger(out / "LOG_EXECUCAO.txt")
    log(f"Inicio/retomada {VERSION}; seeds={seeds}; sem novas extracoes/dependencias.")
    run_signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    csv_write(out / "particoes_participantes.csv", manifest)
    expected_y = meta.drop_duplicates("participant_id").set_index("participant_id").y
    (out / "checkpoints").mkdir(exist_ok=True)
    for seed in seeds:
        for fold, (tr, te) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            marker = destination / "CONCLUIDO.json"
            if marker.exists():
                prior = json.loads(marker.read_text(encoding="utf-8"))
                if prior["run_signature"] != run_signature:
                    raise RuntimeError("Checkpoint pertence a outra assinatura")
                for filename, digest in prior["sha256"].items():
                    if sha(destination / filename) != digest:
                        raise RuntimeError(f"Checkpoint alterado: {destination / filename}")
                if set(prior["sha256"]) != {"predicoes.csv", "candidatos.csv", "oof_interno.csv", "atributos.csv"}:
                    raise RuntimeError("Checkpoint incompleto")
                saved_predictions = pd.read_csv(destination / "predicoes.csv")
                validate_predictions(saved_predictions, set(meta.iloc[te].participant_id), expected_y, seed, fold)
                check_reference(saved_predictions[saved_predictions.arm == ARMS[0]], reference, seed, fold)
                log(f"seed={seed} fold={fold} checkpoint integro; treino dispensado.")
                continue
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Checkpoint parcial em {destination}; preservar e revisar")
            destination.mkdir(exist_ok=True)
            started = time.monotonic()
            baseline = fit_model(make_model(ARMS[0], 128, 1.0), x[tr], meta.iloc[tr], ARMS[0])
            baseline_prediction = part(meta.iloc[te], baseline.decision_function(x[te]))
            check_reference(baseline_prediction, reference, seed, fold)
            if reference is not None and seed == 20261105:
                log(f"seed={seed} fold={fold} controle A reproduziu Exp13 antes da busca.")
            frames = evaluate_fold(x, meta, names, masks, seed, fold, tr, te, log, baseline=baseline)
            for filename, frame in frames.items():
                csv_write(destination / filename, frame)
            json_write(marker, {"finished": stamp(), "elapsed_seconds": time.monotonic() - started,
                                "run_signature": run_signature,
                                "sha256": {filename: sha(destination / filename) for filename in frames}})
            log(f"seed={seed} fold={fold} concluido em {time.monotonic()-started:.1f}s")
    consolidate(out, meta, seeds, log, root)


if __name__ == "__main__":
    main()
