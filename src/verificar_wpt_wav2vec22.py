# -*- coding: utf-8 -*-
"""Auditoria posterior do22: somente leitura de dados/modelos, sem SVC.fit.

O unico arquivo escrito e o relatorio solicitado, depois de todas as
verificacoes passarem e de a exportacao completa existir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp22_audit", HERE / "22_regularizar_wpt_wav2vec_avfad.py")
EXP22 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP22)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def metrics(frame):
    y, prediction = frame.y.to_numpy(int), frame.predicao.to_numpy(int)
    require(np.isfinite(frame.score).all(), "Margens nao finitas")
    require(np.array_equal(prediction, (frame.score.to_numpy() >= 0).astype(int)), "Decisao difere de margem>=0")
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    tp = int(((y == 1) & (prediction == 1)).sum())
    return {"n": len(frame), "BA": .5 * (tn / (tn + fp) + tp / (tp + fn)),
            "acuracia": (tn + tp) / len(frame), "sensibilidade": tp / (tp + fn),
            "especificidade": tn / (tn + fp), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "AUC": float(roc_auc_score(y, frame.score))}


def compare_numbers(actual, expected, tolerance=1e-12):
    for key, value in actual.items():
        require(np.isclose(value, getattr(expected, key), atol=tolerance, rtol=0), f"Metrica incorreta: {key}")


def verify_preprocessor(matrix, y, rows, arrays, prefix, k):
    """Recalcula estatisticas somente das linhas de treino; nao ajusta modelos."""
    raw = matrix[rows]
    require(np.isfinite(raw).all(), "Esta auditoria pressupoe insumos auditados finitos")
    median = np.median(raw, axis=0)
    np.testing.assert_array_equal(median, arrays[prefix + "imputer"])
    variable = np.ptp(raw, axis=0) > 0
    np.testing.assert_array_equal(variable, arrays[prefix + "constant_mask"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = f_classif(raw[:, variable], y[rows])[0]
    np.testing.assert_allclose(scores, arrays[prefix + "selector_scores"], rtol=1e-12, atol=1e-12, equal_nan=True)
    cleaned = np.where(np.isnan(scores), np.finfo(scores.dtype).min, scores)
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(cleaned, kind="mergesort")[-k:]] = True
    np.testing.assert_array_equal(selected, arrays[prefix + "selector_mask"])
    chosen = raw[:, variable][:, selected].astype(np.float64)
    mean, variance = chosen.mean(axis=0), chosen.var(axis=0)
    epsilon = np.finfo(np.float64).eps
    constant = variance <= len(rows) * epsilon * variance + (len(rows) * mean * epsilon) ** 2
    scale = np.sqrt(variance)
    scale[constant] = 1.
    np.testing.assert_allclose(mean, arrays[prefix + "scaler_mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scale, arrays[prefix + "scaler_scale"], rtol=1e-12, atol=1e-12)


def transform(matrix, rows, arrays, prefix):
    values = matrix[rows].copy()
    statistics = arrays[prefix + "imputer"]
    missing = np.isnan(values)
    if missing.any():
        values[missing] = np.broadcast_to(statistics, values.shape)[missing]
    values = values[:, arrays[prefix + "constant_mask"].astype(bool)]
    values = values[:, arrays[prefix + "selector_mask"].astype(bool)]
    # StandardScaler preserva o dtype float32 e arredonda apos cada operacao.
    values -= arrays[prefix + "scaler_mean"]
    values /= arrays[prefix + "scaler_scale"]
    return values


def representation(values, rep):
    if rep == "WPT128":
        return values["WPT128"]
    if rep == "WAV128":
        return values["WAV128"]
    if rep == "WAV256":
        return values["WAV256"] / np.sqrt(2.)
    if rep == "HYB050":
        return np.concatenate([values["WPT128"], values["WAV128"]], axis=1) / np.sqrt(2.)
    alpha = {"HYB025": .25, "HYB075": .75}[rep]
    return np.concatenate([values["WPT128"] * np.sqrt(alpha), values["WAV128"] * np.sqrt(1. - alpha)], axis=1)


def independent_winner(candidates, allowed):
    eligible = candidates[candidates.representation.isin(allowed)].copy()
    peak = eligible.inner_BA.max()
    eligible = eligible[eligible.inner_BA >= peak - 1e-12]
    return min(eligible.itertuples(), key=lambda row: (row.C, abs(row.alpha - .5), row.alpha, row.candidate_id))


def paired_interval(raw, seeds, coefficients):
    ids = sorted(raw.participant_id.unique())
    target = raw.drop_duplicates("participant_id").set_index("participant_id").loc[ids].y.to_numpy()
    contributions = []
    for seed in seeds:
        delta = np.zeros(len(ids), dtype=np.float64)
        for arm, coefficient in coefficients.items():
            rows = raw[(raw.seed == seed) & (raw.arm == arm)].set_index("participant_id").loc[ids]
            delta += coefficient * (rows.predicao.to_numpy() == target)
        contributions.append(delta)
    contribution = np.mean(contributions, axis=0)
    normal, pathological = np.flatnonzero(target == 0), np.flatnonzero(target == 1)
    rng = np.random.default_rng(20261105)
    normal_sample = rng.choice(normal, size=(2000, len(normal)), replace=True)
    pathological_sample = rng.choice(pathological, size=(2000, len(pathological)), replace=True)
    replicate = .5 * (contribution[normal_sample].mean(axis=1) + contribution[pathological_sample].mean(axis=1))
    low, high = np.quantile(replicate, [.025, .975])
    return np.asarray([.5 * (contribution[normal].mean() + contribution[pathological].mean()), low, high])


def audit(out, report):
    started = time.perf_counter()
    marker_path = out / "ARTEFATOS_COMPLETOS.json"
    require(marker_path.is_file(), "Extracao/avaliacao22 ainda nao concluiu; nenhum relatorio sera escrito")
    exports = json.loads(marker_path.read_text(encoding="utf8"))
    require({"predicoes_oof_participante.csv", "metricas_por_seed.csv", "comparacoes_pareadas.csv", "referencia21_oof.csv"}.issubset(exports), "Exportacoes finais incompletas")
    for name, digest in exports.items():
        require(Path(name).name == name and sha(out / name) == digest, f"Exportacao alterada: {name}")
    identity = json.loads((out / "assinatura_execucao.json").read_text(encoding="utf8"))
    for category in ("code", "input_files"):
        for name, digest in identity[category].items():
            require(sha(name) == digest, f"Insumo/fonte alterado: {name}")
    protocol_hash = hashlib.sha256((out / "PROTOCOLO.txt").read_text(encoding="utf8").encode("utf8")).hexdigest()
    require(protocol_hash == identity["protocol_sha256"], "Protocolo alterado")
    seeds = tuple(identity["configuration"]["seeds"])
    require(seeds == EXP22.SEEDS, "Auditoria final exige ciclo das seis sementes completo")
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    matrices, meta, names = EXP22.EXP21.load_inputs(ROOT, ROOT / EXP22.EXP21.DATA20_REL)
    y = meta.y.to_numpy()
    raw = read_csv(out / "predicoes_oof_participante.csv")
    summary = read_csv(out / "metricas_por_seed.csv")
    require(len(raw) == 12330 and len(summary) == 30, "Contagem externa incorreta")
    require(not raw.duplicated(["seed", "arm", "participant_id"]).any(), "OOF externo duplicado")
    reference = read_csv(out / "referencia21_oof.csv")
    source21 = read_csv(ROOT / EXP22.REF21_REL / "predicoes_oof_participante.csv")
    expected_reference = source21.copy()
    expected_reference["arm"] = "REF21_" + expected_reference.arm
    pd.testing.assert_frame_equal(reference, expected_reference)
    partitions = read_csv(out / "particoes_participantes.csv")
    expected_partitions = EXP22.EXP19.fold_table(meta, seeds)
    pd.testing.assert_frame_equal(partitions, expected_partitions)
    totals = {"outer_folds": 0, "inner_folds": 0, "checkpoint_hashes": 0, "inner_OOF": 0,
              "candidate_summaries": 0, "selections": 0, "preprocessors": 0,
              "reconstructed_margins": 0, "max_margin_error": 0.}
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP22.EXP13.row_splits(meta, 5, seed), 1):
            destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            frames, arrays = EXP22.load_checkpoint(destination, signature)
            EXP22.validate(frames, arrays, meta, train, test, names, seed, fold)
            totals["outer_folds"] += 1
            totals["checkpoint_hashes"] += len(EXP22.ARTIFACTS)
            inner, candidates, choices = (frames[key] for key in ("predicoes_oof_internas.csv", "candidatos_internos.csv", "selecoes.csv"))
            totals["inner_OOF"] += len(inner)
            totals["candidate_summaries"] += len(candidates)
            totals["selections"] += len(choices)
            candidate_groups = dict(tuple(inner.groupby("candidate_id")))
            for row in candidates.itertuples():
                values = metrics(candidate_groups[row.candidate_id])
                require(abs(values["BA"] - row.inner_BA) < 1e-14, "BA interna difere da contagem independente")
            for selected in choices.itertuples():
                winner = independent_winner(candidates, EXP22.ARM_REPS[selected.arm])
                require(selected.candidate_id == winner.candidate_id, "C/alpha nao correspondem ao desempate registrado")
            for inner_fold in (1, 2, 3):
                parts = frames["particoes_internas.csv"]
                rows = parts[(parts.inner_fold == inner_fold) & (parts.role == "treino")].matrix_row.to_numpy(int)
                for branch, (modality, k) in EXP22.PREPROCESSORS.items():
                    verify_preprocessor(matrices[modality], y, rows, arrays, f"inner{inner_fold}__{branch}__", k)
                    totals["preprocessors"] += 1
                totals["inner_folds"] += 1
            for branch, (modality, k) in EXP22.PREPROCESSORS.items():
                verify_preprocessor(matrices[modality], y, train, arrays, f"outer__{branch}__", k)
                totals["preprocessors"] += 1
            chosen_rows = test[np.unique(np.linspace(0, len(test) - 1, 3, dtype=int))]
            values = {branch: transform(matrices[modality], chosen_rows, arrays, f"outer__{branch}__")
                      for branch, (modality, _) in EXP22.PREPROCESSORS.items()}
            for arm in EXP22.ARMS:
                rep = str(arrays[f"{arm}__representation"])
                z = representation(values, rep).astype(np.float64)
                margin = (z @ arrays[f"{arm}__coef"].T).reshape(-1) + arrays[f"{arm}__intercept"][0]
                observed = frames["predicoes_teste_gravacao.csv"].query("arm == @arm").set_index("sample_id").loc[meta.iloc[chosen_rows].sample_id].score.to_numpy()
                error = float(np.max(np.abs(margin - observed)))
                require(error < 1e-7, f"Margem nao reproduzida: {arm}, erro={error}")
                totals["max_margin_error"] = max(totals["max_margin_error"], error)
                totals["reconstructed_margins"] += len(margin)
            for row in frames["metricas.csv"].itertuples():
                filename = {("teste", "participante"): "predicoes.csv", ("treino", "participante"): "predicoes_treino_participante.csv",
                            ("teste", "gravacao"): "predicoes_teste_gravacao.csv", ("treino", "gravacao"): "predicoes_treino_gravacao.csv"}[row.phase, row.unit]
                compare_numbers(metrics(frames[filename].query("arm == @row.arm")), row)
            local = raw[(raw.seed == seed) & (raw.fold == fold)].reset_index(drop=True)
            pd.testing.assert_frame_equal(local, frames["predicoes.csv"], check_dtype=False)
    require(totals["inner_OOF"] == 295920, "Numero OOF interno incorreto")
    require(totals["candidate_summaries"] == 900 and totals["selections"] == 150, "Grade/buscas incompletas")
    require(totals["checkpoint_hashes"] == 330 and totals["preprocessors"] == 360, "Hashes/transformadores incompletos")
    require(totals["reconstructed_margins"] == 450, "Reconstrucao de margens incompleta")
    for row in summary.itertuples():
        group = raw[(raw.seed == row.seed) & (raw.arm == row.arm)]
        measured = metrics(group)
        measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        compare_numbers(measured, row)
        aucs, weights = [], []
        for _, fold_group in group.groupby("fold"):
            aucs.append(float(roc_auc_score(fold_group.y, fold_group.score)))
            weights.append(int((fold_group.y == 0).sum() * (fold_group.y == 1).sum()))
        require(abs(np.average(aucs, weights=weights) - row.AUC_media_folds_ponderada_pares) < 1e-14,
                "AUC ponderada dos folds incorreta")
    comparisons_raw = pd.concat([raw, reference], ignore_index=True)
    comparisons = read_csv(out / "comparacoes_pareadas.csv")
    require(len(comparisons) == 2 * len(EXP22.CONTRASTS), "Contrastes incompletos")
    ci_error = 0.
    for row in comparisons.itertuples():
        selected_seeds = [seeds[0]] if row.escopo == "principal_seed20261105" else list(seeds[1:])
        expected = paired_interval(comparisons_raw, selected_seeds, EXP22.CONTRASTS[row.comparacao])
        observed = np.asarray([row.delta_BA, row.IC95_inferior, row.IC95_superior])
        ci_error = max(ci_error, float(np.max(np.abs(expected - observed))))
        np.testing.assert_allclose(expected, observed, atol=1e-14, rtol=0)
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")
    lines = ["AUDITORIA FINAL DO EXPERIMENTO 22 - APROVADA", f"Verificado em: {now}",
             f"Avaliador auditado: {out}", f"Auditor: {Path(__file__).resolve()}", f"SHA256 auditor: {sha(__file__)}", "",
             "Nenhum SVM ou modelo foi reajustado. Dados, fontes e resultados lidos sem alteracao.",
             "O unico arquivo criado nesta chamada e este relatorio posterior a conclusao.",
             "Estatisticas de mediana, variabilidade, ANOVA, ranking, media e escala foram",
             "recalculadas das linhas de treino de cada particao, sem fit de estimadores.",
             "O calculo F utiliza f_classif; a verificacao de isolamento e baseada nas linhas",
             "de treino reais. Escala/mediana/ranking/margens e metricas recalculados explicitamente.", ""]
    lines += [f"{key}: {value}" for key, value in totals.items()]
    lines += [f"outer_OOF: {len(raw)}", f"resultados_seed_braco: {len(summary)}", f"contrastes_IC_recalculados: {len(comparisons)}",
              f"erro_maximo_contrastes_IC: {ci_error:.3g}", f"segundos_auditoria: {time.perf_counter()-started:.3f}", "",
              "Exportacoes finais, fontes e insumos: hashes SHA256 confirmados.",
              "Particoes externas/internas: nenhuma pessoa compartilhada em treino/validacao/teste.",
              "900 candidatos e 150 escolhas confirmados pela BA OOF interna e desempates registrados.",
              "Referencia21 alinhada e identica ao arquivo original; comparacoes pareadas por pessoa.",
              "Bootstrap estratificado com 2000 repeticoes, mantendo seeds da mesma pessoa no mesmo cluster.", "",
              "LIMITES", "Auditoria computacional nao estabelece validade clinica ou teste externo independente.",
              "AVFAD repetidamente explorada; ICs condicionais nao refazem selecao/treino nem corrigem tentativas historicas.",
              "Nenhum dado SVD foi usado. Interpretar resultados pelo contraste primario e controles, sem escolher melhor seed.", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    require(not report.exists(), f"Relatorio existente preservado: {report}")
    report.write_bytes("\n".join(lines).encode("utf8"))
    print("AUDITORIA22 APROVADA", json.dumps(totals, sort_keys=True), flush=True)
    print("Relatorio:", report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", type=Path, default=ROOT / EXP22.OUT_REL)
    parser.add_argument("--relatorio", type=Path, default=ROOT / "results/experiments/documentacao_wav2vec/22_VERIFICACAO_INDEPENDENTE.txt")
    args = parser.parse_args()
    out, report = args.experimento.resolve(), args.relatorio.resolve()
    require(out.is_relative_to(ROOT / "results/experiments"), "Experimento fora do projeto")
    require(report.is_relative_to(ROOT / "results/experiments"), "Relatorio fora do projeto")
    audit(out, report)


if __name__ == "__main__":
    main()
