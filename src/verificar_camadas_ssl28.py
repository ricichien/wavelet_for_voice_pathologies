# -*- coding: utf-8 -*-
"""Auditoria 28 após a conclusão; lê parâmetros e dados, nunca SVC.fit.

CLI sem argumentos: auditoria final completa e laudo TXT/JSON novos.
--autoteste: somente dados artificiais. Nenhuma inferência em áudio ou GPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
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
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp28_independent_audit", HERE / "28_avaliar_camadas_ssl_wavelet_avfad.py")
EXP28 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP28)
COHORT_COUNTS = (411, 2465, 82, 329)


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


def independent_splits(meta, n_splits, seed):
    people = meta[["participant_id", "y"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True)
    require(not people.participant_id.duplicated().any(), "Pessoa com rótulos divergentes")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_person, test_person in splitter.split(np.zeros(len(people)), people.y.to_numpy()):
        train_ids = people.iloc[train_person].participant_id
        test_ids = people.iloc[test_person].participant_id
        require(not set(train_ids) & set(test_ids), "Pessoa compartilhada")
        yield np.flatnonzero(meta.participant_id.isin(train_ids)), np.flatnonzero(meta.participant_id.isin(test_ids))


def metrics(frame):
    y, prediction = frame.y.to_numpy(int), frame.predicao.to_numpy(int)
    require(set(np.unique(y)) == {0, 1}, "Métrica exige as duas classes")
    require(np.isfinite(frame.score).all(), "Margens não finitas")
    require(np.array_equal(prediction, (frame.score.to_numpy() >= 0).astype(int)), "Decisão diferente de margem>=0")
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    tp = int(((y == 1) & (prediction == 1)).sum())
    return {"n": len(frame), "BA": .5 * (tn/(tn+fp) + tp/(tp+fn)),
            "acuracia": (tn+tp)/len(frame), "sensibilidade": tp/(tp+fn), "especificidade": tn/(tn+fp),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp, "AUC": float(roc_auc_score(y, frame.score))}


def compare_numbers(actual, expected, tolerance=1e-12):
    for key, value in actual.items():
        require(np.isclose(value, getattr(expected, key), atol=tolerance, rtol=0), f"Métrica incorreta: {key}")


def verify_preprocessor(matrix, y, rows, arrays, prefix, k):
    """Estatísticas calculadas explicitamente nas linhas de treino declaradas."""
    np.testing.assert_array_equal(arrays[prefix+"train_rows"], rows)
    raw = matrix[rows]
    require(np.isfinite(raw).all(), "Insumos completos devem ser finitos")
    np.testing.assert_array_equal(np.median(raw, axis=0), arrays[prefix+"imputer"])
    variable = np.ptp(raw, axis=0) > 0
    np.testing.assert_array_equal(variable, arrays[prefix+"constant_mask"])
    require(k <= int(variable.sum()), "Atributos não constantes insuficientes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = f_classif(raw[:, variable], y[rows])[0]
    np.testing.assert_allclose(scores, arrays[prefix+"selector_scores"], rtol=1e-12, atol=1e-12, equal_nan=True)
    cleaned = np.where(np.isnan(scores), np.finfo(scores.dtype).min, scores)
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(cleaned, kind="mergesort")[-k:]] = True
    np.testing.assert_array_equal(selected, arrays[prefix+"selector_mask"])
    chosen = raw[:, variable][:, selected].astype(np.float64)
    mean, variance = chosen.mean(axis=0), chosen.var(axis=0)
    epsilon = np.finfo(np.float64).eps
    constant = variance <= len(rows)*epsilon*variance + (len(rows)*mean*epsilon)**2
    scale = np.sqrt(variance); scale[constant] = 1.
    np.testing.assert_allclose(mean, arrays[prefix+"scaler_mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scale, arrays[prefix+"scaler_scale"], rtol=1e-12, atol=1e-12)


def transform(matrix, rows, arrays, prefix):
    values = matrix[rows].copy()
    require(np.isfinite(values).all(), "Atributos não finitos durante replay")
    values = values[:, arrays[prefix+"constant_mask"].astype(bool)]
    values = values[:, arrays[prefix+"selector_mask"].astype(bool)]
    # Atribuição in-place reproduz arredondamento float32 do StandardScaler.
    values -= arrays[prefix+"scaler_mean"]
    values /= arrays[prefix+"scaler_scale"]
    return values


def representation(values, rep):
    if rep == "WPT128":
        return values[rep]
    if rep.startswith("SSL128_L"):
        return values[rep]
    if rep.startswith("SSL256_L"):
        return values[rep]/np.sqrt(2.)
    require(rep.startswith("HYB128_L"), "Representação desconhecida no replay")
    ssl = "SSL128_L" + rep.rsplit("_L", 1)[1]
    return np.concatenate([values["WPT128"], values[ssl]], axis=1)/np.sqrt(2.)


def independent_winner(candidates, allowed):
    eligible = candidates[candidates.representation.isin(allowed)]
    require(len(eligible) > 0 and np.isfinite(eligible.inner_BA).all(), "Candidatos internos inválidos")
    near = eligible[eligible.inner_BA >= eligible.inner_BA.max()-1e-12]
    order = {24: 0, 5: 1, 12: 2, 16: 3}
    return min(near.itertuples(), key=lambda row: (row.C, order[row.layer], row.candidate_id))


def paired_interval(raw, seeds, coefficients, iterations=2000):
    ids = sorted(raw.participant_id.unique())
    target = raw.drop_duplicates("participant_id").set_index("participant_id").loc[ids].y.to_numpy()
    contributions = []
    for seed in seeds:
        delta = np.zeros(len(ids), dtype=np.float64)
        for arm, coefficient in coefficients.items():
            rows = raw[(raw.seed == seed) & (raw.arm == arm)].set_index("participant_id")
            require(not rows.index.duplicated().any() and set(rows.index) == set(ids), "Contraste sem OOF completo")
            rows = rows.loc[ids]
            np.testing.assert_array_equal(rows.y, target)
            delta += coefficient*(rows.predicao.to_numpy() == target)
        contributions.append(delta)
    contribution = np.mean(contributions, axis=0)
    normal, pathological = np.flatnonzero(target == 0), np.flatnonzero(target == 1)
    rng = np.random.default_rng(20261105)
    ns = rng.choice(normal, size=(iterations, len(normal)), replace=True)
    ps = rng.choice(pathological, size=(iterations, len(pathological)), replace=True)
    values = .5*(contribution[ns].mean(axis=1)+contribution[ps].mean(axis=1))
    low, high = np.quantile(values, [.025, .975])
    return np.array([.5*(contribution[normal].mean()+contribution[pathological].mean()), low, high])


def compare_people(frame, source, arm, old_arm, seed, fold):
    actual = frame[(frame.arm == arm) & (frame.seed == seed) & (frame.fold == fold)].set_index("participant_id").sort_index()
    expected = source[(source.arm == old_arm) & (source.seed == seed) & (source.fold == fold)].set_index("participant_id").sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_allclose(actual.score, expected.score, atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(actual.predicao, expected.predicao)
    np.testing.assert_array_equal(actual.y, expected.y)


def audit_checkpoint(matrices, meta, names, train, test, seed, fold, frames, arrays, totals, c_grid=None):
    c_grid = EXP28.C_GRID if c_grid is None else c_grid
    EXP28.validate(frames, arrays, meta, train, test, names, seed, fold, c_grid)
    y = meta.y.to_numpy()
    require(not set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id), "Vazamento externo")
    inner, candidates, choices = (frames[key] for key in ("predicoes_oof_internas.csv", "candidatos_internos.csv", "selecoes.csv"))
    totals["inner_OOF"] += len(inner); totals["candidate_summaries"] += len(candidates); totals["selections"] += len(choices)
    candidate_groups = dict(tuple(inner.groupby("candidate_id")))
    for row in candidates.itertuples():
        measured = metrics(candidate_groups[row.candidate_id])
        for key in ("BA", "sensibilidade", "especificidade", "acuracia"):
            require(abs(measured[key]-getattr(row, "inner_"+key)) < 1e-12, "Métrica interna incorreta")
    for selected in choices.itertuples():
        if selected.arm in EXP28.FIXED:
            require(selected.representation == EXP28.FIXED[selected.arm] and selected.C == 1., "Braço fixo alterado")
            require(pd.isna(selected.inner_BA) and selected.choice_source == "fixed_protocol", "Braço fixo indevidamente otimizado")
        else:
            winner = independent_winner(candidates, EXP28.ARM_REPS[selected.arm])
            require(selected.candidate_id == winner.candidate_id and abs(selected.inner_BA-winner.inner_BA) < 1e-12,
                    "Escolha não é vencedora interna conforme desempate")
    for inner_fold, (it, iv) in enumerate(independent_splits(meta.iloc[train].reset_index(drop=True), 3, seed+100+fold), 1):
        rows, val = train[it], train[iv]
        require(not set(rows) & set(test) and not set(val) & set(test), "Teste entrou no fold interno")
        require(not set(meta.iloc[rows].participant_id) & set(meta.iloc[val].participant_id), "Vazamento interno")
        for branch, (modality, k) in EXP28.PREPROCESSORS.items():
            verify_preprocessor(matrices[modality], y, rows, arrays, f"inner{inner_fold}__{branch}__", k)
            totals["preprocessors"] += 1
        totals["inner_folds"] += 1
    for branch, (modality, k) in EXP28.PREPROCESSORS.items():
        verify_preprocessor(matrices[modality], y, train, arrays, f"outer__{branch}__", k)
        totals["preprocessors"] += 1
    # Todos os scores externos; três gravações de treino por braço/fold.
    probes = ((test, "predicoes_teste_gravacao.csv", "test_margins"),
              (train[np.unique(np.linspace(0, len(train)-1, 3, dtype=int))], "predicoes_treino_gravacao.csv", "sampled_train_margins"))
    for rows, filename, count_key in probes:
        values = {branch: transform(matrices[modality], rows, arrays, f"outer__{branch}__")
                  for branch, (modality, _) in EXP28.PREPROCESSORS.items()}
        for selected in choices.itertuples():
            arm, rep = selected.arm, selected.representation
            z = representation(values, rep).astype(np.float64)
            margin = (z @ arrays[f"{arm}__coef"].T).reshape(-1)+arrays[f"{arm}__intercept"][0]
            observed = frames[filename].query("arm == @arm").set_index("sample_id").loc[meta.iloc[rows].sample_id].score.to_numpy()
            error = float(np.max(np.abs(margin-observed)))
            require(error < 1e-7, f"Margem não reproduzida: {arm}; erro={error}")
            totals["max_margin_error"] = max(totals["max_margin_error"], error)
            totals[count_key] += len(margin)
    for selected in choices.itertuples():
        arm = selected.arm
        require(np.array_equal(arrays[f"{arm}__classes"], [0, 1]), "Classes do classificador diferentes")
        class_counts = np.bincount(y[train], minlength=2)
        np.testing.assert_allclose(arrays[f"{arm}__class_weight"], len(train)/(2*class_counts), atol=1e-12, rtol=0)
        support = arrays[f"{arm}__support_indices"].astype(int)
        require(np.all((0 <= support) & (support < len(train))), "Suporte fora do treino")
        np.testing.assert_array_equal(arrays[f"{arm}__support_original_rows"], train[support])
        branches = {branch: transform(matrices[modality], train[support], arrays, f"outer__{branch}__")
                    for branch, (modality, _) in EXP28.PREPROCESSORS.items()}
        np.testing.assert_allclose(representation(branches, selected.representation), arrays[f"{arm}__support_vectors"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(arrays[f"{arm}__dual_coef"] @ arrays[f"{arm}__support_vectors"], arrays[f"{arm}__coef"], atol=1e-10, rtol=1e-10)
        totals["classifiers"] += 1
    sources = {("teste", "participante"): "predicoes.csv", ("treino", "participante"): "predicoes_treino_participante.csv",
               ("teste", "gravacao"): "predicoes_teste_gravacao.csv", ("treino", "gravacao"): "predicoes_treino_gravacao.csv"}
    for row in frames["metricas.csv"].itertuples():
        compare_numbers(metrics(frames[sources[(row.phase, row.unit)]].query("arm == @row.arm")), row)
        totals["metric_rows"] += 1
    totals["outer_folds"] += 1


def blank_totals():
    return {"outer_folds": 0, "inner_folds": 0, "checkpoint_hashes": 0, "inner_OOF": 0,
            "candidate_summaries": 0, "selections": 0, "preprocessors": 0, "classifiers": 0,
            "test_margins": 0, "sampled_train_margins": 0, "max_margin_error": 0., "metric_rows": 0}


def audit(out, data20, data27, report):
    started = time.perf_counter()
    marker_path = out/"ARTEFATOS_COMPLETOS.json"
    require(marker_path.is_file(), "Experimento28 ainda incompleto; nenhum laudo será escrito")
    exports = json.loads(marker_path.read_text(encoding="utf8"))
    required = {"predicoes_oof_participante.csv", "metricas_por_seed.csv", "comparacoes_pareadas.csv",
                "referencia21_oof.csv", "referencia22_oof.csv", "selecoes_internas.csv", "candidatos_internos.csv"}
    require(required.issubset(exports), "Exports finais incompletos")
    for name, digest in exports.items():
        require(Path(name).name == name and sha(out/name) == digest, f"Export final alterado: {name}")
    identity = json.loads((out/"assinatura_execucao.json").read_text(encoding="utf8"))
    for category in ("code", "input_files"):
        for name, digest in identity[category].items():
            require(sha(name) == digest, f"Fonte/insumo assinado alterado: {name}")
    protocol_hash = hashlib.sha256((out/"PROTOCOLO.txt").read_text(encoding="utf8").encode("utf8")).hexdigest()
    require(protocol_hash == identity["protocol_sha256"], "Protocolo alterado")
    config = identity["configuration"]
    seeds = tuple(config["seeds"])
    require(seeds == EXP28.SEEDS and config["arms"] == list(EXP28.ARMS), "Laudo final exige seis sementes/nove braços")
    require(config["C_grid"] == list(EXP28.C_GRID) and config["layers"] == [24, 5, 12, 16], "Grade/camadas diferentes")
    require(config["outer_folds"] == 5 and config["inner_folds"] == 3 and config["threshold"] == 0., "Protocolo de folds/limiar diferente")
    report_json = report.with_suffix(".json")
    if report.exists() or report_json.exists():
        require(report.is_file() and report_json.is_file(), "Laudo parcial existente preservado")
        prior = json.loads(report_json.read_text(encoding="utf8"))
        require(prior["status"] == "aprovado" and prior["auditor_sha256"] == sha(__file__) and
                prior["marker28_sha256"] == sha(marker_path) and prior["report_txt_sha256"] == sha(report),
                "Laudo anterior diferente/alterado, preservado")
        print("AUDITORIA28 ANTERIOR CONFIRMADA; nenhum arquivo alterado", flush=True)
        return
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    matrices, meta, names = EXP28.load_inputs(ROOT, data20, data27)
    n_people, n_records, n_normal, n_pathological = COHORT_COUNTS
    require(len(meta) == n_records and meta.participant_id.nunique() == n_people, "Coorte incorreta")
    require(meta.drop_duplicates("participant_id").y.value_counts().to_dict() == {1: n_pathological, 0: n_normal}, "Classes incorretas")
    raw = read_csv(out/"predicoes_oof_participante.csv")
    summary = read_csv(out/"metricas_por_seed.csv")
    require(len(raw) == n_people*len(EXP28.ARMS)*len(seeds) and len(summary) == len(EXP28.ARMS)*len(seeds), "Contagem OOF/resultados incorreta")
    require(not raw.duplicated(["seed", "arm", "participant_id"]).any(), "OOF externo duplicado")
    source21 = read_csv(ROOT/EXP28.REF21_REL/"predicoes_oof_participante.csv")
    source22 = read_csv(ROOT/EXP28.REF22_REL/"predicoes_oof_participante.csv")
    expected21, expected22 = source21.copy(), source22[source22.arm.eq("B_WAV2VEC_REG")].copy()
    expected21["arm"], expected22["arm"] = "REF21_"+expected21.arm, "REF22_"+expected22.arm
    pd.testing.assert_frame_equal(read_csv(out/"referencia21_oof.csv"), expected21, check_dtype=False)
    pd.testing.assert_frame_equal(read_csv(out/"referencia22_oof.csv"), expected22.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(read_csv(out/"particoes_participantes.csv"), EXP28.EXP19.fold_table(meta, seeds), check_dtype=False)
    totals = blank_totals()
    aggregate = {name: [] for name in ("predicoes.csv", "metricas.csv", "atributos.csv", "candidatos_internos.csv", "selecoes.csv")}
    for seed in seeds:
        for fold, (train, test) in enumerate(independent_splits(meta, 5, seed), 1):
            destination = out/"checkpoints"/f"seed_{seed}_fold_{fold:02d}"
            frames, arrays = EXP28.load_checkpoint(destination, signature)
            totals["checkpoint_hashes"] += len(EXP28.ARTIFACTS)
            audit_checkpoint(matrices, meta, names, train, test, seed, fold, frames, arrays, totals)
            compare_people(frames["predicoes.csv"], source21, EXP28.ARMS[0], "A_WPT", seed, fold)
            compare_people(frames["predicoes.csv"], source21, EXP28.ARMS[1], "B_WAV2VEC", seed, fold)
            compare_people(frames["predicoes.csv"], source22, EXP28.ARMS[5], "B_WAV2VEC_REG", seed, fold)
            for name in aggregate:
                aggregate[name].append(frames[name])
            print(f"Auditoria28: seed={seed}, fold={fold}, nove braços conferidos", flush=True)
    n_folds, n_arms = 5*len(seeds), len(EXP28.ARMS)
    n_candidates = len(EXP28.REPRESENTATIONS)*len(EXP28.C_GRID)
    expected_totals = {"outer_folds": n_folds, "inner_folds": 3*n_folds,
                       "checkpoint_hashes": len(EXP28.ARTIFACTS)*n_folds,
                       "inner_OOF": n_candidates*len(seeds)*4*n_people,
                       "candidate_summaries": n_candidates*n_folds, "selections": n_arms*n_folds,
                       "preprocessors": len(EXP28.PREPROCESSORS)*4*n_folds,
                       "classifiers": n_arms*n_folds, "test_margins": n_records*n_arms*len(seeds),
                       "sampled_train_margins": 3*n_arms*n_folds, "metric_rows": 4*n_arms*n_folds}
    for key, value in expected_totals.items():
        require(totals[key] == value, f"Contagem final incorreta {key}: {totals[key]} != {value}")
    consolidated = {"predicoes.csv": "predicoes_oof_participante.csv", "metricas.csv": "metricas_por_fold.csv",
                    "atributos.csv": "atributos_selecionados.csv", "candidatos_internos.csv": "candidatos_internos.csv",
                    "selecoes.csv": "selecoes_internas.csv"}
    for name, filename in consolidated.items():
        expected = pd.concat(aggregate[name], ignore_index=True)
        pd.testing.assert_frame_equal(read_csv(out/filename), expected, check_dtype=False)
    for row in summary.itertuples():
        group = raw[(raw.seed == row.seed) & (raw.arm == row.arm)]
        require(len(group) == n_people and not group.participant_id.duplicated().any(), "Resumo sem todas as pessoas únicas")
        measured = metrics(group)
        measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        compare_numbers(measured, row)
        aucs, weights = [], []
        for _, fg in group.groupby("fold"):
            aucs.append(float(roc_auc_score(fg.y, fg.score)))
            weights.append(int(fg.y.eq(0).sum()*fg.y.eq(1).sum()))
        require(abs(np.average(aucs, weights=weights)-row.AUC_media_folds_ponderada_pares) < 1e-12, "AUC ponderada incorreta")
    comparisons = read_csv(out/"comparacoes_pareadas.csv")
    n_scopes = int(20261105 in seeds)+int(any(seed != 20261105 for seed in seeds))
    require(len(comparisons) == n_scopes*len(EXP28.CONTRASTS) and not comparisons.duplicated(["escopo", "comparacao"]).any(), "Contrastes incompletos/duplicados")
    ci_error = 0.
    for row in comparisons.itertuples():
        require(row.escopo in {"principal_seed20261105", "estabilidade_secundaria"} and row.comparacao in EXP28.CONTRASTS, "Escopo/contraste inesperado")
        selected_seeds = [seeds[0]] if row.escopo == "principal_seed20261105" else list(seeds[1:])
        expected = paired_interval(raw, selected_seeds, EXP28.CONTRASTS[row.comparacao])
        observed = np.array([row.delta_BA, row.IC95_inferior, row.IC95_superior])
        np.testing.assert_allclose(expected, observed, atol=1e-14, rtol=0)
        ci_error = max(ci_error, float(np.max(np.abs(expected-observed))))
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")
    lines = ["AUDITORIA FINAL DO EXPERIMENTO 28 — APROVADA", f"Verificado em: {now}",
             f"Avaliador: {out}", f"Auditor: {Path(__file__).resolve()}", f"SHA256 auditor: {sha(__file__)}", "",
             "Nenhum classificador foi ajustado. Nenhum áudio foi decodificado e nenhuma GPU foi usada.",
             "Dados, fontes, caches, modelos e resultados foram lidos sem modificação.",
             "Os dois arquivos criados são este TXT e seu JSON, somente após todos os controles passarem.", "",
             "CONTROLES COMPUTACIONAIS", "",
             f"Hashes de todos os exports, fontes, insumos e {totals['checkpoint_hashes']} arquivos dos {n_folds} checkpoints confirmados.",
             "Folds externos/internos reconstruídos por pessoa com StratifiedKFold e sementes fixas.",
             "Medianas, constantes, ANOVA, ranking, médias e escalas recalculados somente nas linhas de treino.",
             "ANOVA usa f_classif como função estatística, sem fit de pipeline ou classificador.",
             "Classes/pesos balanced por gravação, vetores de suporte, coeficientes lineares e escolha de camada/C conferidos.",
             "Todas as margens de teste por gravação e três margens de treino por braço/fold foram reconstruídas.",
             f"As {totals['candidate_summaries']} tabelas de candidatos e {totals['selections']} escolhas foram conferidas pela BA OOF interna e pelos desempates.",
             "As margens dos classificadores internos não foram refeitas: seus coeficientes não são exportados.",
             "A auditoria interna verifica OOFs salvos, partições, transformadores, métricas e seleção; não reexecuta os 5.400 ajustes internos.",
             f"A/B reproduzem21A/B e F reproduz22B nas {3*n_folds} comparações de controle por fold.",
             "Agregações de frases por pessoa, matrizes de confusão, BA, sensibilidade, especificidade, acurácia e AUC conferidas.",
             "Resultados completos por semente e contrastes bootstrap pareados foram recalculados.", ""]
    lines += [f"{key}: {value}" for key, value in totals.items()]
    lines += [f"outer_OOF: {len(raw)}", f"resultados_seed_braco: {len(summary)}",
              f"contrastes_IC_recalculados: {len(comparisons)}", f"erro_maximo_contrastes_IC: {ci_error:.3g}",
              f"segundos_auditoria: {time.perf_counter()-started:.3f}", "",
              "LIMITES", "Aprovação computacional não demonstra validade clínica, generalização externa ou estado da arte.",
              "A AVFAD foi explorada repetidamente; os ICs condicionais não corrigem toda a seleção histórica.",
              "Sementes reutilizam as mesmas pessoas; não representam seis coortes independentes.",
              "Nenhum dado SVD nem novas tarefas vocais foi usado neste experimento.", ""]
    text_bytes = "\n".join(lines).encode("utf8")
    outcome = {"status": "aprovado", "verified_at": now, "auditor_sha256": sha(__file__),
               "marker28_sha256": sha(marker_path), "n_folds": n_folds, "n_arms": n_arms, "n_outer": n_arms*n_folds,
               "totals": totals, "outer_OOF": len(raw), "summary_rows": len(summary),
               "paired_CIs": len(comparisons), "max_CI_error": ci_error,
               "internal_classifier_margins_replayed": False, "all_external_recording_margins_replayed": True,
               "report_txt_sha256": hashlib.sha256(text_bytes).hexdigest(), "report_txt": str(report),
               "model_refits": 0, "audio_decodes": 0, "gpu_used": False}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("xb") as stream:
        stream.write(text_bytes)
    with report_json.open("x", encoding="utf8") as stream:
        json.dump(outcome, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print("AUDITORIA28 APROVADA", json.dumps(totals, sort_keys=True), flush=True)
    print("Laudos:", report, report_json, flush=True)


def selftest():
    import unittest
    spec = importlib.util.spec_from_file_location("test_independent_audit28", HERE/"test_verificar_camadas_ssl28.py")
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", type=Path, default=ROOT/EXP28.OUT_REL)
    parser.add_argument("--dados20", type=Path, default=ROOT/EXP28.EXP21.DATA20_REL)
    parser.add_argument("--dados27", type=Path, default=ROOT/EXP28.DATA27_REL)
    parser.add_argument("--relatorio", type=Path, default=ROOT/"results/experiments/documentacao_wav2vec/28_VERIFICACAO_INDEPENDENTE.txt")
    parser.add_argument("--autoteste", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        selftest(); return
    out, data20, data27, report = (path.resolve() for path in (args.experimento, args.dados20, args.dados27, args.relatorio))
    require(all(path.is_relative_to(ROOT/"results/experiments") for path in (out, data20, data27, report)), "Caminho fora do workspace de experimentos")
    audit(out, data20, data27, report)


if __name__ == "__main__":
    main()
