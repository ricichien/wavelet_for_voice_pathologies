# -*- coding: utf-8 -*-
"""Exp28: camadas congeladas XLS-R e contribuição Wavelet em CV aninhada.

--verificar só lê; --autoteste usa dados artificiais; --executar é explícito.
Fontes 13–27, dados, caches e resultados antigos permanecem somente leitura.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp22_for_exp28", HERE / "22_regularizar_wpt_wav2vec_avfad.py")
EXP22 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXP22)
EXP21, EXP19, EXP17, EXP13 = EXP22.EXP21, EXP22.EXP19, EXP22.EXP17, EXP22.EXP13
VERSION = "28.1"
SEEDS = EXP21.SEEDS
LAYERS = (24, 5, 12, 16)
C_GRID = EXP22.C_GRID
ARMS = ("A_WPT_FIXED128", "B_L24_FIXED128", "C_L05_FIXED128", "D_L12_FIXED128", "E_L16_FIXED128",
        "F_L24_C128_REG", "G_LAYER_C128_REG", "H_HYBRID_LAYER_C128", "I_LAYER_C256_REG")
OUT_REL = Path("results/experiments/resultados_28_camadas_ssl_wavelet")
DATA27_REL = Path("results/experiments/resultados_27_camadas_ssl_extracao")
REF21_REL = EXP22.REF21_REL
REF22_REL = EXP22.OUT_REL
CSV_FILES, ARTIFACTS = EXP22.CSV_FILES, EXP22.ARTIFACTS
save_checkpoint, load_checkpoint = EXP22.save_checkpoint, EXP22.load_checkpoint
fit_classifier, pre_parameters = EXP22.fit_classifier, EXP22.pre_parameters
PREPROCESSORS = {"WPT128": ("wpt", 128)}
for _layer in LAYERS:
    PREPROCESSORS[f"SSL128_L{_layer:02d}"] = (f"ssl{_layer:02d}", 128)
    PREPROCESSORS[f"SSL256_L{_layer:02d}"] = (f"ssl{_layer:02d}", 256)
REPRESENTATIONS = tuple(f"{kind}_L{layer:02d}" for layer in LAYERS for kind in ("SSL128", "HYB128", "SSL256"))
FIXED = dict(zip(ARMS[:5], ("WPT128", "SSL128_L24", "SSL128_L05", "SSL128_L12", "SSL128_L16")))
ARM_REPS = {ARMS[5]: ("SSL128_L24",),
            ARMS[6]: tuple(f"SSL128_L{layer:02d}" for layer in LAYERS),
            ARMS[7]: tuple(f"HYB128_L{layer:02d}" for layer in LAYERS),
            ARMS[8]: tuple(f"SSL256_L{layer:02d}" for layer in LAYERS)}
CONTRASTS = {
    "H-G_complementaridade_Wavelet_primaria": {ARMS[7]: 1, ARMS[6]: -1},
    "G-F_camadas_vs_ultima_primaria": {ARMS[6]: 1, ARMS[5]: -1},
    "H-I_controle_orcamento_secundaria": {ARMS[7]: 1, ARMS[8]: -1},
    "E-B_camada16_fixa_secundaria": {ARMS[4]: 1, ARMS[1]: -1},
    "C-B_camada05_fixa_secundaria": {ARMS[2]: 1, ARMS[1]: -1},
    "D-B_camada12_fixa_secundaria": {ARMS[3]: 1, ARMS[1]: -1},
    "G-B_busca_vs_fixo_secundaria": {ARMS[6]: 1, ARMS[1]: -1},
    "H-A_hibrido_vs_Wavelet_secundaria": {ARMS[7]: 1, ARMS[0]: -1},
}


def protocol():
    return """EXPERIMENTO 28 — CAMADAS XLS-R CONGELADAS E HÍBRIDO WAVELET
Hipóteses: camadas intermediárias podem superar a última; Wavelet pode
acrescentar informação após seleção interna da camada e da regularização.
Coorte HSA preservada:411 pessoas,82 normais/329 patológicas,2465 frases.
Sem exclusões, novas tarefas, SVD ou adaptação do encoder. WPT RAW48k sym5
L5/L6,1920 atributos; SSL congelado2048 média/DP: camadas24,5,12,16.
Camada24 vem do20; novas camadas27 usam o mesmo leitor/modelo/preprocessamento.
Controle27 da24 deve reproduzir todos os2465 vetores20 antes dos ajustes.
Mesmos cinco folds externos por participante13/17/21/22, seis sementes:
20261105 principal e101..105 estabilidade. Todas as frases da pessoa juntas.
Dentro de cada treino externo: três folds por pessoa,seed=outerseed+100+fold.
Imputação mediana, constantes, ANOVA e StandardScaler ajustados SÓ no treino
interno durante busca; novamente no treino externo completo para o refit.
SVC linear, class_weight=balanced calculado nas gravações de treino,
limiar0; decisão da pessoa=média das margens de suas frases. Sem SMOTE.
A_WPT_FIXED128:128 WPT,C1, controle21A.
B_L24_FIXED128:128 SSL24,C1, controle21B.
C_L05_FIXED128/D_L12_FIXED128/E_L16_FIXED128:128 da camada fixa,C1.
F_L24_C128_REG:128 SSL24,C em{0.0001,0.001,0.01,0.1,1}, controle22B.
G_LAYER_C128_REG:128 SSL, escolhe camada24/5/12/16 e C internamente.
H_HYBRID_LAYER_C128:128 WPT+128 SSL, concatenação/sqrt2, camada/C internos.
I_LAYER_C256_REG:256 SSL/sqrt2, camada/C internos, controle de orçamento.
Camada/C do híbrido e do SSL são escolhidos em buscas próprias comparáveis;
H-G compara pipelines completos, não efeito causal de uma camada idêntica.
Critério: BA nas predições OOF internas de TODOS os participantes de treino,
obtidas pela média de margens de frases; não média de BA de folds/gravações.
Empates<=1e-12: menor C, depois ordem de camadas24,5,12,16.
60 candidatos únicos:4 camadas×3 representações×5 C, compartilhados somente
na mesma partição de treino. F reutiliza candidatos24 de G; sem reajuste.
Ciclo:5400 classificadores internos+270 externos; A–E são fixos por protocolo.
Nenhuma seleção de época, limiar, camada ou braço por resultado externo.
H-G e G-F primários; H-I,camadasfixas-B,G-B,H-A secundários.
Reportar todas as sementes, sem escolher a melhor. IC95 pareado condicional,
2000 bootstraps por pessoa; repetições da pessoa ficam no mesmo cluster.
AUC por fold ponderada por pares; AUC de margens concatenadas exploratória.
Salvar partições, candidatos, OOFs, escolhas, atributos e parâmetros/hashes.
Checkpoints completos retomáveis, parciais preservados sem sobrescrita.
AVFAD já explorada: desenvolvimento, não teste cego; IC não corrige todo
o histórico de tentativas nem refaz treinamento. Sem promessa80–90%/clínica.
"""


def layer_of(rep):
    return 0 if rep == "WPT128" else int(rep.rsplit("_L", 1)[1])


def branches_for(rep):
    if rep in PREPROCESSORS:
        return (rep,)
    if rep.startswith("HYB128_L") and layer_of(rep) in LAYERS:
        return ("WPT128", f"SSL128_L{layer_of(rep):02d}")
    raise ValueError("Representação desconhecida")


def represented(values, rep):
    branches = branches_for(rep)
    if len(branches) == 2:
        return np.concatenate([values[branches[0]], values[branches[1]]], axis=1) / np.sqrt(2.)
    return values[rep] / np.sqrt(2.) if rep.startswith("SSL256_") else values[rep]


def candidate_id(rep, c):
    return f"{rep}_C{float(c):g}"


def prepare(matrices, y, train, validation):
    train_values, validation_values, arrays = {}, {}, {}
    for branch, (modality, k) in PREPROCESSORS.items():
        pre = EXP13.pipeline_for("SVC_linear", k, 1.)[:-1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pre.fit(matrices[modality][train], y[train])
        train_values[branch] = pre.transform(matrices[modality][train])
        validation_values[branch] = pre.transform(matrices[modality][validation])
        params = {**pre_parameters(pre), "train_rows": np.asarray(train, np.int64)}
        arrays.update({f"{branch}__{name}": value for name, value in params.items()})
    return train_values, validation_values, arrays


def choose(candidates, allowed):
    eligible = candidates[candidates.representation.isin(allowed)].copy()
    if eligible.empty or not np.isfinite(eligible.inner_BA).all():
        raise ValueError("Candidatos internos ausentes ou inválidos")
    tied = eligible[eligible.inner_BA >= eligible.inner_BA.max() - 1e-12].copy()
    tied["layer_order"] = tied.layer.map({layer: i for i, layer in enumerate(LAYERS)})
    if tied.layer_order.isna().any():
        raise ValueError("Camada desconhecida na seleção")
    return tied.sort_values(["C", "layer_order", "candidate_id"], kind="stable").iloc[0].drop(labels="layer_order")


def choices(candidates, seed, fold):
    rows = []
    for arm, rep in FIXED.items():
        rows.append({"seed": seed, "fold": fold, "arm": arm, "representation": rep,
                     "layer": layer_of(rep), "C": 1., "candidate_id": candidate_id(rep, 1.),
                     "inner_BA": np.nan, "choice_source": "fixed_protocol"})
    for arm, allowed in ARM_REPS.items():
        rows.append({**choose(candidates, allowed).to_dict(), "arm": arm, "choice_source": "inner_OOF_BA"})
    return pd.DataFrame(rows)


def inner_search(matrices, meta, train, seed, fold, c_grid=C_GRID, log=lambda _: None):
    outer_meta = meta.iloc[train].reset_index(drop=True)
    y = meta.y.to_numpy(); inner_seed = int(seed) + 100 + int(fold)
    records, partitions, parameters = [], [], {}
    for inner_fold, (itr, iva) in enumerate(EXP13.row_splits(outer_meta, 3, inner_seed), 1):
        tr, va = train[itr], train[iva]
        if set(meta.iloc[tr].participant_id) & set(meta.iloc[va].participant_id):
            raise ValueError("Participante compartilhado no fold interno")
        for role, rows in (("treino", tr), ("validacao", va)):
            for index in rows:
                row = meta.iloc[index]
                partitions.append({"seed": seed, "fold": fold, "inner_seed": inner_seed, "inner_fold": inner_fold,
                                   "role": role, "matrix_row": int(index), "sample_id": row.sample_id,
                                   "participant_id": row.participant_id, "y": row.y})
        values_tr, values_va, arrays = prepare(matrices, y, tr, va)
        parameters.update({f"inner{inner_fold}__{key}": value for key, value in arrays.items()})
        for rep in REPRESENTATIONS:
            xtr, xva = represented(values_tr, rep), represented(values_va, rep)
            for c in c_grid:
                model = fit_classifier(xtr, y[tr], c)
                pred = EXP17.collapse(meta.iloc[va].reset_index(drop=True), model.decision_function(xva))
                pred["candidate_id"], pred["representation"], pred["C"], pred["layer"] = candidate_id(rep, c), rep, float(c), layer_of(rep)
                pred["seed"], pred["fold"], pred["inner_fold"] = seed, fold, inner_fold
                records.append(pred)
        log(f"Busca28 seed={seed} fold={fold} inner={inner_fold}: {len(REPRESENTATIONS)*len(c_grid)} candidatos por participante")
    predictions = pd.concat(records, ignore_index=True)
    candidate_rows = []
    for cid, sub in predictions.groupby("candidate_id", sort=False):
        if sub.participant_id.duplicated().any() or set(sub.participant_id) != set(outer_meta.participant_id):
            raise ValueError("OOF interno incompleto ou duplicado")
        first, measured = sub.iloc[0], EXP17.measures(sub)
        candidate_rows.append({"seed": seed, "fold": fold, "candidate_id": cid, "representation": first.representation,
                               "layer": int(first.layer), "C": float(first.C), "inner_BA": measured["BA"],
                               "inner_sensibilidade": measured["sensibilidade"], "inner_especificidade": measured["especificidade"],
                               "inner_acuracia": measured["acuracia"], "n_participantes": len(sub)})
    candidates = pd.DataFrame(candidate_rows)
    return {"candidatos_internos.csv": candidates, "predicoes_oof_internas.csv": predictions,
            "particoes_internas.csv": pd.DataFrame(partitions), "selecoes.csv": choices(candidates, seed, fold)}, parameters


def compare_reference(pred, reference, old_arm, seed, fold):
    if reference is None:
        return
    source = reference[(reference.seed == seed) & (reference.fold == fold) & (reference.arm == old_arm)]
    actual, expected = pred.set_index("participant_id").sort_index(), source.set_index("participant_id").sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    np.testing.assert_allclose(actual.score, expected.score, atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(actual.predicao, expected.predicao)


def evaluate_fold(matrices, meta, names, train, test, seed, fold, log=lambda _: None, c_grid=C_GRID, reference21=None, reference22=None):
    if set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise ValueError("Participante compartilhado entre treino e teste externo")
    frames, parameters = inner_search(matrices, meta, train, seed, fold, c_grid, log)
    y = meta.y.to_numpy(); values_tr, values_te, arrays = prepare(matrices, y, train, test)
    parameters.update({f"outer__{key}": value for key, value in arrays.items()})
    output, training, recording_test, recording_train, metrics, attrs = [], [], [], [], [], []
    train_meta, test_meta = meta.iloc[train].reset_index(drop=True), meta.iloc[test].reset_index(drop=True)
    for row in frames["selecoes.csv"].itertuples():
        arm, rep, c = row.arm, row.representation, float(row.C)
        model = fit_classifier(represented(values_tr, rep), y[train], c)
        train_score, test_score = model.decision_function(represented(values_tr, rep)), model.decision_function(represented(values_te, rep))
        pred, fit_pred = EXP17.collapse(test_meta, test_score), EXP17.collapse(train_meta, train_score)
        rt, rr = EXP19.recording_predictions(test_meta, test_score, arm, seed, fold), EXP19.recording_predictions(train_meta, train_score, arm, seed, fold)
        for frame in (pred, fit_pred):
            frame["arm"], frame["seed"], frame["fold"], frame["limiar"] = arm, seed, fold, 0.
        for phase, unit, frame in (("teste", "participante", pred), ("treino", "participante", fit_pred), ("teste", "gravacao", rt), ("treino", "gravacao", rr)):
            metrics.append({"arm": arm, "seed": seed, "fold": fold, "phase": phase, "unit": unit,
                            "C": c, "layer": layer_of(rep), **EXP17.measures(frame)})
        parameters.update({f"{arm}__{key}": value for key, value in EXP21.classifier_parameters(model, train).items()})
        parameters.update({f"{arm}__C": np.asarray(c), f"{arm}__layer": np.asarray(layer_of(rep)), f"{arm}__representation": np.asarray(rep)})
        for branch in branches_for(rep):
            modality, _ = PREPROCESSORS[branch]
            mask, select = arrays[f"{branch}__constant_mask"], arrays[f"{branch}__selector_mask"]
            attrs.extend({"arm": arm, "seed": seed, "fold": fold, "modality": modality, "branch": branch,
                          "layer": 0 if modality == "wpt" else layer_of(branch), "feature": str(name)} for name in names[modality][mask][select])
        if arm in ARMS[:2]:
            compare_reference(pred, reference21, "A_WPT" if arm == ARMS[0] else "B_WAV2VEC", seed, fold)
        if arm == ARMS[5]:
            compare_reference(pred, reference22, "B_WAV2VEC_REG", seed, fold)
        output.append(pred); training.append(fit_pred); recording_test.append(rt); recording_train.append(rr)
        log(f"28 seed={seed} fold={fold} {arm}: camada={layer_of(rep)} C={c:g}; BA teste={EXP17.measures(pred)['BA']:.6f}; treino={EXP17.measures(fit_pred)['BA']:.6f}")
    frames.update({"predicoes.csv": pd.concat(output, ignore_index=True), "predicoes_treino_participante.csv": pd.concat(training, ignore_index=True),
                   "predicoes_teste_gravacao.csv": pd.concat(recording_test, ignore_index=True), "predicoes_treino_gravacao.csv": pd.concat(recording_train, ignore_index=True),
                   "metricas.csv": pd.DataFrame(metrics), "atributos.csv": pd.DataFrame(attrs)})
    return frames, parameters


def validate(frames, arrays, meta, train, test, names, seed, fold, c_grid=C_GRID):
    if set(frames) != CSV_FILES or set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise ValueError("Checkpoint incompleto ou vazamento externo")
    candidates, selected, inner = frames["candidatos_internos.csv"], frames["selecoes.csv"], frames["predicoes_oof_internas.csv"]
    expected_candidates = {candidate_id(rep, c) for rep in REPRESENTATIONS for c in c_grid}
    if len(candidates) != len(expected_candidates) or set(candidates.candidate_id) != expected_candidates or candidates.candidate_id.duplicated().any():
        raise ValueError("Candidatos internos incompletos ou duplicados")
    if len(selected) != len(ARMS) or set(selected.arm) != set(ARMS):
        raise ValueError("Escolhas dos nove braços incompletas")
    expected_selected = choices(candidates, seed, fold).set_index("arm")
    expected_y = meta.drop_duplicates("participant_id").set_index("participant_id").y
    train_ids, test_ids = set(meta.iloc[train].participant_id), set(meta.iloc[test].participant_id)
    partitions = frames["particoes_internas.csv"]
    expected_inner = {}
    for inner_fold, (itr, iva) in enumerate(EXP13.row_splits(meta.iloc[train].reset_index(drop=True), 3, int(seed)+100+fold), 1):
        sub = partitions[partitions.inner_fold == inner_fold]
        for role, rows in (("treino", train[itr]), ("validacao", train[iva])):
            actual = sub[sub.role == role]
            if len(actual) != len(rows) or set(actual.matrix_row) != set(rows):
                raise ValueError("Partição interna diferente do protocolo")
            original = meta.iloc[actual.matrix_row.to_numpy(dtype=int)]
            for key in ("sample_id", "participant_id", "y"):
                np.testing.assert_array_equal(actual[key], original[key])
        if set(sub[sub.role == "treino"].participant_id) & set(sub[sub.role == "validacao"].participant_id):
            raise ValueError("Vazamento por participante na validação interna")
        expected_inner[inner_fold] = set(meta.iloc[train[iva]].participant_id)
        for branch in PREPROCESSORS:
            np.testing.assert_array_equal(arrays[f"inner{inner_fold}__{branch}__train_rows"], train[itr])
    if set(partitions.inner_fold) != {1, 2, 3} or not partitions.seed.eq(seed).all() or not partitions.fold.eq(fold).all() or not partitions.inner_seed.eq(int(seed)+100+fold).all():
        raise ValueError("Identidade da partição interna inconsistente")
    if set(inner.candidate_id) != expected_candidates:
        raise ValueError("OOFs dos candidatos incompletos")
    for row in candidates.itertuples():
        sub = inner[inner.candidate_id == row.candidate_id]
        if sub.participant_id.duplicated().any() or set(sub.participant_id) != train_ids or set(sub.participant_id) & test_ids:
            raise ValueError("OOF interno omite/duplica pessoas ou inclui teste")
        if not np.isfinite(sub.score).all() or not np.array_equal(sub.predicao, (sub.score >= 0).astype(int)):
            raise ValueError("Score/decisão interna inválida")
        if not np.array_equal(sub.y, sub.participant_id.map(expected_y)) or any(set(sub[sub.inner_fold == i].participant_id) != ids for i, ids in expected_inner.items()):
            raise ValueError("Rótulo ou fold OOF interno incorreto")
        if not sub.C.eq(row.C).all() or not sub.representation.eq(row.representation).all() or not sub.layer.eq(row.layer).all() or row.layer != layer_of(row.representation) or row.candidate_id != candidate_id(row.representation, row.C):
            raise ValueError("Identidade de candidato inconsistente")
        measures = EXP17.measures(sub)
        for key, value in (("inner_BA", measures["BA"]), ("inner_sensibilidade", measures["sensibilidade"]), ("inner_especificidade", measures["especificidade"]), ("inner_acuracia", measures["acuracia"])):
            if not np.isclose(getattr(row, key), value, atol=1e-12, rtol=0):
                raise ValueError("Métrica interna inconsistente")
        if row.n_participantes != len(sub):
            raise ValueError("Contagem interna inconsistente")
    for row in selected.itertuples():
        expected = expected_selected.loc[row.arm]
        for key in ("representation", "C", "layer", "candidate_id", "choice_source"):
            if getattr(row, key) != expected[key]:
                raise ValueError("Escolha não corresponde ao protocolo/CV interna")
        if row.arm not in FIXED and not np.isclose(row.inner_BA, expected.inner_BA, atol=1e-12, rtol=0):
            raise ValueError("BA escolhida inconsistente")
        if row.arm in FIXED and not pd.isna(row.inner_BA):
            raise ValueError("Braço fixo indevidamente marcado como busca")
        for key in ("representation", "C", "layer"):
            if arrays[f"{row.arm}__{key}"].item() != getattr(row, key):
                raise ValueError("Classificador não corresponde à escolha")
        support = arrays[f"{row.arm}__support_indices"]
        if (support < 0).any() or (support >= len(train)).any() or not np.array_equal(arrays[f"{row.arm}__support_original_rows"], train[support]):
            raise ValueError("Vetores de suporte fora do treino")
        expected_features = []
        for branch in branches_for(row.representation):
            modality, k = PREPROCESSORS[branch]
            var, sel = arrays[f"outer__{branch}__constant_mask"].astype(bool), arrays[f"outer__{branch}__selector_mask"].astype(bool)
            if len(var) != len(names[modality]) or len(sel) != var.sum() or sel.sum() != k:
                raise ValueError("Seleção de atributos inválida")
            expected_features.extend(names[modality][var][sel])
        if list(frames["atributos.csv"].query("arm == @row.arm").feature) != list(expected_features):
            raise ValueError("Atributos exportados diferentes da seleção")
    for branch in PREPROCESSORS:
        np.testing.assert_array_equal(arrays[f"outer__{branch}__train_rows"], train)
    for filename, indices, unit in (("predicoes.csv", test, "participante"), ("predicoes_treino_participante.csv", train, "participante"),
                                    ("predicoes_teste_gravacao.csv", test, "gravacao"), ("predicoes_treino_gravacao.csv", train, "gravacao")):
        frame = frames[filename]; key = "participant_id" if unit == "participante" else "sample_id"
        if set(frame.arm) != set(ARMS):
            raise ValueError("Predições com braços incompletos")
        for arm, sub in frame.groupby("arm"):
            if sub[key].duplicated().any() or set(sub[key]) != set(meta.iloc[indices][key]):
                raise ValueError("IDs de predições incorretos")
            if not sub.seed.eq(seed).all() or not sub.fold.eq(fold).all() or not np.array_equal(sub.y, sub.participant_id.map(expected_y)):
                raise ValueError("Fold ou rótulo externo incorreto")
            if not np.isfinite(sub.score).all() or not np.array_equal(sub.predicao, (sub.score >= 0).astype(int)):
                raise ValueError("Score/decisão externo inválido")
            if unit == "participante":
                counts = meta.iloc[indices].groupby("participant_id").size()
                np.testing.assert_array_equal(sub.n_gravacoes, sub.participant_id.map(counts))
            else:
                original = meta.iloc[indices].set_index("sample_id").loc[sub.sample_id]
                np.testing.assert_array_equal(sub.participant_id, original.participant_id)
                np.testing.assert_array_equal(sub.task.astype(str).str.zfill(3), original.task.astype(str).str.zfill(3))
    metrics = frames["metricas.csv"]
    if len(metrics) != 4*len(ARMS) or metrics.duplicated(["arm", "phase", "unit"]).any():
        raise ValueError("Métricas incompletas ou duplicadas")
    sources = {("teste", "participante"): "predicoes.csv", ("treino", "participante"): "predicoes_treino_participante.csv",
               ("teste", "gravacao"): "predicoes_teste_gravacao.csv", ("treino", "gravacao"): "predicoes_treino_gravacao.csv"}
    for row in metrics.itertuples():
        source = frames[sources[(row.phase, row.unit)]].query("arm == @row.arm")
        for key, value in EXP17.measures(source).items():
            if not np.isclose(value, getattr(row, key), atol=1e-12, rtol=0):
                raise ValueError("Métrica externa inconsistente")
    for people_file, record_file in (("predicoes.csv", "predicoes_teste_gravacao.csv"), ("predicoes_treino_participante.csv", "predicoes_treino_gravacao.csv")):
        for arm in ARMS:
            records = frames[record_file].query("arm == @arm")
            people = frames[people_file].query("arm == @arm").set_index("participant_id").sort_index()
            aggregated = EXP17.collapse(records, records.score.to_numpy()).set_index("participant_id").sort_index()
            np.testing.assert_allclose(people.score, aggregated.score, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(people.predicao, aggregated.predicao)


def intermediate_names(layer):
    return np.asarray([f"WAV2VEC__xlsr300m__layer{layer:02d}__{stat}__{i:04d}"
                       for stat in ("mean", "std") for i in range(1024)])


def validate_multilayer(matrices, meta, names, meta27, embedded, identity27, indices, real=True):
    config = identity27.get("configuration", {})
    expected = {"model_id": EXP21.MODEL_ID, "model_revision": "1a640f32ac3e39899438a2931f9924c02f080a54",
                "frozen": True, "sample_rate": 16000, "hidden_layer": "multiple_intermediate", "hidden_layers": [5, 12, 16],
                "feature_dimension": 6144, "feature_dimension_per_layer": 2048, "pooling": "mean_std",
                "num_hidden_layers": 24, "hidden_size": 1024, "control_layer": 24,
                "control_reference": "complete_exp20_same_recording", "intermediate_final_layer_norm": False,
                "control_atol": 2e-5, "control_rtol": 2e-5, "output_hidden_states": True}
    if identity27.get("mode") != "completo" or any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Configuração27 não corresponde às camadas congeladas previstas")
    if not np.array_equal(indices, [5, 12, 16]) or not np.array_equal(meta27.matrix_row, np.arange(len(meta))):
        raise ValueError("Camadas ou matrix_row27 incorretos")
    if meta27.sample_id.duplicated().any() or meta27.duplicated(["participant_id", "task"]).any():
        raise ValueError("IDs27 duplicados")
    for key in ("sample_id", "participant_id", "task", "y"):
        if key not in meta27 or meta27[key].isna().any() or not np.array_equal(meta27[key].astype(str), meta[key].astype(str)):
            raise ValueError(f"Metadados27 discordam da coorte20/WPT: {key}")
    for key in ("sample_id", "participant_id", "task"):
        if key not in embedded or not np.array_equal(np.asarray(embedded[key]).astype(str), meta[key].astype(str)):
            raise ValueError("IDs NPZ27 diferentes da coorte")
    for layer in LAYERS:
        modality = f"ssl{layer:02d}"
        if matrices[modality].dtype != np.float32 or matrices[modality].shape != (len(meta), 2048) or not np.isfinite(matrices[modality]).all():
            raise ValueError("Matriz de camada inválida")
        expected_names = EXP21.embedding_names() if layer == 24 else intermediate_names(layer)
        if not np.array_equal(names[modality], expected_names):
            raise ValueError("Nomes/ordem das camadas incorretos")
    if meta27.groupby("participant_id").y.nunique().ne(1).any() or set(meta27.y.unique()) != {0, 1}:
        raise ValueError("Rótulos27 inconsistentes por pessoa")
    if not meta27.control24_allclose.eq(True).all() or not np.array_equal(meta27.hidden_frames, meta27.control24_reference_frames):
        raise ValueError("Controle27 contra20 falhou")
    for key in ("control24_max_abs_error", "control24_mean_abs_error"):
        if not np.isfinite(meta27[key]).all() or meta27[key].lt(0).any():
            raise ValueError("Diagnóstico do controle27 inválido")
    if (meta27.control24_mean_abs_error > meta27.control24_max_abs_error).any():
        raise ValueError("Erro médio maior que erro máximo")
    if real:
        people = meta[["participant_id", "y"]].drop_duplicates()
        if (len(meta), len(people), people.y.value_counts().to_dict()) != (2465, 411, {1: 329, 0: 82}):
            raise ValueError("Coorte HSA mudou")


def load_inputs(root, data20, data27):
    matrices20, meta, names20 = EXP21.load_inputs(root, data20)
    matrices, names = {"wpt": matrices20["wpt"], "ssl24": matrices20["wav2vec"]}, {"wpt": names20["wpt"], "ssl24": names20["wav2vec"]}
    marker = json.loads((data27 / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    expected_exports = {"features_wav2vec_camadas.npz", "metadata_wav2vec_camadas.csv", "CONTROLE_REFERENCIA_20.csv",
                        "assinatura_execucao.json", "PROTOCOLO.txt", "RESUMO_EXTRACAO.txt"}
    if set(marker) != expected_exports:
        raise ValueError("Extração27 incompleta")
    for name, digest in marker.items():
        if Path(name).name != name or EXP17.sha(data27 / name) != digest:
            raise ValueError("Export27 alterado")
    ident = json.loads((data27 / "assinatura_execucao.json").read_text(encoding="utf8"))
    for name, digest in ident["code"].items():
        if EXP17.sha(name) != digest:
            raise ValueError("Fonte da extração27 alterada")
    for name, info in ident["inputs"].items():
        path = Path(name)
        if path.stat().st_size != info["size_bytes"] or EXP17.sha(path) != info["sha256"]:
            raise ValueError("Insumo assinado27 alterado")
    for name, info in ident["raw_zip_metadata_only"].items():
        stat = Path(name).stat()
        if stat.st_size != info["size_bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise ValueError("ZIP mudou desde a extração27")
    meta27 = pd.read_csv(data27 / "metadata_wav2vec_camadas.csv", dtype={"sample_id": str, "participant_id": str, "task": str}, float_precision="round_trip")
    meta27["task"] = meta27.task.str.zfill(3)
    with np.load(data27 / "features_wav2vec_camadas.npz", allow_pickle=False) as z:
        for layer in LAYERS[1:]:
            matrices[f"ssl{layer:02d}"] = z[f"embeddings_layer{layer:02d}"]
            names[f"ssl{layer:02d}"] = z[f"feature_names_layer{layer:02d}"].astype(str)
        embedded = {key: z[key] for key in ("sample_id", "participant_id", "task")}
        indices = z["hidden_layer_indices"]
    validate_multilayer(matrices, meta, names, meta27, embedded, ident, indices)
    if ident.get("sample_ids") != meta.sample_id.tolist():
        raise ValueError("Manifesto27 não corresponde à coorte")
    control = pd.read_csv(data27 / "CONTROLE_REFERENCIA_20.csv", dtype={"sample_id": str, "participant_id": str, "task": str}, float_precision="round_trip")
    control["task"] = control.task.str.zfill(3)
    for key in control.columns:
        np.testing.assert_array_equal(control[key], meta27[key])
    if set(control.columns) != {"sample_id", "participant_id", "task", "hidden_frames", "control24_reference_frames", "control24_allclose", "control24_exact", "control24_max_abs_error", "control24_mean_abs_error"}:
        raise ValueError("Controle27 incompleto")
    return matrices, meta, names


def reference22(folder, meta, seeds):
    marker = json.loads((folder / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    if not {"predicoes_oof_participante.csv", "selecoes_internas.csv", "assinatura_execucao.json", "particoes_participantes.csv"}.issubset(marker):
        raise ValueError("Referência22 incompleta")
    for name, digest in marker.items():
        if Path(name).name != name or EXP17.sha(folder / name) != digest:
            raise ValueError("Referência22 alterada")
    raw = pd.read_csv(folder / "predicoes_oof_participante.csv", float_precision="round_trip")
    raw = raw[(raw.arm == "B_WAV2VEC_REG") & raw.seed.isin(seeds)].copy()
    labels = meta.drop_duplicates("participant_id").set_index("participant_id").y
    for seed in seeds:
        sub = raw[raw.seed == seed]
        if sub.participant_id.duplicated().any() or set(sub.participant_id) != set(labels.index) or not np.array_equal(sub.y, sub.participant_id.map(labels)):
            raise ValueError("OOF22 incompleto ou rótulos incorretos")
        for fold, (_, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            if set(sub[sub.fold == fold].participant_id) != set(meta.iloc[test].participant_id):
                raise ValueError("Folds28 divergem do controle22")
    return raw


def identity(root, data20, data27, ref21, ref22, seeds):
    inputs = [root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz",
              root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv"]
    inputs += [data20 / name for name in ("features_wav2vec_xlsr300m.npz", "metadata_wav2vec.csv", "ARTEFATOS_COMPLETOS.json", "assinatura_execucao.json")]
    inputs += [data27 / name for name in ("features_wav2vec_camadas.npz", "metadata_wav2vec_camadas.csv", "CONTROLE_REFERENCIA_20.csv", "ARTEFATOS_COMPLETOS.json", "assinatura_execucao.json")]
    for ref in (ref21, ref22):
        inputs += [ref / name for name in ("predicoes_oof_participante.csv", "ARTEFATOS_COMPLETOS.json", "assinatura_execucao.json", "particoes_participantes.csv")]
    code_names = ("28_avaliar_camadas_ssl_wavelet_avfad.py", "27_extrair_camadas_ssl_avfad.py", "22_regularizar_wpt_wav2vec_avfad.py",
                  "21_avaliar_wpt_wav2vec_avfad.py", "20_extrair_wav2vec_avfad.py", "19_avaliar_wpt_16khz_avfad.py",
                  "17_robustez_distribuicao_wpt_avfad.py", "13_otimizar_wpt_sym5.py")
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
        environment[package] = importlib.metadata.version(package)
    return {"version": VERSION, "configuration": {"arms": list(ARMS), "seeds": list(seeds), "outer_folds": 5, "inner_folds": 3,
            "C_grid": list(C_GRID), "layers": list(LAYERS), "k_per_branch": 128, "budget_k": 256, "threshold": 0.,
            "inner_seed": "outer_seed+100+fold", "selection": "pooled_inner_OOF_BA_participant",
            "tie": "BAwithin1e-12;lowerC;layer_order24_5_12_16", "hybrid_scaling": "concat128_128/sqrt2",
            "budget_scaling": "256/sqrt2", "class_weight": "balanced_recordings_train_only", "bootstrap_iterations": 2000},
            "environment": environment, "input_files": {str(path.resolve()): EXP17.sha(path) for path in inputs},
            "code": {str((HERE / name).resolve()): EXP17.sha(HERE / name) for name in code_names},
            "protocol_sha256": hashlib.sha256(protocol().encode("utf8")).hexdigest()}


def consolidate(out, meta, seeds, ref21, ref22, log):
    destinations = [out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" for seed in seeds for fold in range(1, 6)]
    frames = {name: pd.concat([pd.read_csv(p / name, float_precision="round_trip") for p in destinations], ignore_index=True)
              for name in ("predicoes.csv", "metricas.csv", "atributos.csv", "candidatos_internos.csv", "selecoes.csv")}
    raw, folds = frames["predicoes.csv"], frames["metricas.csv"]
    summaries = []
    for (seed, arm), group in raw.groupby(["seed", "arm"]):
        if group.participant_id.duplicated().any() or len(group) != meta.participant_id.nunique():
            raise ValueError("OOF externo incompleto ou duplicado")
        measured = EXP17.measures(group); measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        sub = folds[(folds.seed == seed) & (folds.arm == arm) & (folds.phase == "teste") & (folds.unit == "participante")]
        measured["AUC_media_folds_ponderada_pares"] = float(np.average(sub.AUC, weights=(sub.TN+sub.FP)*(sub.TP+sub.FN)))
        summaries.append({"seed": seed, "arm": arm, **measured})
    summary = pd.DataFrame(summaries)
    scopes = [("principal_seed20261105", [20261105])] if 20261105 in seeds else []
    secondary = [seed for seed in seeds if seed != 20261105]
    if secondary:
        scopes.append(("estabilidade_secundaria", secondary))
    comparisons = [{"escopo": scope, "comparacao": name, "seeds": str(chosen),
                    **EXP17.intervals(raw, chosen, coefficients, iterations=2000)}
                   for scope, chosen in scopes for name, coefficients in CONTRASTS.items()]
    old21, old22 = ref21.copy(), ref22.copy()
    old21["arm"], old22["arm"] = "REF21_" + old21.arm, "REF22_" + old22.arm
    exports = {"predicoes_oof_participante.csv": raw, "metricas_por_fold.csv": folds, "metricas_por_seed.csv": summary,
               "comparacoes_pareadas.csv": pd.DataFrame(comparisons), "atributos_selecionados.csv": frames["atributos.csv"],
               "candidatos_internos.csv": frames["candidatos_internos.csv"], "selecoes_internas.csv": frames["selecoes.csv"],
               "referencia21_oof.csv": old21, "referencia22_oof.csv": old22}
    for name, frame in exports.items():
        EXP17.csv_write(out / name, frame)
    lines = ["EXPERIMENTO 28 — CAMADAS CONGELADAS E HÍBRIDO WAVELET", f"Consolidado: {EXP17.now()}",
             "Todos os folds solicitados concluídos; seleção exclusivamente interna por participante.",
             "Ciclo completo de seis sementes/nove braços." if tuple(seeds) == SEEDS else "Ciclo parcial/diferente das seis sementes.", ""]
    for row in summary.itertuples():
        lines.append(f"seed={row.seed} {row.arm}: BA={row.BA:.6f}; sens={row.sensibilidade:.6f}; esp={row.especificidade:.6f}; acc={row.acuracia:.6f}; AUCfold={row.AUC_media_folds_ponderada_pares:.6f}; TN/FP/FN/TP={row.TN}/{row.FP}/{row.FN}/{row.TP}")
    lines += ["", "CONTRASTES PAREADOS — IC95 CONDICIONAL"]
    for row in comparisons:
        lines.append(f"{row['escopo']} {row['comparacao']}: ΔBA={row['delta_BA']:+.6f}; IC95=[{row['IC95_inferior']:+.6f}; {row['IC95_superior']:+.6f}]")
    if secondary:
        lines += ["", "ESTABILIDADE NAS CINCO SEMENTES SECUNDÁRIAS — DP NÃO É IC"]
        for arm, group in summary[summary.seed.isin(secondary)].groupby("arm"):
            lines.append(f"{arm}: BA média={group.BA.mean():.6f}; DP={group.BA.std():.6f}; mínimo={group.BA.min():.6f}; máximo={group.BA.max():.6f}")
    lines += ["", "ESCOLHAS INTERNAS POR CAMADA/C"]
    for keys, count in frames["selecoes.csv"].groupby(["arm", "layer", "C"]).size().items():
        lines.append(f"{keys}: {count} folds")
    lines += ["", "DIAGNÓSTICO DE TREINO"]
    for arm, group in folds[(folds.phase == "treino") & (folds.unit == "participante")].groupby("arm"):
        lines.append(f"{arm}: BA treino média entre ajustes={group.BA.mean():.6f}")
    lines += ["", "LIMITAÇÕES", "AVFAD já explorada: desenvolvimento, não teste independente.",
              "Mais camadas significam mais candidatos; G-F testa pipelines completos com busca interna.",
              "H-G e H-I são necessários para contribuição Wavelet; ganho contra A sozinho não a demonstra.",
              "Orçamento256 não iguala representações nem prova causalidade; pooling descarta ordem temporal.",
              "Camada/C do híbrido podem diferir dos escolhidos para SSL isolado; escolhas nunca usam teste externo.",
              "ICs condicionais não refazem seleção/modelos nem corrigem o histórico de múltiplas explorações.",
              "Pesos balanced são por gravação de treino; quase todas as pessoas têm seis frases, uma tem cinco.",
              "Encoder congelado, limiar0 fixo; nenhuma adaptação, SVD ou promessa de resultado clínico.",
              "A/B reproduzem21; F reproduz22B em cada fold antes de publicar os novos resultados.\n"]
    EXP17.text_write(out / "RELATORIO_RESULTADOS.txt", "\n".join(lines))
    names = list(exports) + ["RELATORIO_RESULTADOS.txt", "assinatura_execucao.json", "particoes_participantes.csv", "PROTOCOLO.txt"]
    EXP17.json_write(out / "ARTEFATOS_COMPLETOS.json", {name: EXP17.sha(out / name) for name in names})
    log("Consolidação28 concluída; todos os candidatos, controles e resultados preservados.")


def selftest():
    import unittest
    spec = importlib.util.spec_from_file_location("test_camadas_ssl28", HERE / "test_camadas_ssl28.py")
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT); parser.add_argument("--dados20", type=Path)
    parser.add_argument("--dados27", type=Path); parser.add_argument("--referencia21", type=Path); parser.add_argument("--referencia22", type=Path)
    parser.add_argument("--saida", type=Path, default=ROOT / OUT_REL); parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    actions = parser.add_mutually_exclusive_group(); actions.add_argument("--verificar", action="store_true")
    actions.add_argument("--autoteste", action="store_true"); actions.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        selftest(); return
    root, out, seeds = args.dados.resolve(), args.saida.resolve(), tuple(args.seeds)
    if not out.is_relative_to(root / "results/experiments") or out == root / "results/experiments":
        parser.error("Saída precisa ser subpasta de results/experiments")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed > 2**32-1024 for seed in seeds):
        parser.error("Sementes únicas e válidas obrigatórias")
    data20, data27 = (args.dados20 or root / EXP21.DATA20_REL).resolve(), (args.dados27 or root / DATA27_REL).resolve()
    ref21, ref22 = (args.referencia21 or root / REF21_REL).resolve(), (args.referencia22 or root / REF22_REL).resolve()
    matrices, meta, names = load_inputs(root, data20, data27)
    old21, old22 = EXP22.reference21(ref21, meta, seeds), reference22(ref22, meta, seeds)
    ident = identity(root, data20, data27, ref21, ref22, seeds)
    print(f"ENTRADAS28 OK:411 pessoas/2465 gravações; quatro camadas; nove braços;60 candidatos/innerfold; seeds={seeds}", flush=True)
    if not args.executar:
        print("Somente verificação, sem ajuste ou escrita.", flush=True); return
    signature_file = out / "assinatura_execucao.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf8")) != ident:
            raise RuntimeError("Assinatura28 mudou; preservar todos os artefatos")
    else:
        if out.exists() and any(path.name != "PROTOCOLO.txt" for path in out.iterdir()):
            raise RuntimeError("Pasta sem assinatura contém artefatos; preservar")
        out.mkdir(parents=True, exist_ok=True)
        EXP17.freeze_text(out / "PROTOCOLO.txt", protocol()); EXP17.json_write(signature_file, ident)
    EXP17.freeze_text(out / "PROTOCOLO.txt", protocol())
    EXP17.freeze_text(out / "particoes_participantes.csv", EXP19.fold_table(meta, seeds).to_csv(index=False, lineterminator="\n"))
    signature = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest(); log = EXP17.Logger(out / "LOG_EXECUCAO.txt")
    log(f"Início/retomada28: nove braços, Cgrid={C_GRID}, camadas={LAYERS}; sem escolher por teste externo.")
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            if (destination / "CONCLUIDO.json").exists():
                frames, arrays = load_checkpoint(destination, signature); validate(frames, arrays, meta, train, test, names, seed, fold)
                log(f"Checkpoint28 seed={seed} fold={fold} íntegro; nenhum reajuste"); continue
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Checkpoint parcial preservado: {destination}")
            started = time.monotonic()
            frames, arrays = evaluate_fold(matrices, meta, names, train, test, seed, fold, log, reference21=old21, reference22=old22)
            validate(frames, arrays, meta, train, test, names, seed, fold)
            save_checkpoint(destination, frames, arrays, signature)
            log(f"Concluído28 seed={seed} fold={fold}; segundos={time.monotonic()-started:.2f}")
    consolidate(out, meta, seeds, old21, old22, log)


if __name__ == "__main__":
    main()

