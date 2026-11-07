# -*- coding: utf-8 -*-
"""Exp22: regularizacao/fusao selecionadas em CV interna por participante."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp21_for_exp22", HERE / "21_avaliar_wpt_wav2vec_avfad.py")
EXP21 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP21)
EXP19, EXP17, EXP13 = EXP21.EXP19, EXP21.EXP17, EXP21.EXP13
VERSION = "22.1"
SEEDS = EXP21.SEEDS
C_GRID = (1e-4, 1e-3, 1e-2, .1, 1.0)
ALPHAS = (.25, .5, .75)
ARMS = ("A_WPT_REG", "B_WAV2VEC_REG", "C_HIBRIDO_REG", "D_WAV2VEC256_REG", "E_HIBRIDO_PONDERADO_REG")
REPRESENTATIONS = ("WPT128", "WAV128", "WAV256", "HYB025", "HYB050", "HYB075")
ALPHA = {"WPT128": 1., "WAV128": 0., "WAV256": 0., "HYB025": .25, "HYB050": .5, "HYB075": .75}
PREPROCESSORS = {"WPT128": ("wpt", 128), "WAV128": ("wav2vec", 128), "WAV256": ("wav2vec", 256)}
ARM_REPS = {ARMS[0]: ("WPT128",), ARMS[1]: ("WAV128",), ARMS[2]: ("HYB050",),
            ARMS[3]: ("WAV256",), ARMS[4]: ("HYB025", "HYB050", "HYB075")}
OUT_REL = Path("results/experiments/resultados_22_regularizacao_wpt_wav2vec")
REF21_REL = Path("results/experiments/resultados_21_wpt_wav2vec")
CSV_FILES = {"predicoes.csv", "predicoes_treino_participante.csv", "predicoes_teste_gravacao.csv",
             "predicoes_treino_gravacao.csv", "metricas.csv", "atributos.csv", "candidatos_internos.csv",
             "predicoes_oof_internas.csv", "particoes_internas.csv", "selecoes.csv"}
ARTIFACTS = CSV_FILES | {"parametros_ajuste.npz"}
CONTRASTS = {"C-B_complementaridade_primaria": {ARMS[2]: 1, ARMS[1]: -1},
             "E-B_fusao_ponderada_secundaria": {ARMS[4]: 1, ARMS[1]: -1},
             "C-A_WPT_secundaria": {ARMS[2]: 1, ARMS[0]: -1},
             "C-D_orcamento256_secundaria": {ARMS[2]: 1, ARMS[3]: -1},
             "E-D_orcamento256_secundaria": {ARMS[4]: 1, ARMS[3]: -1},
             "E-C_pesos_aprendidos_secundaria": {ARMS[4]: 1, ARMS[2]: -1},
             "C-C21_regularizacao_secundaria": {ARMS[2]: 1, "REF21_C_HIBRIDO": -1},
             "E-C21_regularizacao_fusao_secundaria": {ARMS[4]: 1, "REF21_C_HIBRIDO": -1},
             "B-B21_regularizacao_secundaria": {ARMS[1]: 1, "REF21_B_WAV2VEC": -1},
             "D-D21_regularizacao_secundaria": {ARMS[3]: 1, "REF21_D_WAV2VEC_256": -1},
             "A-A21_regularizacao_secundaria": {ARMS[0]: 1, "REF21_A_WPT": -1}}


def protocol():
    return """EXPERIMENTO22 - REGULARIZACAO E FUSAO WPT + WAV2VEC2
Hipotese: reduzir sobreajuste observado no21 com C menor pode melhorar
generalizacao; ponderar os ramos pode reduzir informacao redundante.
Mantem coorte411/82/329/2465, WPT48k sym5 L5+L6(1920) e embeddings20
XLS-R300M congelados(2048, ultima camada, media/DP). Nenhuma nova extracao,
camada, pooling, exclusao, SMOTE, dado SVD ou ajuste do encoder.
Seis sementes externas21:20261105 principal e101..105 estabilidade, cinco
folds estratificados na tabela de pessoas; frases sempre juntas.
Tres folds INTERNOS por pessoa em cada treino externo, seed=outerseed+100+fold.
Imputacao mediana/constantes/ANOVA/StandardScaler sao reajustados somente no
treino interno ao buscar; depois reajustados no treino externo completo.
Seleciona C em {0.0001,0.001,0.01,0.1,1}; SVClinear/class_weightbalanced,
limiar0, mesmo k do21:128WPT,128wav ou256wav. Sem ajuste de limiar.
A_WPT_REG:128WPT; B_WAV2VEC_REG:128wav;
C_HIBRIDO_REG:128WPT+128wav, ambos divididos por sqrt2;
D_WAV2VEC256_REG:256wav divididos por sqrt2, controle de orcamento;
E_HIBRIDO_PONDERADO_REG:128WPT+128wav, pesos sqrt(alpha),sqrt(1-alpha),
alphaWPT em{0.25,0.5,0.75} escolhido junto comC somente internamente.
A/B/C/D buscam5C cada; E busca15paresC/alpha. O candidatoalpha.5 e comum
aC/E, sem repetir ajustes.30candidatos unicos por treino interno.
Criterio: BA sobre TODAS as predicoesOOF internas por pessoa (media das
margens das frases, limiar0), nao BA por gravacao nem acuracia bruta.
Empates a1e-12: menorC, depois alpha mais proximo de.5, depois menoralpha.
2700ajustesSVC internos +150externos no ciclo completo. Preprocessamento
compartilhado somente dentro da mesma particao de treino e independente deC.
Sem escolha por resultado externo. Todas as tentativas e negativos mantidos.
C-B primario; E-B,C-A,C-D,E-D,E-C e comparacoes de cada ramo com21 secundarios.
Referencias21 importadas com IDs/rotulos/folds/hashes, sem reajustar modelos.
Reporta seedprincipal e cinco deestabilidade separadamente, BA/sens/esp/acc,
matriz/AUCfold ponderada porpares; AUC concatenada apenas exploratoria.
IC95pareado2000bootstraps estratificados porpessoa; repeticoes de seed da
mesma pessoa ficam juntas. ICcondicional nao refaz treino/selecao nem corrige
exploracao historica da AVFAD. Desenvolvimento, nao teste cego independente.
Capacidade/representacoes e pre-processamento diferem; ganho contraWavelet
isolada nao prova contribuicaoWavelet: avaliarC-B/E-B e controlesC-D/E-D.
Salvar particoes internas/externas, candidatos, OOFs, selecoes e parametros
aprendidos/hashes/checkpoints. Dados/caches/experimentos anteriores somenteleitura.
"""


def classifier(c):
    return SVC(kernel="linear", C=float(c), class_weight="balanced", cache_size=300)


def fit_classifier(x, y, c):
    model = classifier(c).fit(x, y)
    if model.fit_status_ != 0:
        raise RuntimeError("SVC nao convergiu; preservar falha sem reportar modelo parcial")
    return model


def pre_parameters(model):
    s = model.named_steps
    return {"imputer": s["imputer"].statistics_.copy(), "constant_mask": s["remove_constants"].get_support(),
            "selector_scores": s["select"].scores_.copy(), "selector_mask": s["select"].get_support(),
            "scaler_mean": s["scale"].mean_.copy(), "scaler_scale": s["scale"].scale_.copy()}


def prepare(matrices, y, train, validation):
    """Nenhum y/X de validation entra em fit ou influencia parametros."""
    train_values, validation_values, arrays = {}, {}, {}
    for key, (modality, k) in PREPROCESSORS.items():
        pre = EXP13.pipeline_for("SVC_linear", k, 1.0)[:-1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pre.fit(matrices[modality][train], y[train])
        train_values[key] = pre.transform(matrices[modality][train])
        validation_values[key] = pre.transform(matrices[modality][validation])
        arrays.update({f"{key}__{name}": value for name, value in pre_parameters(pre).items()})
    return train_values, validation_values, arrays


def represented(values, rep):
    if rep in ("WPT128", "WAV128"):
        return values[rep]
    if rep == "WAV256":
        return values[rep] / np.sqrt(2.)
    alpha = ALPHA[rep]
    # alpha=.5 usa exatamente a operacao numerica do21.
    if alpha == .5:
        return np.concatenate([values["WPT128"], values["WAV128"]], axis=1) / np.sqrt(2.)
    return np.concatenate([values["WPT128"] * np.sqrt(alpha), values["WAV128"] * np.sqrt(1. - alpha)], axis=1)


def candidate_id(rep, c):
    return f"{rep}_C{float(c):g}"


def choose(candidates, allowed):
    eligible = candidates[candidates.representation.isin(allowed)].copy()
    if eligible.empty or not np.isfinite(eligible.inner_BA).all():
        raise ValueError("Candidatos internos ausentes/invalidos")
    tied = eligible[eligible.inner_BA >= eligible.inner_BA.max() - 1e-12].copy()
    tied["alpha_distance"] = (tied.alpha - .5).abs()
    return tied.sort_values(["C", "alpha_distance", "alpha", "candidate_id"], kind="stable").iloc[0]


def inner_search(matrices, meta, train, seed, fold, c_grid=C_GRID, log=lambda _: None):
    outer_meta = meta.iloc[train].reset_index(drop=True)
    y = meta.y.to_numpy()
    inner_seed = int(seed) + 100 + int(fold)
    records, partition_rows, parameters = [], [], {}
    for inner_fold, (local_train, local_val) in enumerate(EXP13.row_splits(outer_meta, 3, inner_seed), 1):
        tr, va = train[local_train], train[local_val]
        if set(meta.iloc[tr].participant_id) & set(meta.iloc[va].participant_id):
            raise ValueError("Pessoa compartilhada no fold interno")
        for role, rows in (("treino", tr), ("validacao", va)):
            for index in rows:
                row = meta.iloc[index]
                partition_rows.append({"seed": seed, "fold": fold, "inner_seed": inner_seed,
                                       "inner_fold": inner_fold, "role": role, "matrix_row": int(index),
                                       "sample_id": row.sample_id, "participant_id": row.participant_id, "y": row.y})
        values_tr, values_va, arrays = prepare(matrices, y, tr, va)
        parameters.update({f"inner{inner_fold}__{key}": value for key, value in arrays.items()})
        for rep in REPRESENTATIONS:
            xtr, xva = represented(values_tr, rep), represented(values_va, rep)
            for c in c_grid:
                model = fit_classifier(xtr, y[tr], c)
                pred = EXP17.collapse(meta.iloc[va].reset_index(drop=True), model.decision_function(xva))
                pred["candidate_id"], pred["representation"], pred["C"], pred["alpha"] = candidate_id(rep, c), rep, float(c), ALPHA[rep]
                pred["seed"], pred["fold"], pred["inner_fold"] = seed, fold, inner_fold
                records.append(pred)
        log(f"Busca22 seed={seed} fold={fold} inner={inner_fold}: {len(REPRESENTATIONS)*len(c_grid)} candidatos, validacao por pessoa")
    predictions = pd.concat(records, ignore_index=True)
    rows = []
    for cid, group in predictions.groupby("candidate_id", sort=False):
        if group.participant_id.duplicated().any() or set(group.participant_id) != set(outer_meta.participant_id):
            raise ValueError("OOF interno incompleto/duplicado")
        measured = EXP17.measures(group)
        first = group.iloc[0]
        rows.append({"seed": seed, "fold": fold, "candidate_id": cid,
                     "representation": first.representation, "C": first.C, "alpha": first.alpha,
                     "inner_BA": measured["BA"], "inner_sensibilidade": measured["sensibilidade"],
                     "inner_especificidade": measured["especificidade"], "inner_acuracia": measured["acuracia"],
                     "n_participantes": len(group)})
    candidates = pd.DataFrame(rows)
    selections = pd.DataFrame([{**choose(candidates, ARM_REPS[arm]).drop(labels="alpha_distance", errors="ignore").to_dict(), "arm": arm}
                               for arm in ARMS])
    return {"candidatos_internos.csv": candidates, "predicoes_oof_internas.csv": predictions,
            "particoes_internas.csv": pd.DataFrame(partition_rows), "selecoes.csv": selections}, parameters


def evaluate_fold(matrices, meta, names, train, test, seed, fold, log=lambda _: None, c_grid=C_GRID, reference21=None):
    if set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise ValueError("Pessoa compartilhada em treino/teste externo")
    frames, parameters = inner_search(matrices, meta, train, seed, fold, c_grid, log)
    y = meta.y.to_numpy()
    values_tr, values_te, arrays = prepare(matrices, y, train, test)
    parameters.update({f"outer__{key}": value for key, value in arrays.items()})
    output, training, recording_test, recording_train, metrics, attributes = [], [], [], [], [], []
    train_meta, test_meta = meta.iloc[train].reset_index(drop=True), meta.iloc[test].reset_index(drop=True)
    for selected in frames["selecoes.csv"].itertuples():
        arm, rep, c = selected.arm, selected.representation, float(selected.C)
        xtr, xte = represented(values_tr, rep), represented(values_te, rep)
        model = fit_classifier(xtr, y[train], c)
        score_train, score_test = model.decision_function(xtr), model.decision_function(xte)
        pred, fit_pred = EXP17.collapse(test_meta, score_test), EXP17.collapse(train_meta, score_train)
        rt = EXP19.recording_predictions(test_meta, score_test, arm, seed, fold)
        rr = EXP19.recording_predictions(train_meta, score_train, arm, seed, fold)
        for frame in (pred, fit_pred):
            frame["arm"], frame["seed"], frame["fold"], frame["limiar"] = arm, seed, fold, 0.
        for phase, unit, frame in (("teste", "participante", pred), ("treino", "participante", fit_pred),
                                   ("teste", "gravacao", rt), ("treino", "gravacao", rr)):
            metrics.append({"arm": arm, "seed": seed, "fold": fold, "phase": phase, "unit": unit,
                            "C": c, "alpha": ALPHA[rep], **EXP17.measures(frame)})
        parameters.update({f"{arm}__{key}": value for key, value in EXP21.classifier_parameters(model, train).items()})
        parameters[f"{arm}__C"] = np.asarray(c)
        parameters[f"{arm}__alpha"] = np.asarray(ALPHA[rep])
        parameters[f"{arm}__representation"] = np.asarray(rep)
        branches = (rep,) if rep in PREPROCESSORS else ("WPT128", "WAV128")
        for branch in branches:
            modality, _ = PREPROCESSORS[branch]
            var, sel = arrays[f"{branch}__constant_mask"], arrays[f"{branch}__selector_mask"]
            attributes.extend({"arm": arm, "seed": seed, "fold": fold, "modality": modality,
                               "branch": branch, "feature": str(name)} for name in names[modality][var][sel])
        if reference21 is not None and c == 1. and (rep != "HYB025" and rep != "HYB075"):
            ref_arm = {"WPT128": "A_WPT", "WAV128": "B_WAV2VEC", "WAV256": "D_WAV2VEC_256", "HYB050": "C_HIBRIDO"}[rep]
            reference = reference21[(reference21.seed == seed) & (reference21.fold == fold) & (reference21.arm == ref_arm)]
            left, right = pred.set_index("participant_id").sort_index(), reference.set_index("participant_id").sort_index()
            pd.testing.assert_index_equal(left.index, right.index)
            np.testing.assert_allclose(left.score, right.score, atol=1e-9, rtol=1e-9)
            np.testing.assert_array_equal(left.predicao, right.predicao)
        output.append(pred); training.append(fit_pred); recording_test.append(rt); recording_train.append(rr)
        log(f"seed={seed} fold={fold} {arm}: C={c:g} alpha={ALPHA[rep]:g}; BA_teste={EXP17.measures(pred)['BA']:.6f}; BA_treino={EXP17.measures(fit_pred)['BA']:.6f}")
    frames.update({"predicoes.csv": pd.concat(output, ignore_index=True),
                   "predicoes_treino_participante.csv": pd.concat(training, ignore_index=True),
                   "predicoes_teste_gravacao.csv": pd.concat(recording_test, ignore_index=True),
                   "predicoes_treino_gravacao.csv": pd.concat(recording_train, ignore_index=True),
                   "metricas.csv": pd.DataFrame(metrics), "atributos.csv": pd.DataFrame(attributes)})
    return frames, parameters


def validate(frames, arrays, meta, train, test, names, seed, fold, c_grid=C_GRID):
    if set(frames) != CSV_FILES or set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise ValueError("Checkpoint incompleto ou participante compartilhado")
    selections, candidates, inner = frames["selecoes.csv"], frames["candidatos_internos.csv"], frames["predicoes_oof_internas.csv"]
    if len(selections) != len(ARMS) or set(selections.arm) != set(ARMS) or len(candidates) != len(REPRESENTATIONS)*len(c_grid):
        raise ValueError("Candidatos/selecoes incompletos")
    outer_ids, test_ids = set(meta.iloc[train].participant_id), set(meta.iloc[test].participant_id)
    expected_y = meta.drop_duplicates("participant_id").set_index("participant_id").y
    partitions = frames["particoes_internas.csv"]
    expected_inner = []
    for inner_fold, (tr, va) in enumerate(EXP13.row_splits(meta.iloc[train].reset_index(drop=True), 3, int(seed)+100+fold), 1):
        sub = partitions[partitions.inner_fold == inner_fold]
        for role, rows in (("treino", train[tr]), ("validacao", train[va])):
            actual = sub[sub.role == role]
            if set(actual.matrix_row) != set(rows) or len(actual) != len(rows):
                raise ValueError("Particoes internas diferem da regra registrada")
            original = meta.iloc[actual.matrix_row.to_numpy(dtype=int)]
            if not np.array_equal(actual.sample_id, original.sample_id) or not np.array_equal(actual.participant_id, original.participant_id):
                raise ValueError("Identidade de particao interna incorreta")
            if not np.array_equal(actual.y, original.y):
                raise ValueError("Rotulos de particao interna incorretos")
        if set(sub[sub.role == "treino"].participant_id) & set(sub[sub.role == "validacao"].participant_id):
            raise ValueError("Vazamento interno")
        expected_inner.append(set(meta.iloc[train[va]].participant_id))
    for row in candidates.itertuples():
        sub = inner[inner.candidate_id == row.candidate_id]
        if sub.participant_id.duplicated().any() or set(sub.participant_id) != outer_ids or set(sub.participant_id) & test_ids:
            raise ValueError("OOF interno duplica, omite ou inclui teste externo")
        if not np.isfinite(sub.score).all() or not np.array_equal(sub.predicao, (sub.score >= 0).astype(int)):
            raise ValueError("Score/decisao interna invalida")
        if not np.array_equal(sub.y, sub.participant_id.map(expected_y)):
            raise ValueError("Rotulo interno incorreto")
        if any(set(sub[sub.inner_fold == i].participant_id) != ids for i, ids in enumerate(expected_inner, 1)):
            raise ValueError("Fold OOF interno incorreto")
        if not np.isclose(EXP17.measures(sub)["BA"], row.inner_BA, atol=1e-12, rtol=0):
            raise ValueError("BA interna incorreta")
        if not sub.C.eq(row.C).all() or not sub.representation.eq(row.representation).all() or not sub.alpha.eq(row.alpha).all():
            raise ValueError("Identidade de candidato inconsistente")
    for selected in selections.itertuples():
        winner = choose(candidates, ARM_REPS[selected.arm])
        if selected.candidate_id != winner.candidate_id or selected.C != winner.C or selected.alpha != winner.alpha:
            raise ValueError("Selecao nao corresponde exclusivamente ao criterio interno")
        if float(arrays[f"{selected.arm}__C"]) != selected.C or str(arrays[f"{selected.arm}__representation"]) != selected.representation:
            raise ValueError("Classificador nao corresponde ao candidato escolhido")
        support = arrays[f"{selected.arm}__support_indices"]
        if (support < 0).any() or (support >= len(train)).any() or not np.array_equal(arrays[f"{selected.arm}__support_original_rows"], train[support]):
            raise ValueError("Vetores de suporte fora de treino")
        rep = selected.representation
        branches = (rep,) if rep in PREPROCESSORS else ("WPT128", "WAV128")
        selected_attrs = frames["atributos.csv"].query("arm == @selected.arm")
        expected_features = []
        for branch in branches:
            modality, k = PREPROCESSORS[branch]
            var, sel = arrays[f"outer__{branch}__constant_mask"].astype(bool), arrays[f"outer__{branch}__selector_mask"].astype(bool)
            if sel.sum() != k or len(var) != len(names[modality]) or len(sel) != var.sum():
                raise ValueError("Selecao de atributos invalida")
            expected_features.extend(names[modality][var][sel])
        if list(selected_attrs.feature) != list(expected_features):
            raise ValueError("Atributos exportados diferem do ajuste")
    for filename, indices, unit in (("predicoes.csv", test, "participante"), ("predicoes_treino_participante.csv", train, "participante"),
                                    ("predicoes_teste_gravacao.csv", test, "gravacao"), ("predicoes_treino_gravacao.csv", train, "gravacao")):
        frame = frames[filename]
        if set(frame.arm) != set(ARMS):
            raise ValueError("Bracos incompletos")
        key = "participant_id" if unit == "participante" else "sample_id"
        for arm, sub in frame.groupby("arm"):
            if sub[key].duplicated().any() or set(sub[key]) != set(meta.iloc[indices][key]):
                raise ValueError("IDs de predicoes inesperados")
            if not sub.seed.eq(seed).all() or not sub.fold.eq(fold).all() or not np.array_equal(sub.y, sub.participant_id.map(expected_y)):
                raise ValueError("Particao/rotulo incorreto em predicoes")
            if not np.isfinite(sub.score).all() or not np.array_equal(sub.predicao, (sub.score >= 0).astype(int)):
                raise ValueError("Score/decisao invalida")
            if unit == "participante":
                counts = meta.iloc[indices].groupby("participant_id").size()
                if not np.array_equal(sub.n_gravacoes, sub.participant_id.map(counts)):
                    raise ValueError("Contagem de frases incorreta")
            else:
                original = meta.iloc[indices].set_index("sample_id").loc[sub.sample_id]
                if not np.array_equal(sub.participant_id, original.participant_id) or not np.array_equal(sub.task.astype(str).str.zfill(3), original.task.astype(str).str.zfill(3)):
                    raise ValueError("Identidade de gravacao incorreta")
    if len(frames["metricas.csv"]) != 4*len(ARMS) or frames["metricas.csv"].duplicated(["arm", "phase", "unit"]).any():
        raise ValueError("Metricas incompletas/duplicadas")
    sources = {("teste", "participante"): "predicoes.csv", ("treino", "participante"): "predicoes_treino_participante.csv",
               ("teste", "gravacao"): "predicoes_teste_gravacao.csv", ("treino", "gravacao"): "predicoes_treino_gravacao.csv"}
    for row in frames["metricas.csv"].itertuples():
        source = frames[sources[(row.phase, row.unit)]].query("arm == @row.arm")
        for key, value in EXP17.measures(source).items():
            if not np.isclose(value, getattr(row, key), atol=1e-12, rtol=0):
                raise ValueError("Metrica nao corresponde a predicoes")
    for people_file, record_file in (("predicoes.csv", "predicoes_teste_gravacao.csv"), ("predicoes_treino_participante.csv", "predicoes_treino_gravacao.csv")):
        for arm in ARMS:
            records = frames[record_file].query("arm == @arm")
            people = frames[people_file].query("arm == @arm").set_index("participant_id").sort_index()
            aggregated = EXP17.collapse(records, records.score.to_numpy()).set_index("participant_id").sort_index()
            np.testing.assert_allclose(people.score, aggregated.score, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(people.predicao, aggregated.predicao)


def save_checkpoint(destination, frames, arrays, signature):
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        EXP17.csv_write(destination / name, frame)
    stream = io.BytesIO(); np.savez_compressed(stream, **arrays)
    EXP17.atomic_bytes(destination / "parametros_ajuste.npz", stream.getvalue())
    EXP17.json_write(destination / "CONCLUIDO.json", {"finished": EXP17.now(), "run_signature": signature,
                      "sha256": {name: EXP17.sha(destination / name) for name in ARTIFACTS}})


def load_checkpoint(destination, signature):
    marker = json.loads((destination / "CONCLUIDO.json").read_text(encoding="utf-8"))
    if marker["run_signature"] != signature or set(marker["sha256"]) != ARTIFACTS:
        raise ValueError("Checkpoint incompleto ou de outra configuracao")
    for name, digest in marker["sha256"].items():
        if EXP17.sha(destination / name) != digest:
            raise ValueError(f"Checkpoint alterado:{name}")
    frames = {name: pd.read_csv(destination / name, float_precision="round_trip") for name in CSV_FILES}
    with np.load(destination / "parametros_ajuste.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    return frames, arrays


def reference21(ref, meta, seeds):
    marker = json.loads((ref / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    for name, digest in marker.items():
        if Path(name).name != name or EXP17.sha(ref / name) != digest:
            raise ValueError("Referencia21 alterada")
    raw = pd.read_csv(ref / "predicoes_oof_participante.csv", float_precision="round_trip")
    expected_y = meta.drop_duplicates("participant_id").set_index("participant_id").y
    for seed in seeds:
        for arm in EXP21.ARMS:
            sub = raw[(raw.seed == seed) & (raw.arm == arm)]
            if sub.participant_id.duplicated().any() or set(sub.participant_id) != set(expected_y.index) or not np.array_equal(sub.y, sub.participant_id.map(expected_y)):
                raise ValueError("Referencia21 incompleta/rotulos incorretos")
            for fold, (_, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
                if set(sub[sub.fold == fold].participant_id) != set(meta.iloc[test].participant_id):
                    raise ValueError("Folds22 divergem da referencia21")
    return raw[raw.seed.isin(seeds)].copy()


def identity(root, data20, ref21, seeds):
    inputs = [root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz",
              root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv",
              *[data20 / name for name in ("features_wav2vec_xlsr300m.npz", "metadata_wav2vec.csv", "ARTEFATOS_COMPLETOS.json", "assinatura_execucao.json")],
              *[ref21 / name for name in ("predicoes_oof_participante.csv", "particoes_participantes.csv", "ARTEFATOS_COMPLETOS.json", "assinatura_execucao.json")]]
    code = [Path(__file__).resolve(), HERE / "21_avaliar_wpt_wav2vec_avfad.py", HERE / "20_extrair_wav2vec_avfad.py",
            HERE / "19_avaliar_wpt_16khz_avfad.py", HERE / "17_robustez_distribuicao_wpt_avfad.py", HERE / "13_otimizar_wpt_sym5.py"]
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
        environment[package] = importlib.metadata.version(package)
    return {"version": VERSION, "configuration": {"arms": ARMS, "seeds": seeds, "outer_folds": 5,
            "inner_folds": 3, "inner_seed": "outer_seed+100+outer_fold", "C_grid": C_GRID, "alpha_grid_E": ALPHAS,
            "fixed_alpha_C": .5, "k_per_branch": 128, "budget_k": 256, "threshold": 0.,
            "class_weight": "balanced", "kernel": "linear", "selection": "pooled_inner_OOF_BA_participant",
            "tie": "BA1e-12;lowerC;alpha_closest_.5;loweralpha", "bootstrap_iterations": 2000},
            "environment": environment, "input_files": {str(p.resolve()): EXP17.sha(p) for p in inputs},
            "code": {str(p.resolve()): EXP17.sha(p) for p in code},
            "protocol_sha256": hashlib.sha256(protocol().encode("utf8")).hexdigest()}


def consolidate(out, meta, seeds, reference, log):
    destinations = [out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" for seed in seeds for fold in range(1, 6)]
    frames = {name: pd.concat([pd.read_csv(p/name, float_precision="round_trip") for p in destinations], ignore_index=True)
              for name in ("predicoes.csv", "metricas.csv", "atributos.csv", "candidatos_internos.csv", "selecoes.csv")}
    raw, folds = frames["predicoes.csv"], frames["metricas.csv"]
    reference = reference.copy(); reference["arm"] = "REF21_" + reference.arm
    comparisons_raw = pd.concat([raw, reference], ignore_index=True)
    summaries = []
    for (seed, arm), group in raw.groupby(["seed", "arm"]):
        if group.participant_id.duplicated().any() or len(group) != meta.participant_id.nunique():
            raise ValueError("OOF externo incompleto/duplicado")
        measured = EXP17.measures(group); measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        sub = folds[(folds.seed == seed) & (folds.arm == arm) & (folds.phase == "teste") & (folds.unit == "participante")]
        measured["AUC_media_folds_ponderada_pares"] = float(np.average(sub.AUC, weights=(sub.TN+sub.FP)*(sub.TP+sub.FN)))
        summaries.append({"seed": seed, "arm": arm, **measured})
    summary = pd.DataFrame(summaries)
    scopes = [("principal_seed20261105", [20261105])] if 20261105 in seeds else []
    secondary = [s for s in seeds if s != 20261105]
    if secondary: scopes.append(("estabilidade_secundaria", secondary))
    contrasts = [{"escopo": label, "comparacao": name, "seeds": str(chosen),
                  **EXP17.intervals(comparisons_raw, chosen, coefficients, iterations=2000)}
                 for label, chosen in scopes for name, coefficients in CONTRASTS.items()]
    exports = {"predicoes_oof_participante.csv": raw, "metricas_por_fold.csv": folds,
               "metricas_por_seed.csv": summary, "comparacoes_pareadas.csv": pd.DataFrame(contrasts),
               "atributos_selecionados.csv": frames["atributos.csv"], "candidatos_internos.csv": frames["candidatos_internos.csv"],
               "selecoes_internas.csv": frames["selecoes.csv"], "referencia21_oof.csv": reference}
    for name, frame in exports.items(): EXP17.csv_write(out/name, frame)
    lines = ["EXPERIMENTO22 - CV ANINHADA REGULARIZACAO/FUSAO", f"Consolidado:{EXP17.now()}",
             "Todos os folds solicitados concluidos; sem ajuste de limiar ou uso do teste na selecao.",
             "Ciclo seis sementes completo." if tuple(seeds) == SEEDS else "Ciclo parcial/diferente do protocolo.", ""]
    for row in summary.itertuples():
        lines.append(f"seed={row.seed} {row.arm}:BA={row.BA:.6f};sens={row.sensibilidade:.6f};esp={row.especificidade:.6f};acc={row.acuracia:.6f};AUCfold={row.AUC_media_folds_ponderada_pares:.6f};TN/FP/FN/TP={row.TN}/{row.FP}/{row.FN}/{row.TP}")
    lines += ["", "CONTRASTES PAREADOS; IC95CONDICIONAL2000BOOTSTRAPS POR PESSOA"]
    for row in contrasts: lines.append(f"{row['escopo']} {row['comparacao']}:deltaBA={row['delta_BA']:+.6f};IC95=[{row['IC95_inferior']:+.6f},{row['IC95_superior']:+.6f}]")
    if secondary:
        lines += ["", "ESTABILIDADE NAS CINCO SEMENTES SECUNDARIAS"]
        for arm, group in summary[summary.seed.isin(secondary)].groupby("arm"):
            lines.append(f"{arm}:BAmedia={group.BA.mean():.6f};min={group.BA.min():.6f};max={group.BA.max():.6f};DP={group.BA.std():.6f}")
    lines += ["", "ESCOLHAS INTERNAS POR BRACO/C/ALPHA"]
    for keys, count in frames["selecoes.csv"].groupby(["arm", "C", "alpha"]).size().items(): lines.append(f"{keys}: {count} folds")
    lines += ["", "TREINO APENAS COMO DIAGNOSTICO"]
    for (arm, unit), group in folds[folds.phase == "treino"].groupby(["arm", "unit"]): lines.append(f"{arm}/{unit}:BAmediaentreajustes={group.BA.mean():.6f}")
    lines += ["", "LIMITACOES", "Coorte AVFAD repetidamente explorada: desenvolvimento, nao teste independente.",
              "CV interna evita vazamento direto de selecao, mas nao elimina adaptacao historica ao conjunto.",
              "ICs condicionais aos folds/modelos; nao refazem selecao/treinamento e nao corrigem multiplas tentativas.",
              "Media das margens porpessoa e threshold0; AUCconcatenada apenas descritiva.",
              "C/E256dimensoes vsB128;D controla256/sqrt2, mas representacoes e selecao diferem.",
              "Pesoalpha muda variancia por ramo; C e alpha sao selecionados conjuntamente so na CV interna.",
              "Embedding pode capturar sexo/aquisicao; nenhum dado SVD foi avaliado.",
              "Nenhuma conclusao final do TG nem afirmacao de auxilioWavelet automatica.\n"]
    EXP17.text_write(out/"RELATORIO_RESULTADOS.txt", "\n".join(lines))
    names = list(exports) + ["RELATORIO_RESULTADOS.txt", "assinatura_execucao.json", "particoes_participantes.csv", "PROTOCOLO.txt"]
    EXP17.json_write(out/"ARTEFATOS_COMPLETOS.json", {name: EXP17.sha(out/name) for name in names})
    log("Consolidacao22 concluida; todos os bracos, candidatos e negativos preservados.")


def selftest():
    import unittest
    spec = importlib.util.spec_from_file_location("test_wpt_wav2vec22", HERE / "test_wpt_wav2vec22.py")
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path, default=ROOT/OUT_REL)
    parser.add_argument("--dados20", type=Path)
    parser.add_argument("--referencia21", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true")
    action.add_argument("--executar", action="store_true")
    action.add_argument("--autoteste", action="store_true")
    args = parser.parse_args()
    if args.autoteste: selftest(); return
    root, out, seeds = args.dados.resolve(), args.saida.resolve(), tuple(args.seeds)
    if not out.is_relative_to(root/"results/experiments") or out == root/"results/experiments": parser.error("Saida precisa ser subpasta de results/experiments")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s > 2**32-1024 for s in seeds): parser.error("Seeds precisam ser unicas e validas para folds internos")
    data20, ref21 = (args.dados20 or root/EXP21.DATA20_REL).resolve(), (args.referencia21 or root/REF21_REL).resolve()
    matrices, meta, names = EXP21.load_inputs(root, data20)
    reference = reference21(ref21, meta, seeds)
    ident = identity(root, data20, ref21, seeds)
    print(f"ENTRADAS22 OK:411pessoas/2465gravacoes;matrizes{matrices['wpt'].shape}/{matrices['wav2vec'].shape};seeds={seeds};30candidatosunicos/innerfold", flush=True)
    if not args.executar: print("Somente verificacao, sem treino ou escrita.", flush=True); return
    signature_file = out/"assinatura_execucao.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf8")) != json.loads(json.dumps(ident)): raise RuntimeError("Configuracao/dados/codigo/ambiente mudaram; preservar pasta")
    else:
        if out.exists() and any(p.name != "PROTOCOLO.txt" for p in out.iterdir()): raise RuntimeError("Pasta sem assinatura contem artefatos; preservar")
        out.mkdir(parents=True, exist_ok=True)
        EXP17.freeze_text(out/"PROTOCOLO.txt", protocol()); EXP17.json_write(signature_file, ident)
    EXP17.freeze_text(out/"PROTOCOLO.txt", protocol())
    EXP17.freeze_text(out/"particoes_participantes.csv", EXP19.fold_table(meta, seeds).to_csv(index=False, lineterminator="\n"))
    signature = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    log = EXP17.Logger(out/"LOG_EXECUCAO.txt")
    log(f"Inicio/retomada22;seeds={seeds};CVinterna3;Cgrid={C_GRID};sem novos atributos/limiar.")
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            destination = out/"checkpoints"/f"seed_{seed}_fold_{fold:02d}"
            if (destination/"CONCLUIDO.json").exists():
                frames, arrays = load_checkpoint(destination, signature); validate(frames, arrays, meta, train, test, names, seed, fold)
                log(f"Checkpoint22 seed={seed} fold={fold} validado; nenhum reajuste"); continue
            if destination.exists() and any(destination.iterdir()): raise RuntimeError(f"Checkpoint parcial preservado:{destination}")
            started = time.monotonic()
            frames, arrays = evaluate_fold(matrices, meta, names, train, test, seed, fold, log, reference21=reference)
            validate(frames, arrays, meta, train, test, names, seed, fold)
            save_checkpoint(destination, frames, arrays, signature)
            log(f"Concluido22 seed={seed} fold={fold};segundos={time.monotonic()-started:.2f}")
    consolidate(out, meta, seeds, reference, log)


if __name__ == "__main__":
    main()
