# -*- coding: utf-8 -*-
"""Exp30: frases + vogal sustentada /a/, uma amostra por participante.

Média das características de frases antes de ANOVA/escala; não há métricas
por gravação. --autoteste usa somente dados sintéticos; --verificar não ajusta.
"""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp28_for_exp30", HERE / "28_avaliar_camadas_ssl_wavelet_avfad.py")
EXP28 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXP28)
EXP22, EXP21, EXP19, EXP17, EXP13 = EXP28.EXP22, EXP28.EXP21, EXP28.EXP19, EXP28.EXP17, EXP28.EXP13
VERSION = "30.1"
SEEDS, LAYERS, C_GRID = EXP28.SEEDS, EXP28.LAYERS, EXP22.C_GRID
OUT_REL = Path("results/experiments/resultados_30_frase_vogal_wavelet")
DATA23_REL = Path("results/experiments/resultados_23_wpt_rms_sequencias")
DATA29_REL = Path("results/experiments/resultados_29_vogal_wavelet_ssl_extracao")
ARMS = ("A_WPT_SENT128", "B_WPT_VOW128", "C_WPT_SENT_VOW256", "D_SSL_SENT128", "E_SSL_VOW128",
        "F_SSL_SENT_VOW256", "G_HYB_SENT_VOW512", "H_SSL_SENT_VOW512", "I_HYB_SENT256")
WPT_REPS = ("WPT_SENT128", "WPT_VOW128", "WPT_SENT_VOW256")
LAYER_KINDS = ("SSL_SENT128", "SSL_VOW128", "SSL_SENT_VOW256", "HYB_SENT_VOW512", "SSL_SENT_VOW512", "HYB_SENT256")
REPRESENTATIONS = WPT_REPS + tuple(f"{kind}_L{layer:02d}" for layer in LAYERS for kind in LAYER_KINDS)
ARM_REPS = {arm: (rep,) for arm, rep in zip(ARMS[:3], WPT_REPS)}
ARM_REPS.update({arm: tuple(f"{kind}_L{layer:02d}" for layer in LAYERS) for arm, kind in zip(ARMS[3:], LAYER_KINDS)})
PREPROCESSORS = {"WPT_SENT128": ("wpt_sent", 128), "WPT_VOW128": ("wpt_vow", 128)}
for _layer in LAYERS:
    for _source in ("SENT", "VOW"):
        for _k in (128, 256):
            PREPROCESSORS[f"SSL_{_source}{_k}_L{_layer:02d}"] = (f"ssl_{_source.lower()}{_layer:02d}", _k)
CSV_FILES = {"predicoes.csv", "predicoes_treino_participante.csv", "metricas.csv", "atributos.csv",
             "candidatos_internos.csv", "predicoes_oof_internas.csv", "particoes_internas.csv", "selecoes.csv", "amostras_suporte.csv"}
ARTIFACTS = CSV_FILES | {"parametros_ajuste.npz"}
CONTRASTS = {
    "C-A_vogal_Wavelet_primaria": {ARMS[2]: 1, ARMS[0]: -1},
    "G-F_contribuicao_Wavelet_multitarefa_primaria": {ARMS[6]: 1, ARMS[5]: -1},
    "F-D_vogal_SSL_secundaria": {ARMS[5]: 1, ARMS[3]: -1},
    "G-I_vogal_hibrida_secundaria": {ARMS[6]: 1, ARMS[8]: -1},
    "G-H_controle_orcamento512_secundaria": {ARMS[6]: 1, ARMS[7]: -1},
    "I-D_Wavelet_frases_secundaria": {ARMS[8]: 1, ARMS[3]: -1},
    "C-B_frases_Wavelet_secundaria": {ARMS[2]: 1, ARMS[1]: -1},
    "F-E_frases_SSL_secundaria": {ARMS[5]: 1, ARMS[4]: -1},
    "B-A_Wavelet_fontes_secundaria": {ARMS[1]: 1, ARMS[0]: -1},
    "G-C_SSL_sobre_Wavelet_multitarefa_secundaria": {ARMS[6]: 1, ARMS[2]: -1},
}


def protocol():
    return """EXPERIMENTO 30 — FRASES + VOGAL /a/, CLASSIFICAÇÃO POR PESSOA
Hipóteses: a vogal sustentada pode acrescentar informação à Wavelet das frases;
Wavelet pode contribuir ao SSL ao combinar tarefas de fala e fonação.
Coorte HSA preservada: 411 participantes, 82 normais/329 patológicos. Cada linha
é UMA PESSOA: média aritmética de características das suas cinco/seis frases
CAPE-V 004–009 e sua única vogal /a/ 002. Não juntar gravações como pacientes.
Média em float64 e saída float32, igual peso 1/n por frase; agregação individual,
sem rótulos ou estatísticas entre pessoas. Depois começa a CV por participante.
WPT RAW a 48 kHz, sym5 L5/L6, 1.920 atributos por gravação; sem busca de RMS.
RAW 23 reproduziu exatamente o cache original; esse mesmo vetor é reutilizado.
SSL XLS-R congelado, camadas 24/5/12/16, 2.048 atributos de média/DP.
As frases vêm dos experimentos 20/27; a vogal vem do 29.
Não usar idade, sexo, ano, local, diagnóstico específico ou gravidade como input.
Esses fatores continuam potenciais confundidores, não são corrigidos pelo modelo.
Imputação mediana/constantes/ANOVA/StandardScaler ajustados somente no treino
de PESSOAS de cada partição. São 18 preprocessadores por treino: WPT de
frases/vogal com 128 atributos; SSL de frases/vogal × 4 camadas × 128/256.
Compartilhar somente dentro do mesmo treino. SVC linear, class_weight=balanced
calculado nas PESSOAS de treino, limiar 0.
A_WPT_SENT128 e B_WPT_VOW128: 128 atributos da respectiva fonte.
C_WPT_SENT_VOW256: 128 WPT de cada fonte, concatenação/sqrt(2).
D_SSL_SENT128 e E_SSL_VOW128: 128 SSL da respectiva fonte.
F_SSL_SENT_VOW256: 128 SSL de cada fonte, concatenação/sqrt(2).
G_HYB_SENT_VOW512: 128 WPT/frases + 128 WPT/vogal + 128 SSL/frases +
128 SSL/vogal, concatenação dividida por 2.
H_SSL_SENT_VOW512: 256 SSL/frases + 256 SSL/vogal, concatenação dividida
por 2, controle de orçamento de 512 atributos.
I_HYB_SENT256: 128 WPT/frases + 128 SSL/frases, concatenação/sqrt(2).
Todos os braços escolhem C em {0.0001,0.001,0.01,0.1,1}, CV interna de
três folds por pessoa. D–I também escolhem camada 24/5/12/16.
Nas fusões, a mesma camada é usada nas
duas fontes. Cada braço tem busca própria; comparam-se pipelines completos,
não um efeito causal isolado mantendo uma camada escolhida externamente.
São 135 candidatos únicos: 3 Wavelet × 5 C + 6 SSL/híbridos × 4 camadas × 5 C.
Cinco folds externos por participante, seed 20261105 principal e 101–105 de
estabilidade. Innerseed=outerseed+100+fold, mesmos folds de pessoas anteriores.
Critério: BA em TODAS as predições OOF internas de pessoas, limiar 0.
Empates até 1e-12: menor C; depois ordem 24/5/12/16. Nenhum teste na escolha.
Ciclo completo: 12.150 ajustes internos e 270 externos. Sem SMOTE, limiar adaptado,
fine-tuning, novas exclusões ou dados SVD. Nenhuma métrica de gravação.
Primários C-A e G-F; secundários F-D,G-I,G-H,I-D,C-B,F-E,B-A,G-C.
Bootstrap com 2.000 reamostragens por pessoa, repetições de seeds no mesmo cluster;
semente principal e cinco de estabilidade separadas. DP entre seeds não é IC.
AUC por fold ponderada por pares; AUC concatenada apenas exploratória.
Unidade de ajuste mudou frente ao 28: agora pessoas, não frases. Não exigir
reprodução do 28 nem atribuir uma diferença com ele somente à nova vogal.
AVFAD já explorada é desenvolvimento, não teste independente; ICs condicionais
não refazem seleção/treino nem corrigem múltiplas tentativas históricas.
Salvar agregação/entradas/partições/OOFs/candidatos/escolhas/atributos/modelos,
hashes/checkpoints/fontes congeladas. Nenhum resultado negativo é omitido.
Sem promessa de 80–90% ou afirmação de contribuição Wavelet sem contraste.
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def layer_of(rep):
    return int(rep.rsplit("_L", 1)[1]) if "_L" in rep else 0


def branches_for(rep):
    if rep in ("WPT_SENT128", "WPT_VOW128"):
        return (rep,)
    if rep == "WPT_SENT_VOW256":
        return ("WPT_SENT128", "WPT_VOW128")
    layer = layer_of(rep); require(layer in LAYERS, "Camada desconhecida")
    kind = rep.rsplit("_L", 1)[0]
    ssl_s, ssl_v = f"SSL_SENT128_L{layer:02d}", f"SSL_VOW128_L{layer:02d}"
    choices = {"SSL_SENT128": (ssl_s,), "SSL_VOW128": (ssl_v,), "SSL_SENT_VOW256": (ssl_s, ssl_v),
               "HYB_SENT_VOW512": ("WPT_SENT128", "WPT_VOW128", ssl_s, ssl_v),
               "SSL_SENT_VOW512": (f"SSL_SENT256_L{layer:02d}", f"SSL_VOW256_L{layer:02d}"),
               "HYB_SENT256": ("WPT_SENT128", ssl_s)}
    require(kind in choices, "Representação desconhecida")
    return choices[kind]


def represented(values, rep):
    branches = branches_for(rep)
    if len(branches) == 1:
        return values[branches[0]]
    divisor = 2. if rep.startswith(("HYB_SENT_VOW512", "SSL_SENT_VOW512")) else np.sqrt(2.)
    return np.concatenate([values[branch] for branch in branches], axis=1) / divisor


def candidate_id(rep, c):
    return f"{rep}_C{float(c):g}"


def validate_people(meta, matrices=None, names=None, real=True):
    required = {"participant_id", "y", "matrix_row", "n_sentences", "vowel_sample_id"}
    require(required.issubset(meta) and not meta[list(required)].isna().any().any(), "Metadados de pessoas incompletos")
    require(not meta.participant_id.duplicated().any() and not meta.vowel_sample_id.duplicated().any(), "Pessoa ou vogal duplicada")
    require(set(meta.y.unique()) == {0, 1} and np.array_equal(meta.matrix_row, np.arange(len(meta))), "Rótulos/ordem de pessoas inválidos")
    require(meta.participant_id.astype(str).is_monotonic_increasing, "Pessoas devem estar ordenadas por PID")
    require((meta.n_sentences > 0).all() and meta.n_sentences.eq(meta.n_sentences.astype(int)).all(), "Contagem de frases inválida")
    require(np.array_equal(meta.vowel_sample_id, meta.participant_id.astype(str)+"002"), "Vogal não é tarefa002 da mesma pessoa")
    if real:
        require((len(meta), meta.y.value_counts().to_dict()) == (411, {1: 329, 0: 82}), "Coorte HSA411/82/329 alterada")
        require(meta.n_sentences.value_counts().to_dict() == {6: 410, 5: 1}, "Número de frases da coorte mudou")
    if matrices is not None:
        expected = {modality for modality, _ in PREPROCESSORS.values()}
        require(set(matrices) == expected and set(names) == expected, "Modalidades acústicas incompletas ou extras")
        for modality in sorted(expected):
            width = 1920 if modality.startswith("wpt_") else 2048
            x = matrices[modality]
            require(x.shape == (len(meta), width) and x.dtype == np.float32 and np.isfinite(x).all(), f"Matriz inválida: {modality}")
            require(len(names[modality]) == width and len(set(names[modality])) == width, "Nomes de atributos inválidos")
            require(all(str(name).startswith("SENT__" if "sent" in modality else "VOW__") for name in names[modality]), "Fonte ausente nos nomes")


def aggregate_sentences(matrices, meta, names, real=True):
    require({"participant_id", "sample_id", "task", "y"}.issubset(meta), "Manifesto das frases incompleto")
    require(not meta.sample_id.duplicated().any() and not meta.duplicated(["participant_id", "task"]).any(), "Frase duplicada")
    require(meta.groupby("participant_id").y.nunique().eq(1).all(), "Rótulo inconsistente entre frases")
    if real:
        require(len(meta) == 2465 and set(meta.task.astype(str).str.zfill(3)) == {"004", "005", "006", "007", "008", "009"}, "Tarefas/coorte de frases diferentes")
    require(set(matrices) == {"wpt", "ssl24", "ssl05", "ssl12", "ssl16"}, "Modalidades das frases inesperadas")
    people = EXP13.person_table(meta)
    people["matrix_row"] = np.arange(len(people)); people["vowel_sample_id"] = people.participant_id.astype(str)+"002"
    groups = {pid: np.flatnonzero(meta.participant_id.to_numpy() == pid) for pid in people.participant_id}
    people["n_sentences"] = [len(groups[pid]) for pid in people.participant_id]
    result, new_names, mapping = {}, {}, []
    for modality, matrix in matrices.items():
        key = "wpt_sent" if modality == "wpt" else "ssl_sent"+modality[3:]
        require(matrix.shape[0] == len(meta) and np.isfinite(matrix).all(), "Features das frases inválidas")
        result[key] = np.stack([np.mean(matrix[groups[pid]], axis=0, dtype=np.float64).astype(np.float32) for pid in people.participant_id])
        new_names[key] = np.asarray(["SENT__"+str(name) for name in names[modality]])
    for row in people.itertuples():
        for index in groups[row.participant_id]:
            record = meta.iloc[index]
            mapping.append({"participant_id": row.participant_id, "person_row": row.matrix_row,
                            "source_row": int(index), "sample_id": record.sample_id, "task": str(record.task).zfill(3),
                            "weight": 1./row.n_sentences, "n_sentences": row.n_sentences})
    validate_people(people, real=real)
    return result, people, new_names, pd.DataFrame(mapping)


def prepare(matrices, y, train, validation):
    train_values, validation_values, arrays = {}, {}, {}
    for branch, (modality, k) in PREPROCESSORS.items():
        pre = EXP13.pipeline_for("SVC_linear", k, 1.)[:-1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pre.fit(matrices[modality][train], y[train])
        train_values[branch] = pre.transform(matrices[modality][train])
        validation_values[branch] = pre.transform(matrices[modality][validation])
        params = {**EXP22.pre_parameters(pre), "train_rows": np.asarray(train, np.int64)}
        arrays.update({f"{branch}__{name}": value for name, value in params.items()})
    return train_values, validation_values, arrays


def choose(candidates, allowed):
    eligible = candidates[candidates.representation.isin(allowed)].copy()
    require(not eligible.empty and np.isfinite(eligible.inner_BA).all(), "Candidatos ausentes ou não finitos")
    tied = eligible[eligible.inner_BA >= eligible.inner_BA.max()-1e-12].copy()
    tied["layer_order"] = tied.layer.map({0: -1, **{layer: i for i, layer in enumerate(LAYERS)}})
    require(tied.layer_order.notna().all(), "Camada desconhecida na seleção")
    return tied.sort_values(["C", "layer_order", "candidate_id"], kind="stable").iloc[0].drop(labels="layer_order")


def predictions(meta, rows, scores, arm, seed, fold):
    frame = meta.iloc[rows][["participant_id", "y", "n_sentences", "vowel_sample_id"]].copy().reset_index(drop=True)
    frame["score"], frame["predicao"] = scores, (np.asarray(scores) >= 0).astype(int)
    frame["arm"], frame["seed"], frame["fold"], frame["limiar"] = arm, seed, fold, 0.
    return frame


def inner_search(matrices, meta, train, seed, fold, c_grid=C_GRID, log=lambda _: None):
    inner_meta = meta.iloc[train].reset_index(drop=True); y = meta.y.to_numpy(); inner_seed = int(seed)+100+int(fold)
    records, partitions, arrays = [], [], {}
    for inner_fold, (itr, iva) in enumerate(EXP13.row_splits(inner_meta, 3, inner_seed), 1):
        tr, va = train[itr], train[iva]
        require(not set(meta.iloc[tr].participant_id) & set(meta.iloc[va].participant_id), "Vazamento interno")
        for role, rows in (("treino", tr), ("validacao", va)):
            for index in rows:
                person = meta.iloc[index]
                partitions.append({"seed": seed, "fold": fold, "inner_seed": inner_seed, "inner_fold": inner_fold,
                                   "role": role, "matrix_row": int(index), "participant_id": person.participant_id, "y": person.y})
        values_tr, values_va, learned = prepare(matrices, y, tr, va)
        arrays.update({f"inner{inner_fold}__{key}": value for key, value in learned.items()})
        for rep in REPRESENTATIONS:
            xtr, xva = represented(values_tr, rep), represented(values_va, rep)
            for c in c_grid:
                model = EXP22.fit_classifier(xtr, y[tr], c)
                pred = predictions(meta, va, model.decision_function(xva), "candidate", seed, fold)
                pred["candidate_id"], pred["representation"], pred["C"], pred["layer"] = candidate_id(rep, c), rep, float(c), layer_of(rep)
                pred["inner_fold"] = inner_fold; records.append(pred)
        log(f"Busca30 seed={seed} fold={fold} inner={inner_fold}: {len(REPRESENTATIONS)*len(c_grid)} candidatos; amostra=pessoa")
    inner = pd.concat(records, ignore_index=True); candidate_rows = []
    for cid, sub in inner.groupby("candidate_id", sort=False):
        require(not sub.participant_id.duplicated().any() and set(sub.participant_id) == set(inner_meta.participant_id), "OOF interno incompleto")
        first, measured = sub.iloc[0], EXP17.measures(sub)
        candidate_rows.append({"seed": seed, "fold": fold, "candidate_id": cid, "representation": first.representation,
                               "C": float(first.C), "layer": int(first.layer), "n_participantes": len(sub),
                               "inner_BA": measured["BA"], "inner_sensibilidade": measured["sensibilidade"],
                               "inner_especificidade": measured["especificidade"], "inner_acuracia": measured["acuracia"]})
    candidates = pd.DataFrame(candidate_rows)
    choices = pd.DataFrame([{**choose(candidates, allowed).to_dict(), "arm": arm} for arm, allowed in ARM_REPS.items()])
    return {"candidatos_internos.csv": candidates, "predicoes_oof_internas.csv": inner,
            "particoes_internas.csv": pd.DataFrame(partitions), "selecoes.csv": choices}, arrays


def evaluate_fold(matrices, meta, names, train, test, seed, fold, log=lambda _: None, c_grid=C_GRID):
    validate_people(meta, matrices, names, real=False)
    require(not set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id), "Vazamento externo por pessoa")
    frames, arrays = inner_search(matrices, meta, train, seed, fold, c_grid, log)
    y = meta.y.to_numpy(); values_tr, values_te, learned = prepare(matrices, y, train, test)
    arrays.update({f"outer__{key}": value for key, value in learned.items()})
    output, fitted, metrics, attrs, supports = [], [], [], [], []
    for row in frames["selecoes.csv"].itertuples():
        arm, rep, c = row.arm, row.representation, float(row.C)
        model = EXP22.fit_classifier(represented(values_tr, rep), y[train], c)
        pred = predictions(meta, test, model.decision_function(represented(values_te, rep)), arm, seed, fold)
        fit = predictions(meta, train, model.decision_function(represented(values_tr, rep)), arm, seed, fold)
        for phase, frame in (("teste", pred), ("treino", fit)):
            metrics.append({"seed": seed, "fold": fold, "arm": arm, "phase": phase, "unit": "participante",
                            "C": c, "layer": layer_of(rep), **EXP17.measures(frame)})
        arrays.update({f"{arm}__{key}": value for key, value in EXP21.classifier_parameters(model, train).items()})
        arrays.update({f"{arm}__C": np.asarray(c), f"{arm}__layer": np.asarray(layer_of(rep)), f"{arm}__representation": np.asarray(rep)})
        for branch in branches_for(rep):
            modality, _ = PREPROCESSORS[branch]; var, sel = learned[f"{branch}__constant_mask"], learned[f"{branch}__selector_mask"]
            attrs.extend({"seed": seed, "fold": fold, "arm": arm, "branch": branch, "modality": modality,
                          "feature": str(name)} for name in names[modality][var][sel])
        supports.extend({"seed": seed, "fold": fold, "arm": arm, "person_row": int(train[i]),
                         "participant_id": meta.iloc[train[i]].participant_id, "y": int(y[train[i]])} for i in model.support_)
        output.append(pred); fitted.append(fit)
        log(f"30 seed={seed} fold={fold} {arm}: camada={layer_of(rep)} C={c:g}; BA teste={EXP17.measures(pred)['BA']:.6f}; treino={EXP17.measures(fit)['BA']:.6f}")
    frames.update({"predicoes.csv": pd.concat(output, ignore_index=True), "predicoes_treino_participante.csv": pd.concat(fitted, ignore_index=True),
                   "metricas.csv": pd.DataFrame(metrics), "atributos.csv": pd.DataFrame(attrs), "amostras_suporte.csv": pd.DataFrame(supports)})
    return frames, arrays


def validate(frames, arrays, meta, train, test, names, seed, fold, c_grid=C_GRID):
    validate_people(meta, real=False)
    require(set(frames) == CSV_FILES and not set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id), "Checkpoint incompleto/vazamento externo")
    candidates, choices, inner = frames["candidatos_internos.csv"], frames["selecoes.csv"], frames["predicoes_oof_internas.csv"]
    expected_cids = {candidate_id(rep, c) for rep in REPRESENTATIONS for c in c_grid}
    require(len(candidates) == len(expected_cids) and set(candidates.candidate_id) == expected_cids and not candidates.candidate_id.duplicated().any(), "Candidatos incompletos")
    require(len(choices) == len(ARMS) and set(choices.arm) == set(ARMS), "Escolhas incompletas")
    labels = meta.set_index("participant_id").y; train_ids = set(meta.iloc[train].participant_id)
    partitions = frames["particoes_internas.csv"]; expected_inner = {}
    for inner_fold, (itr, iva) in enumerate(EXP13.row_splits(meta.iloc[train].reset_index(drop=True), 3, int(seed)+100+fold), 1):
        sub = partitions[partitions.inner_fold == inner_fold]
        for role, rows in (("treino", train[itr]), ("validacao", train[iva])):
            actual = sub[sub.role == role]
            require(len(actual) == len(rows) and set(actual.matrix_row) == set(rows), "Partição interna incorreta")
            original = meta.iloc[actual.matrix_row.to_numpy(dtype=int)]
            np.testing.assert_array_equal(actual.participant_id, original.participant_id); np.testing.assert_array_equal(actual.y, original.y)
        require(not set(sub[sub.role == "treino"].participant_id) & set(sub[sub.role == "validacao"].participant_id), "Vazamento interno")
        expected_inner[inner_fold] = set(meta.iloc[train[iva]].participant_id)
        for branch in PREPROCESSORS:
            np.testing.assert_array_equal(arrays[f"inner{inner_fold}__{branch}__train_rows"], train[itr])
    require(set(partitions.inner_fold) == {1, 2, 3} and partitions.seed.eq(seed).all() and partitions.fold.eq(fold).all() and partitions.inner_seed.eq(int(seed)+100+fold).all(), "Identidade da partição inválida")
    require(set(inner.candidate_id) == expected_cids, "OOF interno incompleto")
    for row in candidates.itertuples():
        sub = inner[inner.candidate_id == row.candidate_id]
        require(not sub.participant_id.duplicated().any() and set(sub.participant_id) == train_ids, "OOF interno omite/duplica pessoas")
        require(np.array_equal(sub.y, sub.participant_id.map(labels)) and np.isfinite(sub.score).all() and np.array_equal(sub.predicao, (sub.score >= 0).astype(int)), "Score/rótulo interno inválido")
        require(all(set(sub[sub.inner_fold == i].participant_id) == ids for i, ids in expected_inner.items()), "Fold OOF interno incorreto")
        require(sub.C.eq(row.C).all() and sub.layer.eq(row.layer).all() and sub.representation.eq(row.representation).all() and row.layer == layer_of(row.representation) and row.candidate_id == candidate_id(row.representation, row.C), "Identidade do candidato inconsistente")
        measured = EXP17.measures(sub)
        for key, value in (("inner_BA", measured["BA"]), ("inner_sensibilidade", measured["sensibilidade"]), ("inner_especificidade", measured["especificidade"]), ("inner_acuracia", measured["acuracia"])):
            require(np.isclose(getattr(row, key), value, atol=1e-12, rtol=0), "Métrica interna inconsistente")
        require(row.n_participantes == len(sub), "Contagem de pessoas interna incorreta")
    for row in choices.itertuples():
        chosen = choose(candidates, ARM_REPS[row.arm])
        for key in ("candidate_id", "representation", "C", "layer"):
            require(getattr(row, key) == chosen[key], "Escolha não determinada pela CV interna")
        require(np.isclose(row.inner_BA, chosen.inner_BA, atol=1e-12, rtol=0), "BA da escolha inconsistente")
        for key in ("representation", "C", "layer"):
            require(arrays[f"{row.arm}__{key}"].item() == getattr(row, key), "Modelo não corresponde à escolha")
        support = arrays[f"{row.arm}__support_indices"]
        require((support >= 0).all() and (support < len(train)).all(), "Suporte fora do treino")
        np.testing.assert_array_equal(arrays[f"{row.arm}__support_original_rows"], train[support])
        expected_features = []
        for branch in branches_for(row.representation):
            modality, k = PREPROCESSORS[branch]
            var, sel = arrays[f"outer__{branch}__constant_mask"].astype(bool), arrays[f"outer__{branch}__selector_mask"].astype(bool)
            require(len(var) == len(names[modality]) and len(sel) == var.sum() and sel.sum() == k, "Seleção de atributos inválida")
            expected_features.extend(names[modality][var][sel])
        require(list(frames["atributos.csv"].query("arm == @row.arm").feature) == list(expected_features), "Atributos exportados inconsistentes")
        actual = frames["amostras_suporte.csv"].query("arm == @row.arm")
        np.testing.assert_array_equal(actual.person_row, train[support]); np.testing.assert_array_equal(actual.participant_id, meta.iloc[train[support]].participant_id)
        np.testing.assert_array_equal(actual.y, meta.iloc[train[support]].y)
        expected_weights = len(train)/(2*np.bincount(meta.iloc[train].y.to_numpy(dtype=int), minlength=2))
        np.testing.assert_allclose(arrays[f"{row.arm}__class_weight"], expected_weights, atol=1e-15, rtol=0)
    for branch in PREPROCESSORS:
        np.testing.assert_array_equal(arrays[f"outer__{branch}__train_rows"], train)
    for filename, rows in (("predicoes.csv", test), ("predicoes_treino_participante.csv", train)):
        frame = frames[filename]; require(set(frame.arm) == set(ARMS), "Braços incompletos nas predições")
        for arm, sub in frame.groupby("arm"):
            require(not sub.participant_id.duplicated().any() and set(sub.participant_id) == set(meta.iloc[rows].participant_id), "Pessoas omitidas/duplicadas nas predições")
            require(sub.seed.eq(seed).all() and sub.fold.eq(fold).all() and np.array_equal(sub.y, sub.participant_id.map(labels)), "Partição/rótulo externo incorreto")
            require(np.isfinite(sub.score).all() and np.array_equal(sub.predicao, (sub.score >= 0).astype(int)), "Score externo inválido")
            original = meta.set_index("participant_id").loc[sub.participant_id]
            np.testing.assert_array_equal(sub.n_sentences, original.n_sentences); np.testing.assert_array_equal(sub.vowel_sample_id, original.vowel_sample_id)
    metrics = frames["metricas.csv"]
    require(len(metrics) == 2*len(ARMS) and not metrics.duplicated(["arm", "phase", "unit"]).any() and metrics.unit.eq("participante").all(), "Métricas por pessoa incompletas")
    for row in metrics.itertuples():
        frame = frames["predicoes.csv" if row.phase == "teste" else "predicoes_treino_participante.csv"].query("arm == @row.arm")
        for key, value in EXP17.measures(frame).items():
            require(np.isclose(value, getattr(row, key), atol=1e-12, rtol=0), "Métrica externa inconsistente")


def verify_exports(folder, required=(), selected_only=False):
    marker = json.loads((folder / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    require(isinstance(marker, dict) and bool(marker) and set(required).issubset(marker), "Exports incompletos")
    for name, digest in marker.items():
        require(Path(name).name == name and name not in (".", ".."), "Caminho inválido no marcador")
        if not selected_only or name in required:
            require(EXP17.sha(folder / name) == digest, "Export alterado")
    return marker


def verify_signed_inputs(ident):
    for name, digest in ident["code"].items():
        require(EXP17.sha(name) == digest, "Fonte assinada da extração mudou")
    for name, info in ident["inputs"].items():
        require(Path(name).stat().st_size == info["size_bytes"] and EXP17.sha(name) == info["sha256"], "Insumo assinado mudou")
    for name, info in ident["raw_zip_metadata_only"].items():
        stat = Path(name).stat()
        require(stat.st_size == info["size_bytes"] and stat.st_mtime_ns == info["mtime_ns"], "ZIP original mudou")


def validate_vowel(matrices, people, names, meta29, payload, ident, real=True):
    config = ident.get("configuration", {})
    expected = {"task": "002", "task_definition": "sustained_a_all_recorded_repetitions_VAD_not_Praat_segment",
                "model_id": EXP21.MODEL_ID, "model_revision": "1a640f32ac3e39899438a2931f9924c02f080a54",
                "frozen": True, "sample_rate": 16000, "hidden_layers": [24, 5, 12, 16], "pooling": "mean_std",
                "wavelet_raw_rms": True, "wavelet": "sym5", "wavelet_levels": [5, 6], "wavelet_source_rate": 48000,
                "wavelet_windows": [.4, .2], "wavelet_summaries": ["mean", "std", "q10", "q90"], "wavelet_RMS_target": .1,
                "control_reference": "new_audio_pilot_forward27_vs20_no_sentence_equivalence", "full_requires_signed_pilot": True}
    require(ident.get("mode") == "completo" and all(config.get(key) == value for key, value in expected.items()), "Configuração29 incompatível")
    expected_dimensions = {"wavelet_raw": 1920, "wavelet_rms": 1920, **{f"embeddings_layer{layer:02d}": 2048 for layer in LAYERS}}
    require(config.get("feature_dimension") == 12032 and config.get("feature_dimensions") == expected_dimensions, "Dimensões29 incorretas")
    fields = {"sample_id", "participant_id", "task", "y", "matrix_row"}
    require(fields.issubset(meta29) and not meta29[list(fields)].isna().any().any(), "Metadata29 incompleta")
    require(not meta29.sample_id.duplicated().any() and not meta29.participant_id.duplicated().any(), "Pessoa/vogal29 duplicada")
    require(np.array_equal(meta29.matrix_row, np.arange(len(meta29))) and meta29.task.astype(str).str.zfill(3).eq("002").all(), "Ordem/tarefa29 incorreta")
    require(set(meta29.participant_id) == set(people.participant_id), "Participantes de frases e vogal diferentes")
    require(np.array_equal(meta29.sample_id, meta29.participant_id.astype(str)+"002"), "Vogal29 não corresponde ao PID")
    labels = people.set_index("participant_id").y
    require(np.array_equal(meta29.y, meta29.participant_id.map(labels)), "Rótulos29 diferentes dos das frases")
    require(ident.get("sample_ids") == meta29.sample_id.tolist(), "Manifesto29 diferente da metadata")
    for key in ("sample_id", "participant_id", "task"):
        require(key in payload and np.array_equal(np.asarray(payload[key]).astype(str), meta29[key].astype(str)), "IDs NPZ29 inconsistentes")
    require(np.array_equal(payload["hidden_layer_indices"], LAYERS), "Ordem das camadas29 incorreta")
    for key, width in expected_dimensions.items():
        require(payload[key].dtype == np.float32 and payload[key].shape == (len(meta29), width) and np.isfinite(payload[key]).all(), "Vetor29 inválido")
    expected_wpt_names = np.asarray([str(name).removeprefix("SENT__") for name in names["wpt_sent"]])
    require(np.array_equal(payload["wavelet_feature_names"].astype(str), expected_wpt_names), "Nomes WPT29 diferentes das frases")
    ordering = meta29.set_index("participant_id").loc[people.participant_id].matrix_row.to_numpy(dtype=int)
    matrices["wpt_vow"] = payload["wavelet_raw"][ordering].copy()
    names["wpt_vow"] = np.asarray(["VOW__"+str(name) for name in payload["wavelet_feature_names"]])
    for layer in LAYERS:
        source_names = EXP21.embedding_names() if layer == 24 else EXP28.intermediate_names(layer)
        require(np.array_equal(payload[f"feature_names_layer{layer:02d}"].astype(str), source_names), "Nomes SSL29 inválidos")
        key = f"ssl_vow{layer:02d}"
        matrices[key] = payload[f"embeddings_layer{layer:02d}"][ordering].copy()
        names[key] = np.asarray(["VOW__"+str(name) for name in source_names])
    validate_people(people, matrices, names, real=real)
    return matrices, names


def load_inputs(root=ROOT, data20=None, data23=None, data27=None, data29=None):
    root = Path(root).resolve()
    data20 = Path(data20 or root / EXP21.DATA20_REL).resolve(); data23 = Path(data23 or root / DATA23_REL).resolve()
    data27 = Path(data27 or root / EXP28.DATA27_REL).resolve(); data29 = Path(data29 or root / DATA29_REL).resolve()
    sentence_matrices, sentence_meta, sentence_names = EXP28.load_inputs(root, data20, data27)
    # As sequências23/SQLite não entram: conferir somente os exports utilizados
    # e vincular o marcador completo já auditado, sem reler419MB de sequências.
    verify_exports(data23, ("CONTROLE_REFERENCIA_48KHZ.csv", "metadata_wpt_rms.csv", "assinatura_execucao.json"), selected_only=True)
    identity23 = json.loads((data23 / "assinatura_execucao.json").read_text(encoding="utf8")); verify_signed_inputs(identity23)
    require(identity23.get("mode") == "completo", "Extração23 incompleta")
    meta23 = pd.read_csv(data23 / "metadata_wpt_rms.csv", dtype={"sample_id": str, "participant_id": str, "task": str}, float_precision="round_trip")
    meta23["task"] = meta23.task.str.zfill(3)
    for key in ("sample_id", "participant_id", "task", "y"):
        np.testing.assert_array_equal(meta23[key], sentence_meta[key])
    reference23 = pd.read_csv(data23 / "CONTROLE_REFERENCIA_48KHZ.csv", float_precision="round_trip")
    np.testing.assert_array_equal(reference23.sample_id, sentence_meta.sample_id)
    require(np.array_equal(reference23.source_row, np.arange(len(sentence_meta))) and reference23.equivalent.eq(True).all() and reference23.max_abs_error.eq(0.).all(), "RAW legado não validado exatamente pelo23")
    matrices, people, names, aggregation = aggregate_sentences(sentence_matrices, sentence_meta, sentence_names)
    required29 = {"features_vogal_wavelet_ssl.npz", "metadata_vogal_wavelet_ssl.csv", "QUALIDADE_AUDIO.csv", "CONTROLE_PILOTO_SSL24.csv",
                  "assinatura_execucao.json", "PROTOCOLO.txt", "RESUMO_EXTRACAO.txt"}
    require(set(verify_exports(data29, required29)) == required29, "Extração29 sem todos os sete exports")
    ident = json.loads((data29 / "assinatura_execucao.json").read_text(encoding="utf8")); verify_signed_inputs(ident)
    pilot = ident.get("pilot_reference", {})
    require(set(pilot) == {"directory", "marker_sha256"} and EXP17.sha(Path(pilot["directory"]) / "ARTEFATOS_COMPLETOS.json") == pilot["marker_sha256"], "Referência do piloto29 inválida")
    verify_exports(Path(pilot["directory"]), required29)
    meta29 = pd.read_csv(data29 / "metadata_vogal_wavelet_ssl.csv", dtype={"sample_id": str, "participant_id": str, "task": str}, float_precision="round_trip")
    meta29["task"] = meta29.task.str.zfill(3)
    with np.load(data29 / "features_vogal_wavelet_ssl.npz", allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    matrices, names = validate_vowel(matrices, people, names, meta29, payload, ident)
    return matrices, people, names, aggregation


def save_checkpoint(destination, frames, arrays, signature):
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        EXP17.csv_write(destination / name, frame)
    stream = io.BytesIO(); np.savez_compressed(stream, **arrays)
    EXP17.atomic_bytes(destination / "parametros_ajuste.npz", stream.getvalue())
    EXP17.json_write(destination / "CONCLUIDO.json", {"finished": EXP17.now(), "run_signature": signature,
                     "sha256": {name: EXP17.sha(destination / name) for name in ARTIFACTS}})


def load_checkpoint(destination, signature):
    marker = json.loads((destination / "CONCLUIDO.json").read_text(encoding="utf8"))
    require(marker["run_signature"] == signature and set(marker["sha256"]) == ARTIFACTS, "Checkpoint incompleto/de outra configuração")
    require(all(EXP17.sha(destination / name) == digest for name, digest in marker["sha256"].items()), "Checkpoint alterado")
    frames = {name: pd.read_csv(destination / name, float_precision="round_trip") for name in CSV_FILES}
    with np.load(destination / "parametros_ajuste.npz", allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    return frames, arrays


def identity(root, data20, data23, data27, data29, seeds):
    inputs = [root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz",
              root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv"]
    for folder in (data20, data23, data27, data29):
        names = json.loads((folder / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
        # A sequência23 de419MB não é usada: RAW legado+controle vinculado bastam.
        if folder == data23:
            names = {name: digest for name, digest in names.items() if name in ("metadata_wpt_rms.csv", "CONTROLE_REFERENCIA_48KHZ.csv", "assinatura_execucao.json")}
        inputs.extend([folder / name for name in names]); inputs.append(folder / "ARTEFATOS_COMPLETOS.json")
    code_names = ("30_avaliar_frase_vogal_wavelet_avfad.py", "29_extrair_vogal_wavelet_ssl_avfad.py", "28_avaliar_camadas_ssl_wavelet_avfad.py",
                  "27_extrair_camadas_ssl_avfad.py", "23_extrair_wpt_rms_sequencias_avfad.py", "22_regularizar_wpt_wav2vec_avfad.py",
                  "21_avaliar_wpt_wav2vec_avfad.py", "20_extrair_wav2vec_avfad.py", "19_avaliar_wpt_16khz_avfad.py",
                  "17_robustez_distribuicao_wpt_avfad.py", "13_otimizar_wpt_sym5.py")
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
        environment[package] = importlib.metadata.version(package)
    return {"version": VERSION, "configuration": {"arms": list(ARMS), "seeds": list(seeds), "layers": list(LAYERS), "C_grid": list(C_GRID),
            "fit_unit": "participant", "sentence_aggregation": "arithmetic_equal_recording_mean_float64_to_float32_before_CV",
            "wavelet_input": "RAW_legacy_validated_exactly23", "vowel_task": "002", "outer_folds": 5, "inner_folds": 3,
            "inner_seed": "outer_seed+100+fold", "class_balance": "balanced_person_train_only", "threshold": 0.,
            "n_preprocessors_per_train": 18, "n_candidates_per_inner": 135, "selection": "pooled_inner_OOF_BA_person",
            "tie": "BAwithin1e-12;lowerC;layer_order24_5_12_16", "bootstrap_iterations": 2000, "clinical_inputs": False},
            "environment": environment, "input_files": {str(path.resolve()): EXP17.sha(path) for path in inputs},
            "code": {str((HERE / name).resolve()): EXP17.sha(HERE / name) for name in code_names},
            "protocol_sha256": hashlib.sha256(protocol().encode("utf8")).hexdigest()}


def freeze_sources(out, ident):
    destination = out / "codigo_congelado"; destination.mkdir(parents=True, exist_ok=True); hashes = {}
    for source, digest in ident["code"].items():
        target = destination / Path(source).name
        require(EXP17.sha(source) == digest, "Fonte mudou antes de congelar")
        if target.exists():
            require(EXP17.sha(target) == digest, "Cópia congelada diferente; preservar")
        else:
            with target.open("xb") as stream: stream.write(Path(source).read_bytes())
        hashes[target.name] = digest
    path = out / "CODIGO_CONGELADO.json"
    if path.exists():
        require(json.loads(path.read_text(encoding="utf8")) == hashes, "Manifesto congelado diverge")
    else:
        EXP17.json_write(path, hashes)


def consolidate(out, meta, seeds, log):
    destinations = [out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" for seed in seeds for fold in range(1, 6)]
    frames = {name: pd.concat([pd.read_csv(path / name, float_precision="round_trip") for path in destinations], ignore_index=True)
              for name in ("predicoes.csv", "metricas.csv", "atributos.csv", "candidatos_internos.csv", "selecoes.csv", "amostras_suporte.csv")}
    raw, folds = frames["predicoes.csv"], frames["metricas.csv"]; summaries = []
    for (seed, arm), group in raw.groupby(["seed", "arm"]):
        require(not group.participant_id.duplicated().any() and set(group.participant_id) == set(meta.participant_id), "OOF externo incompleto")
        measured = EXP17.measures(group); measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        sub = folds[(folds.seed == seed) & (folds.arm == arm) & (folds.phase == "teste")]
        measured["AUC_media_folds_ponderada_pares"] = float(np.average(sub.AUC, weights=(sub.TN+sub.FP)*(sub.TP+sub.FN)))
        summaries.append({"seed": seed, "arm": arm, **measured})
    summary = pd.DataFrame(summaries)
    scopes = [("principal_seed20261105", [20261105])] if 20261105 in seeds else []
    secondary = [seed for seed in seeds if seed != 20261105]
    if secondary: scopes.append(("estabilidade_secundaria", secondary))
    comparisons = [{"escopo": scope, "comparacao": name, "seeds": str(chosen),
                    **EXP17.intervals(raw, chosen, coefficients, iterations=2000)}
                   for scope, chosen in scopes for name, coefficients in CONTRASTS.items()]
    exports = {"predicoes_oof_participante.csv": raw, "metricas_por_fold.csv": folds, "metricas_por_seed.csv": summary,
               "comparacoes_pareadas.csv": pd.DataFrame(comparisons), "atributos_selecionados.csv": frames["atributos.csv"],
               "candidatos_internos.csv": frames["candidatos_internos.csv"], "selecoes_internas.csv": frames["selecoes.csv"],
               "amostras_suporte.csv": frames["amostras_suporte.csv"]}
    for name, frame in exports.items(): EXP17.csv_write(out / name, frame)
    lines = ["EXPERIMENTO 30 — FRASES + VOGAL /a/, POR PARTICIPANTE", f"Consolidado: {EXP17.now()}",
             "Todos os folds solicitados concluídos. Uma amostra de entrada por pessoa.",
             "Ciclo completo: nove braços, seis sementes, 30 folds externos." if tuple(seeds) == SEEDS else "Ciclo parcial/diferente das seis sementes.", ""]
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
    for keys, count in frames["selecoes.csv"].groupby(["arm", "layer", "C"]).size().items(): lines.append(f"{keys}: {count} folds")
    lines += ["", "DIAGNÓSTICO DE TREINO"]
    for arm, group in folds[folds.phase == "treino"].groupby("arm"): lines.append(f"{arm}: BA treino média={group.BA.mean():.6f}")
    lines += ["", "LIMITAÇÕES", "Agora a unidade de ajuste é a pessoa: médias das características das frases antes de ANOVA/escala.",
              "Mudança de unidade impede atribuir diferenças com28 exclusivamente à vogal. Não se exige reprodução de28.",
              "A vogal contém todas as emissões retidas pelo VAD; não usa o intervalo Praat anotado no legado.",
              "RAW foi predefinido. A extração RMS29 existe, mas não integra a busca desta rodada.",
              "Camada compartilhada entre fontes limita a busca; cada braço seleciona sua própria camada/C no treino interno.",
              "G-F e G-H são necessários para demonstrar incremento Wavelet; ganhar de Wavelet isolada não o demonstra.",
              "Idade, sexo, ano e clínica não entram como features; confundimento de aquisição continua possível.",
              "Pooling por média perde ordem temporal e ignora diferenças de relevância entre frases.",
              "AVFAD já explorada: desenvolvimento, não teste independente. Seis sementes não são seis novas coortes.",
              "ICs condicionais não refazem modelos/seleção nem corrigem múltiplas tentativas históricas.",
              "Nenhuma validação SVD, adaptação do encoder, promessa de 80–90% ou conclusão clínica nesta rodada.\n"]
    EXP17.text_write(out / "RELATORIO_RESULTADOS.txt", "\n".join(lines))
    names = list(exports) + ["RELATORIO_RESULTADOS.txt", "assinatura_execucao.json", "particoes_participantes.csv", "PROTOCOLO.txt",
                             "entradas_participantes.csv", "agregacao_frases.csv", "CODIGO_CONGELADO.json"]
    EXP17.json_write(out / "ARTEFATOS_COMPLETOS.json", {name: EXP17.sha(out / name) for name in names})
    log("Consolidação30 concluída; todos os braços/candidatos/negativos preservados.")


def selftest():
    import unittest
    spec = importlib.util.spec_from_file_location("test_frase_vogal30", HERE / "test_frase_vogal30.py")
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful(): raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT); parser.add_argument("--saida", type=Path, default=ROOT / OUT_REL)
    for number in (20, 23, 27, 29): parser.add_argument(f"--dados{number}", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    actions = parser.add_mutually_exclusive_group(); actions.add_argument("--autoteste", action="store_true")
    actions.add_argument("--verificar", action="store_true"); actions.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    if args.autoteste: selftest(); return
    root, out, seeds = args.dados.resolve(), args.saida.resolve(), tuple(args.seeds)
    if not out.is_relative_to(root / "results/experiments") or out == root / "results/experiments": parser.error("Saída precisa ficar em results/experiments")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed > 2**32-1024 for seed in seeds): parser.error("Sementes únicas e válidas obrigatórias")
    paths = {20: (args.dados20 or root / EXP21.DATA20_REL).resolve(), 23: (args.dados23 or root / DATA23_REL).resolve(),
             27: (args.dados27 or root / EXP28.DATA27_REL).resolve(), 29: (args.dados29 or root / DATA29_REL).resolve()}
    matrices, meta, names, aggregation = load_inputs(root, paths[20], paths[23], paths[27], paths[29])
    ident = identity(root, paths[20], paths[23], paths[27], paths[29], seeds)
    print(f"ENTRADAS30 OK: 411 pessoas, uma linha por pessoa; 9 braços, 135 candidatos/inner; seeds={seeds}", flush=True)
    if not args.executar: print("Somente verificação, sem ajuste ou escrita.", flush=True); return
    signature_file = out / "assinatura_execucao.json"
    if signature_file.exists():
        require(json.loads(signature_file.read_text(encoding="utf8")) == ident, "Assinatura30 mudou; preservar artefatos")
    else:
        require(not out.exists() or not any(path.name != "PROTOCOLO.txt" for path in out.iterdir()), "Pasta sem assinatura contém artefatos")
        out.mkdir(parents=True, exist_ok=True); EXP17.freeze_text(out / "PROTOCOLO.txt", protocol()); EXP17.json_write(signature_file, ident)
    EXP17.freeze_text(out / "PROTOCOLO.txt", protocol()); freeze_sources(out, ident)
    EXP17.freeze_text(out / "particoes_participantes.csv", EXP19.fold_table(meta, seeds).to_csv(index=False, lineterminator="\n"))
    EXP17.freeze_text(out / "entradas_participantes.csv", meta.to_csv(index=False, lineterminator="\n"))
    EXP17.freeze_text(out / "agregacao_frases.csv", aggregation.to_csv(index=False, lineterminator="\n"))
    signature = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest(); log = EXP17.Logger(out / "LOG_EXECUCAO.txt")
    log("Início/retomada30; uma amostra por pessoa; RAW, camada/C escolhidos só na CV interna.")
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            if (destination / "CONCLUIDO.json").exists():
                frames, arrays = load_checkpoint(destination, signature); validate(frames, arrays, meta, train, test, names, seed, fold)
                log(f"Checkpoint30 seed={seed} fold={fold} íntegro, sem reajuste"); continue
            require(not destination.exists() or not any(destination.iterdir()), f"Checkpoint parcial preservado: {destination}")
            started = time.monotonic(); frames, arrays = evaluate_fold(matrices, meta, names, train, test, seed, fold, log)
            validate(frames, arrays, meta, train, test, names, seed, fold); save_checkpoint(destination, frames, arrays, signature)
            log(f"Concluído30 seed={seed} fold={fold}; segundos={time.monotonic()-started:.2f}")
    consolidate(out, meta, seeds, log)


if __name__ == "__main__":
    main()

