# -*- coding: utf-8 -*-
"""Exp21: WPT fixa, XLS-R congelado e fusao de atributos, por participante.

O extrator20 gera embeddings sem usar rotulos nem atualizar o encoder. Este
avaliador usa apenas numpy/sklearn do ambiente original; nao requer PyTorch.
--verificar somente le; --executar ajusta os quatro bracos fixos; --autoteste
exercita somente dados sinteticos. Nao modifica nenhum experimento anterior.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp19_for_exp21", HERE / "19_avaliar_wpt_16khz_avfad.py")
EXP19 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP19)
EXP17, EXP13 = EXP19.EXP17, EXP19.EXP13
VERSION = "21.1"
SEEDS = (20261105, 101, 102, 103, 104, 105)
ARMS = ("A_WPT", "B_WAV2VEC", "C_HIBRIDO", "D_WAV2VEC_256")
MODALITIES = ("wpt", "wav2vec")
MODEL_ID = "facebook/wav2vec2-xls-r-300m"
DATA20_REL = Path("results/experiments/resultados_20_wav2vec_extracao")
REF17_REL = Path("results/experiments/resultados_17_robustez_distribuicao_wpt")
OUT_REL = Path("results/experiments/resultados_21_wpt_wav2vec")
CONTRASTS = {"C-B_contribuicao_wavelet_primaria": {ARMS[2]: 1, ARMS[1]: -1},
             "C-D_orcamento256_secundaria": {ARMS[2]: 1, ARMS[3]: -1},
             "C-A_ganho_sobre_WPT_secundaria": {ARMS[2]: 1, ARMS[0]: -1},
             "B-A_representacoes_descritiva": {ARMS[1]: 1, ARMS[0]: -1}}
save_checkpoint, load_checkpoint = EXP19.save_checkpoint, EXP19.load_checkpoint
fold_table = EXP19.fold_table


def embedding_names():
    return np.asarray([f"WAV2VEC__xlsr300m__{stat}__{j:04d}" for stat in ("mean", "std") for j in range(1024)])


def protocol():
    return """EXPERIMENTO21 - WPT + WAV2VEC2 XLS-R CONGELADO
Hipotese: descritores WPT acrescentam informacao aos embeddings externos.
C-B e o contraste principal de contribuicao incremental da Wavelet.
Examinar tambem C-D para verificar se o ganho decorre apenas de mais atributos.
411 pessoas HSA,82 normais/329 patologicas,2465 frases; nenhuma exclusao nova.
WPT48k sym5 L5+L6:1920 atributos, referencia fixa13/17, sem escolher taxa pelo19.
Encoder20:facebook/wav2vec2-xls-r-300m, revisaoSHA40 congelada, eval/semgrad,
audio16k, ultima camada1024, media e DP populacional temporal:2048 atributos.
Chunks e VAD definidos pelo20, fixados independentemente de rotulos/desempenho.
O cache externo congelado pode ser comum aos folds: nao e adaptado a AVFAD.
A_WPT:imputacao mediana/constantes/ANOVA128/StandardScaler/SVClinearC1balanced.
B_WAV2VEC:mesmo pipeline e classificador, somente embeddings.
C_HIBRIDO:mesmos128 atributos e escala de cada ramo ajustados no treino,
concatenados e divididos por sqrt(2); SVClinearC1balanced,256 atributos totais.
Fusao antes do classificador; nao soma modelos nem calibra margens isoladas.
Mesmos preprocessadores dos ramos isolados; nenhuma informacao do teste no fit.
D_WAV2VEC_256:imputacao/constantes/ANOVA256/StandardScaler nos embeddings,
divisao por sqrt(2),SVClinearC1balanced. Controle de orcamento256 contra256
e mesma energia media apos escala dos blocos; comparacoes A/B usam128.
Mesmo orcamento nao iguala representacoes/selecao ou demonstra causalidade.
Limiarzero em todos, classe patologica=1, balanceamento por classe de gravacao.
SemSMOTE, semfinetuning, sembusca de camada/pooling/C/k/fusao/limiar no teste.
Cincofold estratificados por pessoa, mesmos13/17:20261105principal,101..105
estabilidade. Mesmas frases da pessoa sempre juntas. Nao escolher melhorseed.
Metrica principal BA por pessoa:media das margens das frases, depois limiar0.
Tambem sensibilidade/especificidade/acuracia/matriz/AUCfold ponderada porpares.
AUC de margensOOF concatenadas e descritiva porque margens nao calibradas.
Treino e gravacao:diagnosticos de apoio; nao substituir avaliacao por pessoa.
C-B primario,C-D e C-A secundarios,B-A descritivo,por escopo separado.
IC95 pareado estratificado porpessoa,2000bootstraps; as seeds da mesma pessoa
permanecem juntas. ICcondicional nao refaz selecao/treino nem corrige exploracao
historica da AVFAD. Coorte e desenvolvimento; SVD nao e avaliada neste ciclo.
ControleA reproduz scores/decisoes/IDs/folds17 e13primary antes de interpretar.
Todos os bracos,seeds e resultados negativos sao preservados. Nenhuma conclusao
final ou afirmacao de auxilio Wavelet sera escrita sem evidencia comparativa.
"""


def validate_model_configuration(identity):
    config = identity.get("configuration", {})
    expected = {"model_id": MODEL_ID, "sample_rate": 16000, "frozen": True,
                "hidden_layer": "last", "pooling": "mean_std", "hidden_size": 1024}
    if any(config.get(key) != value for key, value in expected.items()) or config.get("frozen") is not True:
        raise ValueError("Encoder/pooling/taxa/congelamento20 diferem do protocolo21")
    if not isinstance(config.get("model_revision"), str) or re.fullmatch(r"[0-9a-f]{40}", config["model_revision"]) is None:
        raise ValueError("Revisao do modelo20 deve ser SHA40 fixa")


def validate_alignment(matrices, meta, names, embedding_meta, embedded, real=True):
    fields = ("sample_id", "participant_id", "task", "y")
    for frame in (meta, embedding_meta):
        if not set(fields).issubset(frame.columns) or frame[list(fields)].isna().any().any():
            raise ValueError("IDs/rotulos ausentes")
        if frame.sample_id.duplicated().any() or frame.duplicated(["participant_id", "task"]).any():
            raise ValueError("IDs/tarefas duplicados")
        if frame.groupby("participant_id").y.nunique().ne(1).any() or set(frame.y.unique()) != {0, 1}:
            raise ValueError("Classes/rotulos inconsistentes por pessoa")
    for key in fields:
        if not np.array_equal(meta[key].astype(str), embedding_meta[key].astype(str)):
            raise ValueError(f"Ordem/conteudo de {key} diverge entre modalidades")
    for key in fields[:3]:
        if key not in embedded or not np.array_equal(np.asarray(embedded[key]).astype(str), meta[key].astype(str)):
            raise ValueError(f"IDsNPZ e manifesto diferem:{key}")
    if "matrix_row" not in embedding_meta or not np.array_equal(embedding_meta.matrix_row, np.arange(len(meta))):
        raise ValueError("matrix_row nao preserva a ordem")
    for modality, width in (("wpt", 1920), ("wav2vec", 2048)):
        x = matrices[modality]
        if x.shape != (len(meta), width) or len(names[modality]) != width or len(set(names[modality])) != width:
            raise ValueError(f"Dimensao/nomes inesperados:{modality}")
        if not np.isfinite(x).all():
            raise ValueError(f"Atributos nao finitos:{modality}")
    if not np.array_equal(names["wav2vec"], embedding_names()):
        raise ValueError("Nomes/ordem dos embeddings diferem do contrato20")
    if real:
        people = meta[["participant_id", "y"]].drop_duplicates()
        if (len(meta), len(people), people.y.value_counts().to_dict()) != (2465, 411, {1: 329, 0: 82}):
            raise ValueError("Coorte deve ser HSA2465/411/82/329")
        if meta.groupby("participant_id").size().value_counts().to_dict() != {6: 410, 5: 1}:
            raise ValueError("Numero de frases inesperado")


def load_inputs(root, data20):
    xw, meta, nw = EXP17.load_inputs(root)
    npzfile = data20 / "features_wav2vec_xlsr300m.npz"
    metafile = data20 / "metadata_wav2vec.csv"
    signature_file = data20 / "assinatura_execucao.json"
    marker = data20 / "ARTEFATOS_COMPLETOS.json"
    for path in (npzfile, metafile, signature_file, marker):
        if not path.is_file():
            raise FileNotFoundError(f"Extracao20 completa necessaria:{path}")
    exports = json.loads(marker.read_text(encoding="utf-8"))
    if not {npzfile.name, metafile.name, signature_file.name, "RESUMO_EXTRACAO.txt"}.issubset(exports):
        raise ValueError("Manifesto20 incompleto")
    for name, digest in exports.items():
        if Path(name).name != name or name in (".", "..") or EXP17.sha(data20 / name) != digest:
            raise ValueError(f"Artefato20 alterado ou caminho invalido:{name}")
    identity = json.loads(signature_file.read_text(encoding="utf-8"))
    validate_model_configuration(identity)
    embedding_meta = pd.read_csv(metafile, dtype={"sample_id": str, "participant_id": str, "task": str})
    embedding_meta["task"] = embedding_meta.task.str.zfill(3)
    with np.load(npzfile, allow_pickle=False) as data:
        required = {"embeddings", "feature_names", "sample_id", "participant_id", "task"}
        if not required.issubset(data.files):
            raise ValueError("NPZ20 sem embeddings/nomes/IDs")
        xe = np.asarray(data["embeddings"], dtype=np.float32)
        ne = data["feature_names"].astype(str)
        embedded = {key: data[key].astype(str) for key in ("sample_id", "participant_id", "task")}
    matrices, names = {"wpt": xw, "wav2vec": xe}, {"wpt": nw, "wav2vec": ne}
    validate_alignment(matrices, meta, names, embedding_meta, embedded)
    return matrices, meta, names


def references(root, ref17dir, meta, seeds):
    r13 = pd.read_csv(root / "results/experiments/resultados_13_otimizacao_wpt_sym5/predicoes_oof_participante.csv", float_precision="round_trip")
    r13 = r13[r13.configuracao == "WPT_fixa"].copy()
    r17 = pd.read_csv(ref17dir / "predicoes_oof_participante.csv", float_precision="round_trip")
    r17 = r17[r17.arm == "A_STANDARD_LINEAR"].copy()
    ids = set(meta.participant_id)
    for seed in seeds:
        sub = r17[r17.seed == seed]
        if len(sub) != len(ids) or set(sub.participant_id) != ids or sub.participant_id.duplicated().any():
            raise ValueError(f"Controle17 incompleto:seed{seed}")
        for fold, (_, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            if set(sub[sub.fold == fold].participant_id) != set(meta.iloc[test].participant_id):
                raise ValueError("Particoes21 diferem do17")
    return r13, r17


def check_reference(frame, r13, r17, arm, seed, fold):
    if arm != ARMS[0]:
        return
    targets = []
    if r17 is not None:
        targets.append(r17[(r17.seed == seed) & (r17.fold == fold)])
    if r13 is not None and seed == SEEDS[0]:
        targets.append(r13[r13.outer_fold == fold])
    current = frame.set_index("participant_id").sort_index()
    for target in targets:
        expected = target.set_index("participant_id").sort_index()
        pd.testing.assert_index_equal(current.index, expected.index)
        for key in ("y", "predicao", "n_gravacoes"):
            np.testing.assert_array_equal(current[key], expected[key])
        np.testing.assert_allclose(current.score, expected.score, atol=1e-9, rtol=1e-9)


def identity(root, data20, ref17dir, seeds):
    inputs = [root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz",
              root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv",
              *[data20 / name for name in ("features_wav2vec_xlsr300m.npz", "metadata_wav2vec.csv",
                                         "ARTEFATOS_COMPLETOS.json", "assinatura_execucao.json")],
              ref17dir / "predicoes_oof_participante.csv", ref17dir / "assinatura_execucao.json",
              root / "results/experiments/resultados_13_otimizacao_wpt_sym5/predicoes_oof_participante.csv"]
    code = [Path(__file__).resolve(), HERE / "20_extrair_wav2vec_avfad.py",
            HERE / "19_avaliar_wpt_16khz_avfad.py", HERE / "17_robustez_distribuicao_wpt_avfad.py",
            HERE / "13_otimizar_wpt_sym5.py"]
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
        environment[name] = importlib.metadata.version(name)
    return {"version": VERSION, "configuration": {"arms": list(ARMS), "seeds": list(seeds), "folds": 5,
            "features": {"wpt": 1920, "wav2vec": 2048}, "k_per_branch": 128, "hybrid_k_total": 256,
            "C": 1.0, "kernel": "linear", "class_weight": "balanced", "threshold": 0.0,
            "hybrid_divisor": "sqrt(2)", "embedding_budget_control_k": 256,
            "embedding_budget_control_divisor": "sqrt(2)", "bootstrap_iterations": 2000,
            "primary": "C-B", "secondary": ["C-D", "C-A"]},
            "environment": environment, "input_files": {str(p.resolve()): EXP17.sha(p) for p in inputs},
            "code": {str(p): EXP17.sha(p) for p in code},
            "protocol_sha256": hashlib.sha256(protocol().encode("utf-8")).hexdigest()}


def classifier_parameters(classifier, train):
    return {"support_indices": classifier.support_.copy(), "support_original_rows": train[classifier.support_],
            "support_vectors": classifier.support_vectors_.copy(), "dual_coef": classifier.dual_coef_.copy(),
            "coef": classifier.coef_.copy(), "intercept": classifier.intercept_.copy(),
            "n_support": classifier.n_support_.copy(), "classes": classifier.classes_.copy(),
            "class_weight": classifier.class_weight_.copy(), "gamma": np.asarray(classifier._gamma)}


def divide_sqrt2(values):
    return values / np.sqrt(2.0)


def arm_modalities(arm):
    if arm == ARMS[2]:
        return MODALITIES
    return ("wpt",) if arm == ARMS[0] else ("wav2vec",)


def attributes_per_modality(arm):
    return 256 if arm == ARMS[3] else 128


def fit_fold_models(matrices, y, train, test):
    """Ajustes dependem somente de train e y[train]; teste so recebe transform."""
    pipelines, transformed_train, transformed_test, parameters = {}, {}, {}, {}
    scores = {}
    for arm, modality in zip(ARMS[:2], MODALITIES):
        model = EXP17.model_for("A_STANDARD_LINEAR", len(train))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(matrices[modality][train], y[train])
        pipelines[modality] = model
        transformed_train[modality] = model[:-1].transform(matrices[modality][train])
        transformed_test[modality] = model[:-1].transform(matrices[modality][test])
        scores[arm] = (model.decision_function(matrices[modality][train]), model.decision_function(matrices[modality][test]))
        parameters.update({f"{arm}__{key}": value for key, value in EXP17.parameters(model).items()})
        parameters.update({f"{arm}__{key}": value for key, value in classifier_parameters(model.named_steps["classifier"], train).items()})
    hybrid_train = np.concatenate([transformed_train[key] for key in MODALITIES], axis=1) / np.sqrt(2.0)
    hybrid_test = np.concatenate([transformed_test[key] for key in MODALITIES], axis=1) / np.sqrt(2.0)
    classifier = clone(pipelines["wpt"].named_steps["classifier"])
    classifier.fit(hybrid_train, y[train])
    scores[ARMS[2]] = (classifier.decision_function(hybrid_train), classifier.decision_function(hybrid_test))
    parameters.update({f"{ARMS[2]}__{key}": value for key, value in classifier_parameters(classifier, train).items()})
    for modality, arm in zip(MODALITIES, ARMS[:2]):
        for key in ("imputer", "constant_mask", "selector_scores", "selector_mask", "scaler_mean", "scaler_scale"):
            parameters[f"{ARMS[2]}__{modality}__{key}"] = parameters[f"{arm}__{key}"].copy()
    # Controle com o mesmo numero de dimensoes e a mesma energia media
    # apos padronizacao do hibrido. A/B/C permanecem numericamente iguais.
    template = EXP13.pipeline_for("SVC_linear", 256, 1.0)
    steps = template.steps.copy()
    steps.insert(-1, ("budget_scale", FunctionTransformer(divide_sqrt2)))
    budget_model = Pipeline(steps)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        budget_model.fit(matrices["wav2vec"][train], y[train])
    scores[ARMS[3]] = (budget_model.decision_function(matrices["wav2vec"][train]),
                       budget_model.decision_function(matrices["wav2vec"][test]))
    parameters.update({f"{ARMS[3]}__{key}": value for key, value in EXP17.parameters(budget_model).items()})
    parameters.update({f"{ARMS[3]}__{key}": value for key, value in classifier_parameters(budget_model.named_steps["classifier"], train).items()})
    return scores, parameters


def evaluate_fold(matrices, meta, names, train, test, seed, fold, log, r13=None, r17=None):
    train_meta, test_meta = meta.iloc[train].reset_index(drop=True), meta.iloc[test].reset_index(drop=True)
    if set(train_meta.participant_id) & set(test_meta.participant_id):
        raise ValueError("Pessoa compartilhada em treino e teste")
    scores, arrays = fit_fold_models(matrices, meta.y.to_numpy(), train, test)
    outputs, train_outputs, records_test, records_train, metrics, features = [], [], [], [], [], []
    for arm in ARMS:
        train_score, test_score = scores[arm]
        pred, fit_pred = EXP17.collapse(test_meta, test_score), EXP17.collapse(train_meta, train_score)
        check_reference(pred, r13, r17, arm, seed, fold)
        if arm == ARMS[0] and (r13 is not None or r17 is not None):
            log(f"CONTROLE21 seed={seed} fold={fold}: WPT13/17 confirmada")
        for frame in (pred, fit_pred):
            frame["arm"], frame["seed"], frame["fold"], frame["limiar"] = arm, seed, fold, 0.0
        rt = EXP19.recording_predictions(test_meta, test_score, arm, seed, fold)
        rr = EXP19.recording_predictions(train_meta, train_score, arm, seed, fold)
        for phase, unit, frame in (("teste", "participante", pred), ("treino", "participante", fit_pred),
                                   ("teste", "gravacao", rt), ("treino", "gravacao", rr)):
            metrics.append({"arm": arm, "seed": seed, "fold": fold, "phase": phase, "unit": unit, **EXP17.measures(frame)})
        modalities = arm_modalities(arm)
        for modality in modalities:
            branch_arm = ARMS[3] if arm == ARMS[3] else ARMS[MODALITIES.index(modality)]
            var, sel = arrays[f"{branch_arm}__constant_mask"], arrays[f"{branch_arm}__selector_mask"]
            features.extend({"arm": arm, "seed": seed, "fold": fold, "modality": modality, "feature": str(name)}
                            for name in names[modality][var][sel])
        outputs.append(pred); train_outputs.append(fit_pred); records_test.append(rt); records_train.append(rr)
        log(f"seed={seed} fold={fold} {arm}: BA_teste_pessoa={EXP17.measures(pred)['BA']:.6f}; "
            f"BA_treino_pessoa={EXP17.measures(fit_pred)['BA']:.6f}")
    return {"predicoes.csv": pd.concat(outputs, ignore_index=True),
            "predicoes_treino_participante.csv": pd.concat(train_outputs, ignore_index=True),
            "predicoes_teste_gravacao.csv": pd.concat(records_test, ignore_index=True),
            "predicoes_treino_gravacao.csv": pd.concat(records_train, ignore_index=True),
            "metricas.csv": pd.DataFrame(metrics), "atributos.csv": pd.DataFrame(features)}, arrays


def validate(frames, arrays, meta, train, test, names, seed, fold, r13=None, r17=None):
    expected_y = meta.drop_duplicates("participant_id").set_index("participant_id").y
    if set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise ValueError("Participante compartilhado no fold")
    for filename, indices, unit in (("predicoes.csv", test, "participante"),
                                    ("predicoes_treino_participante.csv", train, "participante"),
                                    ("predicoes_teste_gravacao.csv", test, "gravacao"),
                                    ("predicoes_treino_gravacao.csv", train, "gravacao")):
        frame, expected_meta = frames[filename], meta.iloc[indices]
        if set(frame.arm) != set(ARMS):
            raise ValueError("Bracos inesperados")
        key = "participant_id" if unit == "participante" else "sample_id"
        for arm, sub in frame.groupby("arm"):
            if sub[key].duplicated().any() or set(sub[key]) != set(expected_meta[key]):
                raise ValueError(f"IDs/quantidade inesperados:{filename}:{arm}")
            if not sub.seed.eq(seed).all() or not sub.fold.eq(fold).all():
                raise ValueError("Seed/fold incorreto")
            if not np.array_equal(sub.y, sub.participant_id.map(expected_y)):
                raise ValueError("Rotulo diverge do manifesto")
            if not np.isfinite(sub.score).all() or not np.array_equal(sub.predicao, (sub.score >= 0).astype(int)):
                raise ValueError("Score/decisao inconsistente")
            if unit == "participante":
                counts = expected_meta.groupby("participant_id").size()
                if not np.array_equal(sub.n_gravacoes, sub.participant_id.map(counts)):
                    raise ValueError("Contagem de frases inconsistente")
            else:
                original = expected_meta.set_index("sample_id").loc[sub.sample_id]
                if not np.array_equal(sub.participant_id, original.participant_id) or not np.array_equal(sub.task.astype(str).str.zfill(3), original.task.astype(str).str.zfill(3)):
                    raise ValueError("Identidade/tarefa da gravacao incorreta")
            if filename == "predicoes.csv":
                check_reference(sub, r13, r17, arm, seed, fold)
    if set(frames["atributos.csv"].arm) != set(ARMS):
        raise ValueError("Selecao incompleta")
    for arm in ARMS:
        modalities = arm_modalities(arm)
        selected_rows = frames["atributos.csv"][frames["atributos.csv"].arm == arm]
        expected_k = attributes_per_modality(arm)
        if len(selected_rows) != expected_k * len(modalities) or set(selected_rows.modality) != set(modalities):
            raise ValueError("Quantidade/modalidade de atributos incorreta")
        for modality in modalities:
            rows = selected_rows[selected_rows.modality == modality]
            prefix = f"{arm}__{modality}__" if arm == ARMS[2] else f"{arm}__"
            var, sel = arrays[prefix + "constant_mask"].astype(bool), arrays[prefix + "selector_mask"].astype(bool)
            if len(var) != len(names[modality]) or len(sel) != var.sum() or sel.sum() != expected_k:
                raise ValueError("Mascaras de selecao invalidas")
            if rows.feature.duplicated().any() or not np.array_equal(rows.feature, names[modality][var][sel]):
                raise ValueError("Atributos diferem dos parametros salvos")
            if not rows.seed.eq(seed).all() or not rows.fold.eq(fold).all():
                raise ValueError("Particao incorreta em atributos")
        support = arrays[f"{arm}__support_indices"]
        if (support < 0).any() or (support >= len(train)).any() or not np.array_equal(arrays[f"{arm}__support_original_rows"], train[support]):
            raise ValueError("Vetores de suporte fora do treino")
    for modality, arm in zip(MODALITIES, ARMS[:2]):
        for key in ("imputer", "constant_mask", "selector_scores", "selector_mask", "scaler_mean", "scaler_scale"):
            np.testing.assert_array_equal(arrays[f"{arm}__{key}"], arrays[f"{ARMS[2]}__{modality}__{key}"])
    sources = {("teste", "participante"): "predicoes.csv", ("treino", "participante"): "predicoes_treino_participante.csv",
               ("teste", "gravacao"): "predicoes_teste_gravacao.csv", ("treino", "gravacao"): "predicoes_treino_gravacao.csv"}
    metric_rows = frames["metricas.csv"]
    if len(metric_rows) != 4 * len(ARMS) or metric_rows.duplicated(["arm", "phase", "unit"]).any():
        raise ValueError("Metricas incompletas/duplicadas")
    for row in metric_rows.itertuples():
        source = frames[sources[(row.phase, row.unit)]]
        for key, value in EXP17.measures(source[source.arm == row.arm]).items():
            if not np.isclose(value, getattr(row, key), atol=1e-12, rtol=0):
                raise ValueError(f"Metrica incorreta:{key}")
    for phase, people_file, record_file in (("teste", "predicoes.csv", "predicoes_teste_gravacao.csv"),
                                            ("treino", "predicoes_treino_participante.csv", "predicoes_treino_gravacao.csv")):
        for arm in ARMS:
            records = frames[record_file].query("arm == @arm")
            people = frames[people_file].query("arm == @arm").set_index("participant_id").sort_index()
            aggregated = EXP17.collapse(records, records.score.to_numpy()).set_index("participant_id").sort_index()
            np.testing.assert_allclose(people.score, aggregated.score, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(people.predicao, aggregated.predicao)


def consolidate(out, meta, seeds, log):
    destinations = [out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" for seed in seeds for fold in range(1, 6)]
    raw = pd.concat([pd.read_csv(p / "predicoes.csv", float_precision="round_trip") for p in destinations], ignore_index=True)
    folds = pd.concat([pd.read_csv(p / "metricas.csv", float_precision="round_trip") for p in destinations], ignore_index=True)
    features = pd.concat([pd.read_csv(p / "atributos.csv") for p in destinations], ignore_index=True)
    summaries = []
    for (seed, arm), group in raw.groupby(["seed", "arm"]):
        if len(group) != meta.participant_id.nunique() or group.participant_id.duplicated().any():
            raise ValueError("OOF duplicado/incompleto")
        measured = EXP17.measures(group)
        measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        f = folds[(folds.seed == seed) & (folds.arm == arm) & (folds.phase == "teste") & (folds.unit == "participante")]
        measured["AUC_media_folds_ponderada_pares"] = float(np.average(f.AUC, weights=(f.TN + f.FP) * (f.TP + f.FN)))
        summaries.append({"seed": seed, "arm": arm, **measured})
    summary, comparisons = pd.DataFrame(summaries), []
    scopes = [("principal_seed20261105", [20261105])] if 20261105 in seeds else []
    secondary = [seed for seed in seeds if seed != 20261105]
    if secondary:
        scopes.append(("estabilidade_secundaria", secondary))
    for label, selected in scopes:
        for name, coefficients in CONTRASTS.items():
            comparisons.append({"escopo": label, "comparacao": name, "seeds": str(selected),
                                **EXP17.intervals(raw, selected, coefficients, iterations=2000)})
    for filename, frame in (("predicoes_oof_participante.csv", raw), ("metricas_por_fold.csv", folds),
                             ("metricas_por_seed.csv", summary), ("comparacoes_pareadas.csv", pd.DataFrame(comparisons)),
                             ("atributos_selecionados.csv", features)):
        EXP17.csv_write(out / filename, frame)
    lines = ["EXPERIMENTO21 - WPT, WAV2VEC2 XLS-R, HIBRIDO", f"Consolidado:{EXP17.now()}",
             "Todos os folds solicitados concluidos; referenciaWPT13/17 conferida.",
             "Ciclo das seis sementes completo." if tuple(seeds) == SEEDS else "Ciclo parcial/diferente do protocolo.",
             "BA por participante; media das margens das frases e limiarzero.",
             "AUC principal de apoio: mediafold ponderada porpares; AUCconcatenada exploratoria.", ""]
    for row in summary.itertuples():
        lines.append(f"seed={row.seed} {row.arm}:BA={row.BA:.6f};sens={row.sensibilidade:.6f};esp={row.especificidade:.6f};"
                     f"acc={row.acuracia:.6f};AUCfold={row.AUC_media_folds_ponderada_pares:.6f};TN/FP/FN/TP={row.TN}/{row.FP}/{row.FN}/{row.TP}")
    lines += ["", "CONTRASTES PAREADOS; IC95CONDICIONAL 2000BOOTSTRAPS PORPESSOA"]
    for row in comparisons:
        lines.append(f"{row['escopo']} {row['comparacao']}:deltaBA={row['delta_BA']:+.6f};"
                     f"IC95=[{row['IC95_inferior']:+.6f},{row['IC95_superior']:+.6f}]")
    if secondary:
        lines += ["", "ESTABILIDADE SECUNDARIA; DP NAO E IC"]
        for arm, group in summary[summary.seed.isin(secondary)].groupby("arm"):
            lines.append(f"{arm}:BAmedia={group.BA.mean():.6f};min={group.BA.min():.6f};max={group.BA.max():.6f};DP={group.BA.std():.6f}")
    lines += ["", "TREINO APENAS COMO DIAGNOSTICO"]
    for (arm, unit), group in folds[folds.phase == "treino"].groupby(["arm", "unit"]):
        lines.append(f"{arm}/{unit}:BAmediaentreajustes={group.BA.mean():.6f}")
    lines += ["", "LIMITACOES", "AVFAD ja explorada: desenvolvimento, nao teste independente.",
              "C possui256 atributos;D controla256embeddings e divisor sqrt2,C-D e secundario.",
              "Orcamento256 igual nao iguala representacoes/selecao; nao implica causalidade clinica.",
              "Encoder congelado; nenhuma conclusao sobre finetuning ou outras camadas/pooling.",
              "Diferentes pre-processamentos de amplitude/temporalidade entre representacoes fazem parte dos pipelines.",
              "ICs condicionais nao refazem treino nem corrigem adaptacao a tentativas historicas.",
              "Desempenho maior que WPT sozinho nao demonstra contribuicaoWavelet:examinar C-B e C-D.",
              "Nenhum dado SVD foi avaliado; nenhum resultado anterior foi sobrescrito.",
              "Nenhuma conclusao final do TG produzida automaticamente.\n"]
    EXP17.text_write(out / "RELATORIO_RESULTADOS.txt", "\n".join(lines))
    exports = ["predicoes_oof_participante.csv", "metricas_por_fold.csv", "metricas_por_seed.csv",
               "comparacoes_pareadas.csv", "atributos_selecionados.csv", "RELATORIO_RESULTADOS.txt",
               "assinatura_execucao.json", "particoes_participantes.csv", "PROTOCOLO.txt"]
    EXP17.json_write(out / "ARTEFATOS_COMPLETOS.json", {name: EXP17.sha(out / name) for name in exports})
    log("Consolidacao21 concluida; todos os bracos/escopos preservados.")


def selftest():
    import unittest
    spec = importlib.util.spec_from_file_location("test_wpt_wav2vec21", HERE / "test_wpt_wav2vec21.py")
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_module))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path, default=ROOT / OUT_REL)
    parser.add_argument("--dados20", type=Path)
    parser.add_argument("--referencia17", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true")
    action.add_argument("--executar", action="store_true")
    action.add_argument("--autoteste", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        selftest()
        return
    root, out, seeds = args.dados.resolve(), args.saida.resolve(), tuple(args.seeds)
    if not out.is_relative_to(root / "results/experiments"):
        parser.error("Saida deve estar dentro de results/experiments")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        parser.error("Seeds devem ser unicas e validas")
    data20 = args.dados20.resolve() if args.dados20 else root / DATA20_REL
    ref17 = args.referencia17.resolve() if args.referencia17 else root / REF17_REL
    matrices, meta, names = load_inputs(root, data20)
    r13, r17 = references(root, ref17, meta, seeds)
    ident = identity(root, data20, ref17, seeds)
    print(f"ENTRADAS21 OK:WPT{matrices['wpt'].shape};XLSR{matrices['wav2vec'].shape};411pessoas;seeds={seeds}", flush=True)
    if not args.executar:
        print("Somente verificacao; sem treino nem escrita.", flush=True)
        return
    signature_file = out / "assinatura_execucao.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf-8")) != ident:
            raise RuntimeError("Configuracao/dados/codigo/ambiente mudaram; preservar pasta")
    else:
        if out.exists() and any(path.name != "PROTOCOLO.txt" for path in out.iterdir()):
            raise RuntimeError("Pasta sem assinatura contem artefatos; preservar")
        out.mkdir(parents=True, exist_ok=True)
        EXP17.freeze_text(out / "PROTOCOLO.txt", protocol())
        EXP17.json_write(signature_file, ident)
    EXP17.freeze_text(out / "PROTOCOLO.txt", protocol())
    EXP17.freeze_text(out / "particoes_participantes.csv", fold_table(meta, seeds).to_csv(index=False, lineterminator="\n"))
    signature = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    log = EXP17.Logger(out / "LOG_EXECUCAO.txt")
    log(f"Inicio/retomada21;seeds={seeds};sem busca/finetuning/exclusoes.")
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            if (destination / "CONCLUIDO.json").exists():
                frames, arrays = load_checkpoint(destination, signature)
                validate(frames, arrays, meta, train, test, names, seed, fold, r13, r17)
                log(f"Checkpoint21 seed={seed} fold={fold} verificado; nenhum reajuste")
                continue
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Checkpoint parcial preservado:{destination}")
            started = time.monotonic()
            frames, arrays = evaluate_fold(matrices, meta, names, train, test, seed, fold, log, r13, r17)
            validate(frames, arrays, meta, train, test, names, seed, fold, r13, r17)
            save_checkpoint(destination, frames, arrays, signature)
            log(f"Concluido21 seed={seed} fold={fold};segundos={time.monotonic()-started:.2f}")
    consolidate(out, meta, seeds, log)


if __name__ == "__main__":
    main()
