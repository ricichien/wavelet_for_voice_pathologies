# -*- coding: utf-8 -*-
"""Experimento 24: ablação fixa de RMS e logenergia, avaliação por participante.

--verificar apenas lê; --autoteste usa dados sintéticos; --executar ajusta os
três braços na mesma coorte e nos mesmos folds dos experimentos anteriores.
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
SPEC = importlib.util.spec_from_file_location("exp21_for_exp24", HERE / "21_avaliar_wpt_wav2vec_avfad.py")
EXP21 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP21)
EXP19, EXP17, EXP13 = EXP21.EXP19, EXP21.EXP17, EXP21.EXP13
VERSION = "24.1"
SEEDS = EXP21.SEEDS
ARMS = ("A_RAW", "B_RMS", "C_NO_LOG")
DATA23_REL = Path("results/experiments/resultados_23_wpt_rms_sequencias")
OUT_REL = Path("results/experiments/resultados_24_normalizacao_wpt")
REF17_REL = EXP21.REF17_REL
CONTRASTS = {"B-A_RMS_primaria": {ARMS[1]: 1, ARMS[0]: -1},
             "C-A_remover_logenergia_secundaria": {ARMS[2]: 1, ARMS[0]: -1},
             "B-C_RMS_vs_sem_logenergia_secundaria": {ARMS[1]: 1, ARMS[2]: -1}}
save_checkpoint, load_checkpoint = EXP19.save_checkpoint, EXP19.load_checkpoint


def protocol():
    return """EXPERIMENTO 24 — NORMALIZAÇÃO RMS E ABLAÇÃO DE LOGENERGIA WPT
Hipótese principal: diferenças de ganho global entre gravações podem reduzir
a generalização dos descritores WPT. Igualar RMS da voz ativa pode ajudar.
Essa hipótese não pressupõe que amplitude seja apenas ruído ou que haja ganho.
Coorte HSA idêntica: 411 participantes, 82 normais e 329 patológicos;
2.465 frases CAPE-V, nenhuma nova exclusão. SVD não é avaliada neste ciclo.
A_RAW: WPT sym5 legada de 48 kHz, níveis 5 e 6, 1.920 atributos.
B_RMS: mesma extração e gravações, um ganho global por gravação para RMS0,1
da voz ativa, com VAD aplicado ao sinal original ANTES da normalização.
C_NO_LOG: matriz A sem os 384 atributos do bloco log_energy, restando1.536.
Remover logenergia não equivale a RMS: também remove informação espectral.
Não há normalização por janela, novo VAD, denoising ou alteração de canais.
O experimento23 fornece dados novos; os caches antigos permanecem somente leitura.
Todos os braços: imputação mediana, remoção de constantes, ANOVA128,
StandardScaler e SVC linear C=1, class_weight=balanced, limiar zero.
Pré-processamento ajustado SOMENTE no treino do respectivo fold.
Sem busca de hiperparâmetros, SMOTE, ajuste de limiar ou seleção de participantes.
Cinco folds estratificados na tabela de pessoas, frases sempre juntas.
Semente principal20261105; estabilidade101,102,103,104,105, mesmos folds13/17/21.
Uma margem por gravação; uma decisão por pessoa pela média das margens e limiar0.
BA por participante é principal; sensibilidade, especificidade, acurácia,
matriz de confusão e AUC por fold ponderada por pares são reportadas.
Resultados de gravação/treino são apoio. AUC de margens concatenadas é exploratória.
Contraste primário B-A; C-A e B-C secundários. Não escolher melhor semente.
IC95 pareado, 2.000 bootstraps estratificados por participante, mantendo as
repetições da mesma pessoa no mesmo cluster. Não refaz ajuste/seleção/treino.
Controle A reproduz referências13/17 por fold; entradas23 vinculadas a hashes,
IDs, rótulos, configuração RMS e controles48k de todas as gravações.
Guardar partições, atributos, parâmetros, margens, métricas e checkpoints.
AVFAD já explorada: desenvolvimento, não teste cego ou validação clínica externa.
Preservar resultados negativos; nenhuma conclusão final do TCC é automática.
"""


def no_log_mask(names, real=True):
    names = np.asarray(names).astype(str)
    mask = np.asarray(["__log_energy__" not in name for name in names])
    if real and (len(names) != 1920 or int((~mask).sum()) != 384 or int(mask.sum()) != 1536):
        raise ValueError("Esperava 384 atributos log_energy entre os 1.920 da referência")
    return mask


def validate_configuration(identity):
    config = identity.get("configuration", {})
    expected = {"sample_rate": 48000, "target_rms": .1, "normalization": "global_active_rms",
                "vad_before_normalization": True, "wavelet": "sym5", "levels": [5, 6]}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("Configuração23 difere da normalização RMS global de voz ativa prevista")
    if config.get("vad_before_normalization") is not True or identity.get("mode") != "completo":
        raise ValueError("Exige extração23 completa e VAD anterior à normalização")


def validate_alignment(raw, meta, names, rms, rms_meta, rms_names, embedded, real=True):
    fields = ("sample_id", "participant_id", "task", "y")
    for frame in (meta, rms_meta):
        if not set(fields).issubset(frame) or frame[list(fields)].isna().any().any():
            raise ValueError("IDs ou rótulos ausentes")
        if frame.sample_id.duplicated().any() or frame.duplicated(["participant_id", "task"]).any():
            raise ValueError("Gravação/tarefa duplicada")
        if frame.groupby("participant_id").y.nunique().ne(1).any() or set(frame.y.unique()) != {0, 1}:
            raise ValueError("Classes ou rótulos divergentes por pessoa")
    for key in fields:
        if not np.array_equal(meta[key].astype(str), rms_meta[key].astype(str)):
            raise ValueError(f"Conteúdo ou ordem de {key} diverge entre as modalidades")
    for key in fields[:3]:
        if key not in embedded or not np.array_equal(np.asarray(embedded[key]).astype(str), meta[key].astype(str)):
            raise ValueError(f"NPZ23 e manifesto divergem em {key}")
    if raw.shape != rms.shape or raw.shape != (len(meta), len(names)) or len(names) != 1920:
        raise ValueError("Matrizes devem ter as mesmas linhas e 1.920 atributos")
    if not np.array_equal(names, rms_names) or len(set(names)) != len(names):
        raise ValueError("Nomes ou ordem dos atributos RMS divergem da referência")
    if not np.isfinite(raw).all() or not np.isfinite(rms).all():
        raise ValueError("Matrizes contêm valores não finitos")
    if "matrix_row" not in rms_meta or not np.array_equal(rms_meta.matrix_row, np.arange(len(meta))):
        raise ValueError("matrix_row não preserva a ordem do NPZ")
    no_log_mask(names)
    if real:
        people = meta[["participant_id", "y"]].drop_duplicates()
        if (len(meta), len(people), people.y.value_counts().to_dict()) != (2465, 411, {1: 329, 0: 82}):
            raise ValueError("Coorte deve ser HSA 2465/411/82/329")
        if meta.groupby("participant_id").size().value_counts().to_dict() != {6: 410, 5: 1}:
            raise ValueError("Contagens de frases inesperadas")


def load_inputs(root, data23):
    raw, meta, names = EXP17.load_inputs(root)
    marker = data23 / "ARTEFATOS_COMPLETOS.json"
    required = {"features_wpt_rms.npz", "metadata_wpt_rms.csv", "assinatura_execucao.json",
                "CONTROLE_REFERENCIA_48KHZ.csv", "sequencias_wpt.npz", "QUALIDADE_AUDIO.csv", "RESUMO_EXTRACAO.txt"}
    exports = json.loads(marker.read_text(encoding="utf8"))
    if not required.issubset(exports):
        raise ValueError("Marcador23 não contém todos os artefatos exigidos")
    for name, digest in exports.items():
        if Path(name).name != name or name in (".", "..") or EXP17.sha(data23 / name) != digest:
            raise ValueError(f"Artefato23 alterado ou caminho inválido: {name}")
    identity23 = json.loads((data23 / "assinatura_execucao.json").read_text(encoding="utf8"))
    validate_configuration(identity23)
    if identity23.get("sample_ids") != meta.sample_id.tolist():
        raise ValueError("Assinatura23 não corresponde à coorte completa na ordem auditada")
    rms_meta = pd.read_csv(data23 / "metadata_wpt_rms.csv", dtype={"sample_id": str, "participant_id": str, "task": str}, float_precision="round_trip")
    rms_meta["task"] = rms_meta.task.str.zfill(3)
    with np.load(data23 / "features_wpt_rms.npz", allow_pickle=False) as source:
        if not {"wavelet", "feature_names", "sample_id", "participant_id", "task"}.issubset(source.files):
            raise ValueError("NPZ23 não contém matriz, nomes e IDs")
        rms = np.asarray(source["wavelet"], dtype=np.float32)
        rms_names = source["feature_names"].astype(str)
        embedded = {key: source[key].astype(str) for key in ("sample_id", "participant_id", "task")}
    validate_alignment(raw, meta, names, rms, rms_meta, rms_names, embedded)
    control = pd.read_csv(data23 / "CONTROLE_REFERENCIA_48KHZ.csv")
    if len(control) != len(meta) or control.sample_id.duplicated().any() or set(control.sample_id) != set(meta.sample_id) or not control.equivalent.eq(True).all():
        raise ValueError("Controles48k não confirmam todas as gravações")
    mask = no_log_mask(names)
    return {ARMS[0]: raw, ARMS[1]: rms, ARMS[2]: raw[:, mask]}, meta, {ARMS[0]: names, ARMS[1]: names, ARMS[2]: names[mask]}


def references(root, ref17, meta, seeds):
    return EXP21.references(root, ref17, meta, seeds)


def check_reference(frame, r13, r17, arm, seed, fold):
    if arm == ARMS[0]:
        EXP21.check_reference(frame, r13, r17, EXP21.ARMS[0], seed, fold)


def identity(root, data23, ref17, seeds):
    inputs = [root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz",
              root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv",
              *[data23 / name for name in ("features_wpt_rms.npz", "metadata_wpt_rms.csv", "ARTEFATOS_COMPLETOS.json", "assinatura_execucao.json", "CONTROLE_REFERENCIA_48KHZ.csv")],
              ref17 / "predicoes_oof_participante.csv", ref17 / "assinatura_execucao.json",
              root / "results/experiments/resultados_13_otimizacao_wpt_sym5/predicoes_oof_participante.csv"]
    code = [Path(__file__).resolve(), HERE / "23_extrair_wpt_rms_sequencias_avfad.py",
            HERE / "21_avaliar_wpt_wav2vec_avfad.py", HERE / "19_avaliar_wpt_16khz_avfad.py",
            HERE / "17_robustez_distribuicao_wpt_avfad.py", HERE / "13_otimizar_wpt_sym5.py"]
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
        environment[name] = importlib.metadata.version(name)
    return {"version": VERSION, "configuration": {"arms": list(ARMS), "seeds": list(seeds), "folds": 5,
            "features": {ARMS[0]: 1920, ARMS[1]: 1920, ARMS[2]: 1536}, "k": 128,
            "kernel": "linear", "C": 1., "class_weight": "balanced", "threshold": 0.,
            "target_active_rms": .1, "bootstrap_iterations": 2000, "primary": "B-A"},
            "environment": environment, "input_files": {str(p.resolve()): EXP17.sha(p) for p in inputs},
            "code": {str(p.resolve()): EXP17.sha(p) for p in code},
            "protocol_sha256": hashlib.sha256(protocol().encode("utf8")).hexdigest()}


def evaluate_fold(matrices, meta, names, train, test, seed, fold, log=lambda _: None, r13=None, r17=None):
    if set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise ValueError("Pessoa compartilhada em treino e teste")
    train_meta, test_meta = meta.iloc[train].reset_index(drop=True), meta.iloc[test].reset_index(drop=True)
    output, training, records_test, records_train, metrics, attributes, arrays = [], [], [], [], [], [], {}
    for arm in ARMS:
        model = EXP17.model_for("A_STANDARD_LINEAR", len(train))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(matrices[arm][train], train_meta.y.to_numpy())
        if model.named_steps["classifier"].fit_status_ != 0:
            raise RuntimeError(f"SVC não convergiu: {arm}")
        score_train, score_test = model.decision_function(matrices[arm][train]), model.decision_function(matrices[arm][test])
        pred, fit_pred = EXP17.collapse(test_meta, score_test), EXP17.collapse(train_meta, score_train)
        check_reference(pred, r13, r17, arm, seed, fold)
        if arm == ARMS[0] and (r13 is not None or r17 is not None):
            log(f"CONTROLE24 seed={seed} fold={fold}: referência WPT13/17 confirmada")
        for frame in (pred, fit_pred):
            frame["arm"], frame["seed"], frame["fold"], frame["limiar"] = arm, seed, fold, 0.
        rt = EXP19.recording_predictions(test_meta, score_test, arm, seed, fold)
        rr = EXP19.recording_predictions(train_meta, score_train, arm, seed, fold)
        for phase, unit, frame in (("teste", "participante", pred), ("treino", "participante", fit_pred), ("teste", "gravacao", rt), ("treino", "gravacao", rr)):
            metrics.append({"arm": arm, "seed": seed, "fold": fold, "phase": phase, "unit": unit, **EXP17.measures(frame)})
        for key, value in EXP17.parameters(model).items():
            arrays[f"{arm}__{key}"] = value
        arrays.update({f"{arm}__{key}": value for key, value in EXP21.classifier_parameters(model.named_steps["classifier"], train).items()})
        var, sel = arrays[f"{arm}__constant_mask"], arrays[f"{arm}__selector_mask"]
        attributes.extend({"arm": arm, "seed": seed, "fold": fold, "feature": str(name)} for name in names[arm][var][sel])
        output.append(pred); training.append(fit_pred); records_test.append(rt); records_train.append(rr)
        log(f"seed={seed} fold={fold} {arm}: BA teste pessoa={EXP17.measures(pred)['BA']:.6f}; BA treino pessoa={EXP17.measures(fit_pred)['BA']:.6f}")
    return {"predicoes.csv": pd.concat(output, ignore_index=True), "predicoes_treino_participante.csv": pd.concat(training, ignore_index=True),
            "predicoes_teste_gravacao.csv": pd.concat(records_test, ignore_index=True), "predicoes_treino_gravacao.csv": pd.concat(records_train, ignore_index=True),
            "metricas.csv": pd.DataFrame(metrics), "atributos.csv": pd.DataFrame(attributes)}, arrays


def validate(frames, arrays, meta, train, test, names, seed, fold, r13=None, r17=None):
    if set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise ValueError("Participante compartilhado no fold")
    expected_y = meta.drop_duplicates("participant_id").set_index("participant_id").y
    for filename, indices, unit in (("predicoes.csv", test, "participante"), ("predicoes_treino_participante.csv", train, "participante"),
                                    ("predicoes_teste_gravacao.csv", test, "gravacao"), ("predicoes_treino_gravacao.csv", train, "gravacao")):
        frame, original_meta = frames[filename], meta.iloc[indices]
        if set(frame.arm) != set(ARMS):
            raise ValueError("Braços incompletos")
        key = "participant_id" if unit == "participante" else "sample_id"
        for arm, sub in frame.groupby("arm"):
            if sub[key].duplicated().any() or set(sub[key]) != set(original_meta[key]):
                raise ValueError("IDs de previsões incorretos")
            if not sub.seed.eq(seed).all() or not sub.fold.eq(fold).all() or not np.array_equal(sub.y, sub.participant_id.map(expected_y)):
                raise ValueError("Partição ou rótulo incorreto")
            if not np.isfinite(sub.score).all() or not np.array_equal(sub.predicao, (sub.score >= 0).astype(int)):
                raise ValueError("Margem ou decisão incorreta")
            if unit == "participante":
                if not np.array_equal(sub.n_gravacoes, sub.participant_id.map(original_meta.groupby("participant_id").size())):
                    raise ValueError("Contagem de frases incorreta")
            else:
                original = original_meta.set_index("sample_id").loc[sub.sample_id]
                if not np.array_equal(sub.participant_id, original.participant_id) or not np.array_equal(sub.task.astype(str).str.zfill(3), original.task.astype(str).str.zfill(3)):
                    raise ValueError("Identidade da gravação incorreta")
            if filename == "predicoes.csv":
                check_reference(sub, r13, r17, arm, seed, fold)
    if set(frames["atributos.csv"].arm) != set(ARMS):
        raise ValueError("Atributos incompletos")
    for arm in ARMS:
        rows = frames["atributos.csv"].query("arm == @arm")
        var, sel = arrays[f"{arm}__constant_mask"].astype(bool), arrays[f"{arm}__selector_mask"].astype(bool)
        if len(var) != len(names[arm]) or len(sel) != var.sum() or sel.sum() != 128 or len(rows) != 128 or rows.feature.duplicated().any():
            raise ValueError("Máscaras de seleção incorretas")
        if not np.array_equal(rows.feature, names[arm][var][sel]) or not rows.seed.eq(seed).all() or not rows.fold.eq(fold).all():
            raise ValueError("Atributos divergem dos parâmetros")
        support = arrays[f"{arm}__support_indices"]
        if (support < 0).any() or (support >= len(train)).any() or not np.array_equal(arrays[f"{arm}__support_original_rows"], train[support]):
            raise ValueError("Vetores de suporte fora do treino")
    metrics = frames["metricas.csv"]
    if len(metrics) != 12 or metrics.duplicated(["arm", "phase", "unit"]).any():
        raise ValueError("Métricas incompletas ou duplicadas")
    sources = {("teste", "participante"): "predicoes.csv", ("treino", "participante"): "predicoes_treino_participante.csv",
               ("teste", "gravacao"): "predicoes_teste_gravacao.csv", ("treino", "gravacao"): "predicoes_treino_gravacao.csv"}
    for row in metrics.itertuples():
        source = frames[sources[(row.phase, row.unit)]].query("arm == @row.arm")
        for key, value in EXP17.measures(source).items():
            if not np.isclose(value, getattr(row, key), atol=1e-12, rtol=0):
                raise ValueError("Métrica não corresponde às previsões")
    for people_file, record_file in (("predicoes.csv", "predicoes_teste_gravacao.csv"), ("predicoes_treino_participante.csv", "predicoes_treino_gravacao.csv")):
        for arm in ARMS:
            records = frames[record_file].query("arm == @arm")
            people = frames[people_file].query("arm == @arm").set_index("participant_id").sort_index()
            aggregate = EXP17.collapse(records, records.score.to_numpy()).set_index("participant_id").sort_index()
            np.testing.assert_allclose(people.score, aggregate.score, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(people.predicao, aggregate.predicao)


def consolidate(out, meta, seeds, log):
    destinations = [out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" for seed in seeds for fold in range(1, 6)]
    raw = pd.concat([pd.read_csv(p / "predicoes.csv", float_precision="round_trip") for p in destinations], ignore_index=True)
    folds = pd.concat([pd.read_csv(p / "metricas.csv", float_precision="round_trip") for p in destinations], ignore_index=True)
    features = pd.concat([pd.read_csv(p / "atributos.csv") for p in destinations], ignore_index=True)
    summaries = []
    for (seed, arm), group in raw.groupby(["seed", "arm"]):
        if group.participant_id.duplicated().any() or len(group) != meta.participant_id.nunique():
            raise ValueError("OOF incompleto ou duplicado")
        measured = EXP17.measures(group); measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        sub = folds[(folds.seed == seed) & (folds.arm == arm) & (folds.phase == "teste") & (folds.unit == "participante")]
        measured["AUC_media_folds_ponderada_pares"] = float(np.average(sub.AUC, weights=(sub.TN+sub.FP)*(sub.TP+sub.FN)))
        summaries.append({"seed": seed, "arm": arm, **measured})
    summary = pd.DataFrame(summaries)
    scopes = [("principal_seed20261105", [20261105])] if 20261105 in seeds else []
    secondary = [seed for seed in seeds if seed != 20261105]
    if secondary: scopes.append(("estabilidade_secundaria", secondary))
    contrasts = [{"escopo": label, "comparacao": name, "seeds": str(chosen), **EXP17.intervals(raw, chosen, coefficients, iterations=2000)}
                 for label, chosen in scopes for name, coefficients in CONTRASTS.items()]
    exports = {"predicoes_oof_participante.csv": raw, "metricas_por_fold.csv": folds, "metricas_por_seed.csv": summary,
               "comparacoes_pareadas.csv": pd.DataFrame(contrasts), "atributos_selecionados.csv": features}
    for name, frame in exports.items(): EXP17.csv_write(out / name, frame)
    lines = ["EXPERIMENTO 24 — RMS GLOBAL E ABLAÇÃO DE LOGENERGIA WPT", f"Consolidado: {EXP17.now()}",
             "Todos os folds solicitados concluídos; controles WPT13/17 conferidos.",
             "Ciclo das seis sementes completo." if tuple(seeds) == SEEDS else "Ciclo parcial/diferente do protocolo.",
             "BA principal por participante; média das margens das frases e limiar zero.", ""]
    for row in summary.itertuples():
        lines.append(f"seed={row.seed} {row.arm}: BA={row.BA:.6f}; sens={row.sensibilidade:.6f}; esp={row.especificidade:.6f}; acc={row.acuracia:.6f}; AUCfold={row.AUC_media_folds_ponderada_pares:.6f}; TN/FP/FN/TP={row.TN}/{row.FP}/{row.FN}/{row.TP}")
    lines += ["", "CONTRASTES PAREADOS; IC95 CONDICIONAL, 2.000 BOOTSTRAPS POR PESSOA"]
    for row in contrasts:
        lines.append(f"{row['escopo']} {row['comparacao']}: ΔBA={row['delta_BA']:+.6f}; IC95=[{row['IC95_inferior']:+.6f}; {row['IC95_superior']:+.6f}]")
    if secondary:
        lines += ["", "ESTABILIDADE NAS CINCO SEMENTES SECUNDÁRIAS"]
        for arm, group in summary[summary.seed.isin(secondary)].groupby("arm"):
            lines.append(f"{arm}: BA média={group.BA.mean():.6f}; mínimo={group.BA.min():.6f}; máximo={group.BA.max():.6f}; DP={group.BA.std():.6f}")
    lines += ["", "TREINO APENAS COMO DIAGNÓSTICO"]
    for (arm, unit), group in folds[folds.phase == "treino"].groupby(["arm", "unit"]):
        lines.append(f"{arm}/{unit}: BA média entre ajustes={group.BA.mean():.6f}")
    lines += ["", "LIMITAÇÕES", "AVFAD já explorada: desenvolvimento, não teste independente.",
              "RMS pode remover informação de intensidade; remover logenergia também remove informação espectral.",
              "A ablação sem logenergia não é uma reprodução da normalização RMS.",
              "ICs condicionais não refazem seleção/treino nem corrigem tentativas históricas.",
              "Nenhuma conclusão clínica, exclusão por desempenho ou avaliação SVD foi realizada.\n"]
    EXP17.text_write(out / "RELATORIO_RESULTADOS.txt", "\n".join(lines))
    names = list(exports) + ["RELATORIO_RESULTADOS.txt", "assinatura_execucao.json", "particoes_participantes.csv", "PROTOCOLO.txt"]
    EXP17.json_write(out / "ARTEFATOS_COMPLETOS.json", {name: EXP17.sha(out/name) for name in names})
    log("Consolidação24 concluída; todos os braços e resultados negativos preservados.")


def selftest():
    import unittest
    spec = importlib.util.spec_from_file_location("test_wpt_normalizacao24", HERE / "test_wpt_normalizacao24.py")
    tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(tests)
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path, default=ROOT / OUT_REL)
    parser.add_argument("--dados23", type=Path)
    parser.add_argument("--referencia17", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true")
    action.add_argument("--executar", action="store_true")
    action.add_argument("--autoteste", action="store_true")
    args = parser.parse_args()
    if args.autoteste: selftest(); return
    root, out, seeds = args.dados.resolve(), args.saida.resolve(), tuple(args.seeds)
    if not out.is_relative_to(root / "results/experiments") or out == root / "results/experiments": parser.error("Saída deve ser subpasta de results/experiments")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s > 2**32-1 for s in seeds): parser.error("Sementes devem ser únicas e válidas")
    data23, ref17 = (args.dados23 or root / DATA23_REL).resolve(), (args.referencia17 or root / REF17_REL).resolve()
    matrices, meta, names = load_inputs(root, data23)
    r13, r17 = references(root, ref17, meta, seeds)
    ident = identity(root, data23, ref17, seeds)
    print(f"ENTRADAS24 OK: 411 pessoas, 2465 gravações; dimensões {[matrices[a].shape[1] for a in ARMS]}; seeds={seeds}", flush=True)
    if not args.executar: print("Somente verificação, sem treino ou escrita.", flush=True); return
    signature_file = out / "assinatura_execucao.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf8")) != ident: raise RuntimeError("Configuração/dados/código mudaram; preservar pasta")
    else:
        if out.exists() and any(p.name != "PROTOCOLO.txt" for p in out.iterdir()): raise RuntimeError("Pasta sem assinatura contém artefatos; preservar")
        out.mkdir(parents=True, exist_ok=True)
        EXP17.freeze_text(out / "PROTOCOLO.txt", protocol()); EXP17.json_write(signature_file, ident)
    EXP17.freeze_text(out / "PROTOCOLO.txt", protocol())
    EXP17.freeze_text(out / "particoes_participantes.csv", EXP19.fold_table(meta, seeds).to_csv(index=False, lineterminator="\n"))
    signature = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    log = EXP17.Logger(out / "LOG_EXECUCAO.txt")
    log(f"Início/retomada24; seeds={seeds}; três braços fixos; sem busca ou novas exclusões.")
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            if (destination / "CONCLUIDO.json").exists():
                frames, arrays = load_checkpoint(destination, signature)
                validate(frames, arrays, meta, train, test, names, seed, fold, r13, r17)
                log(f"Checkpoint24 seed={seed} fold={fold} confirmado; nenhum reajuste"); continue
            if destination.exists() and any(destination.iterdir()): raise RuntimeError(f"Checkpoint parcial preservado: {destination}")
            started = time.monotonic()
            frames, arrays = evaluate_fold(matrices, meta, names, train, test, seed, fold, log, r13, r17)
            validate(frames, arrays, meta, train, test, names, seed, fold, r13, r17)
            save_checkpoint(destination, frames, arrays, signature)
            log(f"Concluído24 seed={seed} fold={fold}; segundos={time.monotonic()-started:.2f}")
    consolidate(out, meta, seeds, log)


if __name__ == "__main__":
    main()
