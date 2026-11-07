# -*- coding: utf-8 -*-
"""Exp17: fatorial fixo normalizacao quantilica x kernel, WPT sym5 AVFAD."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp13_for_exp17", HERE / "13_otimizar_wpt_sym5.py")
EXP13 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP13)
VERSION = "17.1"
SEEDS = (20261105, 101, 102, 103, 104, 105)
ARMS = ("A_STANDARD_LINEAR", "B_QUANTILE_LINEAR", "C_STANDARD_RBF", "D_QUANTILE_RBF")
CONFIG = {"seeds": list(SEEDS), "outer_folds": 5, "inner_folds": 0,
          "familia": "WPT_sym5_L5_L6", "input_features": 1920,
          "k": 128, "C": 1.0, "threshold": 0.0, "class_weight": "balanced",
          "quantiles": 256, "quantile_output": "normal", "quantile_subsample": None,
          "quantile_seed": 42, "gamma_rbf": "scale", "bootstrap_iterations": 2000,
          "selection_unit": "gravacao", "evaluation_unit": "participante",
          "aggregate": "mean decision_function", "arms": list(ARMS)}
ARTIFACTS = {"predicoes.csv", "metricas_treino.csv", "atributos.csv", "support_vectors.csv",
             "limites_quantile.csv", "parametros_ajuste.npz"}
DEFAULT_OUTPUT = ROOT / "results/experiments/resultados_17_robustez_distribuicao_wpt"


def now():
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)


def text_write(path, content):
    atomic_bytes(path, content.encode("utf-8"))


def json_write(path, content):
    text_write(path, json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def csv_write(path, content):
    text_write(path, content.to_csv(index=False, float_format="%.17g", lineterminator="\n"))


def freeze_text(path, content):
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Protocolo/diagnostico congelado diverge: {path}")
    else:
        text_write(path, content)


class Logger:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self, message):
        line = f"[{now()}] {message}"
        print(line, flush=True)
        with self.filename.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_inputs(root):
    x, meta, _ = EXP13.read_inputs(root)
    with np.load(root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz", allow_pickle=False) as data:
        names = data["feature_names"].astype(str)
        names = names[np.char.startswith(names, "WPT__sym5__")]
    if not np.isfinite(x).all():
        raise ValueError("A matriz auditada do Exp17 deve ser inteiramente finita")
    if meta.duplicated(["participant_id", "task"]).any():
        raise ValueError("Tarefa repetida por pessoa")
    return x, meta, names


def protocol():
    return """EXPERIMENTO 17 - PROTOCOLO ANTERIOR AOS NOVOS AJUSTES
Data de definicao: 05/11/2026, America/Sao_Paulo.

HIPOTESE
As caudas e a assimetria dos descritores WPT, sobretudo energia relativa,
podem prejudicar o ranking ANOVA e a geometria do classificador apos simples
padronizacao. Testar uma transformacao marginal quantilica e, como segundo
fator independente, um kernel nao linear, sem ampliar busca de hiperparametros.
Diagnostico descritivo de distribuicoes nao usa rotulos nem mede BA.
Nao se presume ganho. Todos os bracos e resultados negativos sao mantidos.

COORTE E REPRESENTACAO
AVFAD HSA: 411 participantes, 82 normais e 329 patologicos, 2465 gravacoes.
410 pessoas com seis frases CAPE-V e uma com cinco. Nenhuma exclusao nova.
Unica representacao: WPT temporal sym5 L5+L6, 1920 atributos por gravacao.
Nao alterar WAV, VAD, janelas, amplitude, cache nem matrizes existentes.

FATORIAL 2x2, QUATRO CONFIGURACOES FIXAS
A_STANDARD_LINEAR: imputer mediana -> constantes -> ANOVA k128 ->
StandardScaler -> SVC linear C1, class_weight balanced, limiar0.
B_QUANTILE_LINEAR: igual a A, inserindo QuantileTransformer ANTES da ANOVA.
C_STANDARD_RBF: igual a A, apenas SVC RBF C1/gamma=scale.
D_QUANTILE_RBF: igual a B, apenas SVC RBF C1/gamma=scale.
QuantileTransformer: n_quantiles=256, output_distribution=normal,
subsample=None (todas as linhas de treino), random_state=42.
Para autoteste pequeno apenas, n_quantiles=min(256,n_linhas_treino).
Todos usam k128, C1, balanceamento por classe de gravacao e limiar zero.
Nao ha ajuste de limiar nem escolha de k, C, kernel ou representacao no teste.
Sem SMOTE. Sem ajuste por participante da ANOVA neste experimento.

VALIDACAO
5 folds externos estratificados numa tabela de participantes; todas as frases
da pessoa no mesmo lado. Seeds 20261105 (primaria) e 101,102,103,104,105
(estabilidade secundaria). Mesmos folds para todos os quatro bracos.
Nao escolher melhor seed. Nao sao seis coortes independentes.
Sem CV interna: os quatro modelos sao fixos e todos serao reportados.
Imputacao, constantes, quantis, ANOVA, escala e SVC ajustados exclusivamente
no treino externo. O teste recebe apenas transform, incluindo clipping
quantilico fora da faixa de treino; esses casos serao contados, nao removidos.
Uma decisao por participante: media das margens das frases, seguida de zero.
Controle A deve reproduzir IDs, folds, decisoes e scores do Exp13 na seed
primaria, verificado por fold antes de continuar os outros bracos.

DESFECHOS E CONTRASTES
Primario: B-A em BA por participante na seed20261105.
Secundarios: D-C (quantis sob RBF), D-B (kernel sob quantis), C-A (kernel
sem quantis). Interacao (D-C)-(B-A) e apenas descritiva/exploratoria.
Reportar separadamente seed primaria e estabilidade101..105.
Sensibilidade, especificidade, acuracia, matriz de confusao e BA por fold/seed.
AUC de apoio: media por fold ponderada por numero de pares normal-patologico.
AUC OOF concatenada e exploratoria porque os scores entre folds nao sao
calibrados. B e D devem ter mesmos quantis e atributos; A e C mesmos atributos.
IC95%:2000 bootstraps estratificados pareados POR PESSOA. Reamostrar sua
sequencia completa de predicoes entre seeds, sem multiplicar N por repeticoes.
Os ICs sao condicionais aos modelos/folds e nao refazem selecao/treinamento.
Nao eliminam adaptacao historica nem corrigem todas as tentativas da coorte.

DIAGNOSTICOS E RASTREABILIDADE
Registrar metricas de treino por gravacao e participante para descrever
sobreajuste, contagem de vetores de suporte e exemplos correspondentes,
atributos/ranking, parametros dos quantis e escala, e extremos do teste fora
do intervalo de treino. Esses diagnosticos nao comandam exclusoes ou ajustes.
Salvar hashes SHA256 de insumos, codigo/dependencias locais, versoes,
protocolo, particoes e checkpoints atomicos, logs e relatorio TXT.

INTERPRETACAO
Resultados de desenvolvimento nos mesmos 411 individuos ja explorados.
Nao e teste externo nem conclusao final do TG. Nao instalar dependencias,
nao avaliar SVD e nao usar atribuicoes causais ou clinicas a partir de ranking.
Quantis podem reduzir a influencia das caudas, mas tambem remover informacao
de magnitude util; RBF pode aumentar sobreajuste. Ambos sao possibilidades
a avaliar, nao resultados antecipados. Nenhum ganho sera selecionado pela
melhor divisao ou promovido a validacao independente.
"""


def distributions(x, names):
    rows = []
    for descriptor in ("rel_energy", "log_energy", "entropy", "kurtosis", "zcr"):
        for stat in ("mean", "std", "q10", "q90"):
            take = np.asarray([f"__{descriptor}__" in name and name.endswith(f"__{stat}") for name in names])
            values = x[:, take].astype(np.float64)
            deviation = values.std(axis=0)
            usable = deviation > 1e-14
            values = values[:, usable]
            deviation = deviation[usable]
            asymmetry = np.abs(skew(values, axis=0))
            standardized = np.abs((values - values.mean(axis=0)) / deviation)
            spread = np.quantile(values, .75, axis=0) - np.quantile(values, .25, axis=0)
            rows.append({"descriptor": descriptor, "resumo": stat, "n_colunas": int(take.sum()),
                         "mediana_abs_skew": float(np.median(asymmetry)),
                         "p95_abs_skew": float(np.quantile(asymmetry, .95)),
                         "fracao_colunas_maxabsz_maior8": float(np.mean(standardized.max(axis=0) > 8)),
                         "mediana_dp_sobre_iqr": float(np.median(deviation / (spread + 1e-12)))})
    deviation = x.std(axis=0, dtype=np.float64)
    standardized = np.abs((x - x.mean(axis=0, dtype=np.float64)) / deviation)
    header = ["DIAGNOSTICO DE DISTRIBUICOES WPT SYM5 - SEM USAR ROTULOS", "",
              f"Matriz:{x.shape}; inteiramente finita:{bool(np.isfinite(x).all())}.",
              f"Fracao de colunas com max|z|>8:{np.mean(standardized.max(axis=0)>8):.8f}.",
              f"Fracao de gravacoes com algum |z|>8:{np.mean((standardized>8).any(axis=1)):.8f}.",
              "Descricoes calculadas sobre a matriz existente inteira; nao sao ajustes",
              "do pre-processamento usado no teste. Cada QuantileTransformer sera",
              "ajustado exclusivamente no treino do respectivo fold.",
              "Esses valores NAO sao criterio de exclusao de participantes ou audios.",
              "A assimetria mais forte ocorre em energia relativa. RobustScaler",
              "sozinho nao remove extremos: muda a unidade pela faixa interquartil.",
              "Ver diagnostico_distribuicoes_sem_rotulos.csv para todos os descritores.",
              "Nenhum classificador foi ajustado para produzir este diagnostico.\n"]
    return pd.DataFrame(rows), "\n".join(header)


def document(out, x, names):
    frame, description = distributions(x, names)
    freeze_text(out / "PROTOCOLO.txt", protocol())
    freeze_text(out / "DIAGNOSTICO_DISTRIBUICOES.txt", description)
    freeze_text(out / "diagnostico_distribuicoes_sem_rotulos.csv",
                frame.to_csv(index=False, float_format="%.17g", lineterminator="\n"))


def identity(root, out):
    paths = [root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz",
             root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv",
             root / "results/experiments/resultados_13_otimizacao_wpt_sym5/predicoes_oof_participante.csv"]
    code = [Path(__file__).resolve(), HERE / "13_otimizar_wpt_sym5.py"]
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
        environment[name] = importlib.metadata.version(name)
    return {"version": VERSION, "configuration": CONFIG, "environment": environment,
            "input_files": {str(p.resolve()): {"sha256": sha(p), "bytes": p.stat().st_size} for p in paths},
            "code": {str(p): sha(p) for p in code},
            "documents": {name: sha(out / name) for name in (
                "PROTOCOLO.txt", "DIAGNOSTICO_DISTRIBUICOES.txt", "diagnostico_distribuicoes_sem_rotulos.csv")}}


def model_for(arm, n_train):
    kind = "SVC_RBF" if arm in ARMS[2:] else "SVC_linear"
    baseline = EXP13.pipeline_for(kind, 128, 1.0)
    if arm in (ARMS[1], ARMS[3]):
        steps = baseline.steps.copy()
        steps.insert(2, ("quantile", QuantileTransformer(n_quantiles=min(256, n_train),
                      output_distribution="normal", subsample=None, random_state=42)))
        return Pipeline(steps)
    return baseline


def collapse(meta, score):
    result = EXP13.aggregate(meta.reset_index(drop=True), score)
    result["predicao"] = (result.score >= 0).astype(int)
    return result


def measures(frame):
    tn, fp, fn, tp = confusion_matrix(frame.y, frame.predicao, labels=[0, 1]).ravel()
    return {"n": len(frame), "BA": float(balanced_accuracy_score(frame.y, frame.predicao)),
            "acuracia": float(accuracy_score(frame.y, frame.predicao)),
            "sensibilidade": float(tp / (tp + fn)), "especificidade": float(tn / (tn + fp)),
            "AUC": float(roc_auc_score(frame.y, frame.score)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def compare_reference(prediction, reference, seed, fold):
    if seed != SEEDS[0] or reference is None:
        return
    expected = reference[reference.outer_fold == fold].set_index("participant_id").sort_index()
    current = prediction.set_index("participant_id").sort_index()
    pd.testing.assert_index_equal(expected.index, current.index)
    for column in ("y", "predicao", "n_gravacoes"):
        np.testing.assert_array_equal(expected[column], current[column])
    np.testing.assert_allclose(expected.score, current.score, atol=1e-9, rtol=1e-9)


def parameters(model):
    step = model.named_steps
    return {"imputer": step["imputer"].statistics_.copy(),
            "constant_mask": step["remove_constants"].get_support(),
            "selector_scores": step["select"].scores_.copy(),
            "selector_mask": step["select"].get_support(),
            "scaler_mean": step["scale"].mean_.copy(),
            "scaler_scale": step["scale"].scale_.copy(),
            "support_indices": step["classifier"].support_.copy(),
            "n_support": step["classifier"].n_support_.copy(),
            "gamma": np.asarray(step["classifier"]._gamma)}


def evaluate_fold(x, meta, names, train, test, seed, fold, log, reference=None):
    train_meta = meta.iloc[train].reset_index(drop=True)
    test_meta = meta.iloc[test].reset_index(drop=True)
    if set(train_meta.participant_id) & set(test_meta.participant_id):
        raise ValueError("Participante compartilhado no fold externo")
    predictions, training, attributes, supports, bounds = [], [], [], [], []
    fitted, arrays = {}, {}
    for arm in ARMS:
        started = time.monotonic()
        model = model_for(arm, len(train))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(x[train], train_meta.y.to_numpy())
        test_scores = model.decision_function(x[test])
        pred = collapse(test_meta, test_scores)
        pred["arm"], pred["seed"], pred["fold"], pred["limiar"] = arm, seed, fold, 0.0
        if arm == ARMS[0]:
            compare_reference(pred, reference, seed, fold)
            if seed == SEEDS[0] and reference is not None:
                log(f"Controle A reproduziu Exp13 seed={seed} fold={fold}.")
        predictions.append(pred)
        train_scores = model.decision_function(x[train])
        recordings = train_meta[["participant_id", "y"]].copy()
        recordings["score"], recordings["predicao"] = train_scores, (train_scores >= 0).astype(int)
        people = collapse(train_meta, train_scores)
        svc = model.named_steps["classifier"]
        for level, frame in (("gravacao", recordings), ("participante", people)):
            training.append({"arm": arm, "seed": seed, "fold": fold, "unidade": level,
                             **measures(frame), "n_support_0": int(svc.n_support_[0]),
                             "n_support_1": int(svc.n_support_[1]), "n_linhas_treino": len(train),
                             "segundos_ajuste_avaliacao": time.monotonic() - started})
        fitted[arm] = parameters(model)
        for key, value in fitted[arm].items():
            arrays[f"{arm}__{key}"] = value
        var = model.named_steps["remove_constants"].get_support()
        sel = model.named_steps["select"].get_support()
        selected_indices = np.flatnonzero(var)[sel]
        attributes.extend({"arm": arm, "seed": seed, "fold": fold, "indice": int(index),
                           "atributo": str(names[index]),
                           "F_treino": float(model.named_steps["select"].scores_[sel][j])}
                          for j, index in enumerate(selected_indices))
        supports.extend({"arm": arm, "seed": seed, "fold": fold, "source_row": int(train[index]),
                         "sample_id": train_meta.sample_id.iloc[index],
                         "participant_id": train_meta.participant_id.iloc[index],
                         "y": int(train_meta.y.iloc[index])} for index in svc.support_)
        if arm == ARMS[1]:
            quantile = model.named_steps["quantile"]
            arrays["quantile__quantiles"] = quantile.quantiles_.copy()
            arrays["quantile__references"] = quantile.references_.copy()
            arrays["quantile__source_indices"] = np.flatnonzero(var)
            imputed_test = model.named_steps["imputer"].transform(x[test])
            variable_test = model.named_steps["remove_constants"].transform(imputed_test)
            low = np.sum(variable_test < quantile.quantiles_[0], axis=0)
            high = np.sum(variable_test > quantile.quantiles_[-1], axis=0)
            bounds.extend({"seed": seed, "fold": fold, "indice": int(index), "atributo": str(names[index]),
                           "minimo_treino": float(quantile.quantiles_[0, j]),
                           "maximo_treino": float(quantile.quantiles_[-1, j]),
                           "n_teste_abaixo": int(low[j]), "n_teste_acima": int(high[j]),
                           "n_gravacoes_teste": len(test)} for j, index in enumerate(np.flatnonzero(var)))
        if arm == ARMS[3]:
            np.testing.assert_array_equal(model.named_steps["quantile"].quantiles_, arrays["quantile__quantiles"])
        log(f"seed={seed} fold={fold} {arm} BA_teste_pessoa={measures(pred)['BA']:.6f}; "
            f"support={len(svc.support_)}/{len(train)}; segundos={time.monotonic()-started:.2f}")
    for left, right in ((ARMS[0], ARMS[2]), (ARMS[1], ARMS[3])):
        for key in ("imputer", "constant_mask", "selector_scores", "selector_mask", "scaler_mean", "scaler_scale"):
            np.testing.assert_array_equal(fitted[left][key], fitted[right][key])
    return {"predicoes.csv": pd.concat(predictions, ignore_index=True),
            "metricas_treino.csv": pd.DataFrame(training), "atributos.csv": pd.DataFrame(attributes),
            "support_vectors.csv": pd.DataFrame(supports), "limites_quantile.csv": pd.DataFrame(bounds)}, arrays


def validate(frames, meta, train, test, seed, fold):
    pred = frames["predicoes.csv"]
    expected = meta.iloc[test].drop_duplicates("participant_id").set_index("participant_id").y.to_dict()
    counts = meta.iloc[test].groupby("participant_id").size().to_dict()
    if set(pred.arm) != set(ARMS):
        raise ValueError("Bracos incompletos")
    for arm, group in pred.groupby("arm"):
        if group.participant_id.duplicated().any() or group.set_index("participant_id").y.to_dict() != expected:
            raise ValueError("IDs/rotulos inesperados no checkpoint")
        if group.set_index("participant_id").n_gravacoes.to_dict() != counts:
            raise ValueError("Contagem de frases diverge")
        if not ((group.seed == seed) & (group.fold == fold) & (group.limiar == 0)).all():
            raise ValueError("Seed/fold/limiar invalido")
        if not np.isfinite(group.score).all() or not np.array_equal(group.predicao, (group.score >= 0).astype(int)):
            raise ValueError("Scores/decisoes invalidos")
    attrs = frames["atributos.csv"]
    for arm in ARMS:
        selected = attrs[attrs.arm == arm]
        if len(selected) != 128 or selected.indice.duplicated().any():
            raise ValueError("Selecao nao tem exatamente128 atributos distintos")
    if not set(frames["support_vectors.csv"].participant_id).issubset(set(meta.iloc[train].participant_id)):
        raise ValueError("Vetor de suporte fora do treino")
    if len(frames["metricas_treino.csv"]) != 8:
        raise ValueError("Metricas de treino incompletas")


def save_checkpoint(destination, frames, arrays, run_signature):
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        csv_write(destination / name, frame)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(destination / "parametros_ajuste.npz", buffer.getvalue())
    json_write(destination / "CONCLUIDO.json", {"finished": now(), "run_signature": run_signature,
               "sha256": {name: sha(destination / name) for name in ARTIFACTS}})


def load_checkpoint(destination, run_signature):
    marker = json.loads((destination / "CONCLUIDO.json").read_text(encoding="utf-8"))
    if marker["run_signature"] != run_signature or set(marker["sha256"]) != ARTIFACTS:
        raise ValueError("Assinatura/lista de artefatos diverge no checkpoint")
    if any(sha(destination / name) != digest for name, digest in marker["sha256"].items()):
        raise ValueError("Conteudo do checkpoint foi alterado")
    return {name: pd.read_csv(destination / name, float_precision="round_trip") for name in ARTIFACTS if name.endswith(".csv")}


def intervals(raw, seeds, coefficients, iterations=2000):
    sub = raw[raw.seed.isin(seeds)]
    ids = sorted(sub.participant_id.unique())
    labels = sub.drop_duplicates("participant_id").set_index("participant_id").loc[ids].y.to_numpy()
    deltas = []
    for seed in seeds:
        changes = np.zeros(len(ids))
        for arm, coefficient in coefficients.items():
            frame = sub[(sub.seed == seed) & (sub.arm == arm)].set_index("participant_id").loc[ids]
            changes += coefficient * (frame.predicao.to_numpy() == labels)
        deltas.append(changes)
    changes = np.mean(deltas, axis=0)
    normal, pathological = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    rng = np.random.default_rng(20261105)
    boot = (changes[rng.choice(normal, (iterations, len(normal)), replace=True)].mean(axis=1)
            + changes[rng.choice(pathological, (iterations, len(pathological)), replace=True)].mean(axis=1)) / 2
    low, high = np.quantile(boot, [.025, .975])
    return {"delta_BA": float((changes[normal].mean() + changes[pathological].mean()) / 2),
            "IC95_inferior": float(low), "IC95_superior": float(high)}


def consolidate(out, meta, log):
    folders = [out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" for seed in SEEDS for fold in range(1, 6)]
    pred = pd.concat([pd.read_csv(folder / "predicoes.csv", float_precision="round_trip") for folder in folders], ignore_index=True)
    train = pd.concat([pd.read_csv(folder / "metricas_treino.csv") for folder in folders], ignore_index=True)
    attrs = pd.concat([pd.read_csv(folder / "atributos.csv") for folder in folders], ignore_index=True)
    fold_metrics = []
    for (seed, arm, fold), frame in pred.groupby(["seed", "arm", "fold"]):
        fold_metrics.append({"seed": seed, "arm": arm, "fold": fold, **measures(frame)})
    folds = pd.DataFrame(fold_metrics)
    summaries = []
    for (seed, arm), frame in pred.groupby(["seed", "arm"]):
        if frame.participant_id.duplicated().any() or set(frame.participant_id) != set(meta.participant_id):
            raise ValueError("OOF nao contem uma predicao por pessoa/seed/braco")
        result = measures(frame)
        result["AUC_OOF_exploratoria"] = result.pop("AUC")
        relevant = folds[(folds.seed == seed) & (folds.arm == arm)]
        result["AUC_media_folds_ponderada_pares"] = float(np.average(relevant.AUC, weights=(relevant.TN + relevant.FP) * (relevant.TP + relevant.FN)))
        summaries.append({"seed": seed, "arm": arm, **result})
    summary = pd.DataFrame(summaries)
    contrasts = [("B-A_primaria", {ARMS[1]: 1, ARMS[0]: -1}),
                 ("D-C_secundaria", {ARMS[3]: 1, ARMS[2]: -1}),
                 ("D-B_secundaria", {ARMS[3]: 1, ARMS[1]: -1}),
                 ("C-A_secundaria", {ARMS[2]: 1, ARMS[0]: -1}),
                 ("interacao_exploratoria", {ARMS[3]: 1, ARMS[2]: -1, ARMS[1]: -1, ARMS[0]: 1})]
    comparisons = []
    for scope, seeds in (("principal_seed20261105", [SEEDS[0]]), ("estabilidade101_105", list(SEEDS[1:]))):
        for label, coefficients in contrasts:
            comparisons.append({"escopo": scope, "comparacao": label, **intervals(pred, seeds, coefficients)})
    scopes = []
    for scope, seeds in (("principal_seed20261105", [SEEDS[0]]), ("estabilidade101_105", list(SEEDS[1:]))):
        for arm, frame in summary[summary.seed.isin(seeds)].groupby("arm"):
            scopes.append({"escopo": scope, "arm": arm, "BA_media": frame.BA.mean(), "BA_min": frame.BA.min(),
                           "BA_max": frame.BA.max(), "BA_DP_seeds": frame.BA.std(),
                           "sensibilidade_media": frame.sensibilidade.mean(), "especificidade_media": frame.especificidade.mean(),
                           "acuracia_media": frame.acuracia.mean(),
                           "AUC_media_folds": frame.AUC_media_folds_ponderada_pares.mean()})
    scopes = pd.DataFrame(scopes)
    for name, frame in (("predicoes_oof_participante.csv", pred), ("metricas_por_fold.csv", folds),
                        ("metricas_por_seed.csv", summary), ("metricas_treino_todos_folds.csv", train),
                        ("atributos_selecionados.csv", attrs), ("resumo_final.csv", scopes),
                        ("comparacoes_pareadas.csv", pd.DataFrame(comparisons))):
        csv_write(out / name, frame)
    report = ["EXPERIMENTO 17 - RESULTADOS DE DESENVOLVIMENTO AVFAD", f"Consolidado: {now()}", "",
              "Estado: quatro bracos fixos, seis seeds e todos30folds concluidos.",
              "BA principal por participante, uma decisao por pessoa/seed/braco.",
              "Controle A reproduz Exp13 na seed primaria; demais seeds sao estabilidade.",
              "A:standard/linear; B:quantis/linear; C:standard/RBF; D:quantis/RBF.",
              "Todos C1,k128,limiar0; quantis antesANOVA, apenas no treino.", ""]
    for row in scopes.itertuples(index=False):
        report.append(f"{row.escopo} {row.arm}: BA={row.BA_media:.6f}; min={row.BA_min:.6f}; max={row.BA_max:.6f}; "
                      f"sens={row.sensibilidade_media:.6f}; esp={row.especificidade_media:.6f}; "
                      f"acuracia={row.acuracia_media:.6f}; AUC_fold={row.AUC_media_folds:.6f}")
    report += ["", "CONTRASTES PAREADOS POR PESSOA; IC CONDICIONAL AOS OOF"]
    for row in comparisons:
        report.append(f"{row['escopo']} {row['comparacao']}: deltaBA={row['delta_BA']:+.6f}; "
                      f"IC95=[{row['IC95_inferior']:+.6f},{row['IC95_superior']:+.6f}]")
    report += ["", "SOBREAJUSTE DESCRITIVO (media das metricas de treino por fold, nao OOF)"]
    for (arm, unit), frame in train.groupby(["arm", "unidade"]):
        report.append(f"{arm} treino/{unit}: BA_media={frame.BA.mean():.6f}; "
                      f"suportes_media={(frame.n_support_0+frame.n_support_1).mean():.2f}; "
                      f"fracao_suportes={((frame.n_support_0+frame.n_support_1)/frame.n_linhas_treino).mean():.6f}")
    report += ["", "LIMITACOES", "AVFAD ja explorada: nao e teste externo nem conclusao final do TG.",
               "As repeticoes reutilizam pessoas; DP de seeds nao e IC. Bootstrap reamostra pessoa",
               "com todas suas repeticoes; nao refaz treinamento e nao corrige todas as tentativas.",
               "Nenhum modelo/semente foi escolhido para omitir demais resultados.",
               "Transformacao quantilica pode reduzir caudas ou perder informacao de magnitude.",
               "Ranking ANOVA e treino continuam por gravacao; class_weight balanceia classes",
               "de gravacoes, nao exatamente pessoas com numeros diferentes de frases.",
               "AUC principal de apoio e media por fold ponderada por pares normal-patologico;",
               "AUC OOF concatenada so exploratoria. Margens entre modelos nao sao calibradas.",
               "Contagens fora dos limites de quantis estao em limites_quantile.csv por checkpoint.",
               "Parametros dos quantis/seletores/escalas foram salvos sem dados de teste no NPZ.",
               "Nenhum dado, cache, resultado antigo, audio ou arquivo SVD foi alterado.\n"]
    text_write(out / "RELATORIO_RESULTADOS.txt", "\n".join(report))
    log("Consolidacao TXT/CSV concluida. Nenhum modelo final escolhido por melhor seed.")


def fold_tables(meta):
    rows = []
    for seed in SEEDS:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            for role, indices in (("treino", train), ("teste", test)):
                for row in meta.iloc[indices].drop_duplicates("participant_id").itertuples():
                    rows.append({"seed": seed, "fold": fold, "role": role, "participant_id": row.participant_id, "y": row.y})
    return pd.DataFrame(rows)


def selftest():
    rng = np.random.default_rng(1705)
    meta = pd.DataFrame([{"participant_id": f"S{pid:03d}", "sample_id": f"S{pid:03d}_{task}",
                          "y": pid % 2, "task": str(task)} for pid in range(36) for task in range(3)])
    x = rng.lognormal(size=(len(meta), 160)).astype(np.float32)
    x[:, -1] = 1
    names = np.asarray([f"sintetico_{i}" for i in range(x.shape[1])])
    train, test = next(EXP13.row_splits(meta, 5, SEEDS[0]))
    data, arrays = evaluate_fold(x, meta, names, train, test, SEEDS[0], 1, lambda _: None)
    altered = x.copy()
    altered[test] = rng.normal(1e6, 1e5, size=altered[test].shape)
    changed_meta = meta.copy()
    changed_meta.loc[test, "y"] = 1 - changed_meta.loc[test, "y"]
    again, changed_arrays = evaluate_fold(altered, changed_meta, names, train, test, SEEDS[0], 1, lambda _: None)
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, changed_arrays[key])
    for filename in ("atributos.csv", "support_vectors.csv"):
        pd.testing.assert_frame_equal(data[filename], again[filename])
    pd.testing.assert_frame_equal(data["metricas_treino.csv"].drop(columns="segundos_ajuste_avaliacao"),
                                  again["metricas_treino.csv"].drop(columns="segundos_ajuste_avaliacao"))
    validate(data, meta, train, test, SEEDS[0], 1)
    with TemporaryDirectory(prefix="avfad_exp17_test_") as temporary:
        directory = Path(temporary)
        save_checkpoint(directory, data, arrays, "sintetico")
        loaded = load_checkpoint(directory, "sintetico")
        validate(loaded, meta, train, test, SEEDS[0], 1)
        before = {name: sha(directory / name) for name in ARTIFACTS}
        load_checkpoint(directory, "sintetico")
        assert before == {name: sha(directory / name) for name in ARTIFACTS}
        corrupted = loaded["predicoes.csv"].copy()
        corrupted.loc[0, "y"] = 1 - corrupted.loc[0, "y"]
        broken = dict(loaded, **{"predicoes.csv": corrupted})
        try:
            validate(broken, meta, train, test, SEEDS[0], 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupcao de rotulo aceita")
    prediction = data["predicoes.csv"]
    first = intervals(prediction, [SEEDS[0]], {ARMS[1]: 1, ARMS[0]: -1}, iterations=500)
    copied = prediction.assign(seed=999)
    repeated = intervals(pd.concat([prediction, copied]), [SEEDS[0], 999], {ARMS[1]: 1, ARMS[0]: -1}, iterations=500)
    assert first == repeated
    print("AUTOTESTE OK: testeX/y nao altera ajustes/ranking/suportes; fatores compartilham pre-processamento; "
          "checkpoint/roundtrip/corrupcao e bootstrap pessoa-seed verificados.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path, default=DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true", help="Leitura apenas, sem ajuste ou escrita")
    action.add_argument("--preparar", action="store_true", help="Grava so protocolo/diagnostico sem modelos")
    action.add_argument("--executar", action="store_true", help="Executa/retoma os modelos fixos")
    action.add_argument("--autoteste", action="store_true", help="Dados sinteticos temporarios, sem AVFAD")
    args = parser.parse_args()
    if args.autoteste:
        selftest()
        return
    root, out = args.dados.resolve(), args.saida.resolve()
    if not out.is_relative_to(root / "results/experiments"):
        parser.error("Pasta de saida deve estar em results/experiments")
    x, meta, names = load_inputs(root)
    print(f"ENTRADAS OK:{x.shape}; pessoas={meta.participant_id.nunique()}; seeds={SEEDS}", flush=True)
    if not args.executar and not args.preparar:
        print("VERIFICACAO OK. Sem treino ou escrita.", flush=True)
        return
    document(out, x, names)
    if args.preparar:
        print("Protocolo e diagnostico TXT/CSV registrados; nenhum modelo ajustado.", flush=True)
        return
    ident = identity(root, out)
    signature = out / "assinatura_execucao.json"
    if signature.exists():
        if json.loads(signature.read_text(encoding="utf-8")) != ident:
            raise RuntimeError("Dados/codigo/ambiente/protocolo divergem; preservar pasta e revisar")
    else:
        allowed = {"PROTOCOLO.txt", "DIAGNOSTICO_DISTRIBUICOES.txt", "diagnostico_distribuicoes_sem_rotulos.csv"}
        if any(path.name not in allowed for path in out.iterdir()):
            raise RuntimeError("Pasta nao vazia sem assinatura; nao reutilizar")
        json_write(signature, ident)
    run_signature = hashlib.sha256(json.dumps(ident, sort_keys=True).encode("utf-8")).hexdigest()
    log = Logger(out / "LOG_EXECUCAO.txt")
    log("Inicio/retomada Exp17; todos os bracos e seis sementes fixados antes de ajustar modelos.")
    csv_write(out / "particoes_participantes.csv", fold_tables(meta))
    reference = pd.read_csv(root / "results/experiments/resultados_13_otimizacao_wpt_sym5/predicoes_oof_participante.csv")
    reference = reference[reference.configuracao == "WPT_fixa"]
    for seed in SEEDS:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            if (destination / "CONCLUIDO.json").exists():
                frames = load_checkpoint(destination, run_signature)
                validate(frames, meta, train, test, seed, fold)
                compare_reference(frames["predicoes.csv"][frames["predicoes.csv"].arm == ARMS[0]], reference, seed, fold)
                log(f"seed={seed} fold={fold} checkpoint verificado, sem repetir ajustes.")
                continue
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Checkpoint parcial preservado: {destination}")
            log(f"INICIO seed={seed} fold={fold}/5")
            frames, arrays = evaluate_fold(x, meta, names, train, test, seed, fold, log, reference)
            validate(frames, meta, train, test, seed, fold)
            save_checkpoint(destination, frames, arrays, run_signature)
            log(f"CONCLUIDO seed={seed} fold={fold}/5")
    consolidate(out, meta, log)


if __name__ == "__main__":
    main()
