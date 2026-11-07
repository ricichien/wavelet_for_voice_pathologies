# -*- coding: utf-8 -*-
"""Exp16: desenvolvimento SVD com coorte canônica, folds e métricas por pessoa.

Não transfere o modelo AVFAD. Lê CSVs antigos somente para recuperar atributos
de linhas explicitamente aprovadas no manifesto produzido por svd_cohort_manifest.
Use --autoteste antes de --verificar; --executar exige PROTOCOLO.txt congelado.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
import time
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wavelet_participant_selection import ParticipantMeanANOVA

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VERSION = "16.1"
DEFAULT_SOURCE = Path(os.environ.get("AVFAD_SVD_FEATURES_CSV", str(ROOT / "data" / "SVD" / "svd_features_mfcc_wavelet_temporal.csv")))
DEFAULT_OUTPUT = ROOT / "results/experiments/resultados_16_svd_wavelet_participante"
DEFAULT_SEEDS = (20261105, 101, 102, 103, 104, 105)
SOURCES = ("svd_features_mfcc_wavelet_temporal.csv", "svd_features_healthy.csv")
TASKS = {"a_n", "i_n", "u_n"}
COHORTS = ("MATCHED", "FULL")
ARM_CONFIG = {
    "WPTt_ANOVA_pessoa": ("WPTt", "pessoa"),
    "WPTt_ANOVA_gravacao": ("WPTt", "gravacao"),
    "WPTglobal_ANOVA_pessoa": ("WPTglobal", "pessoa"),
    "DWT_ANOVA_pessoa": ("DWT", "pessoa"),
    "MFCC_baseline": ("MFCC", "todos"),
    "DEMOGRAFIA": ("DEMOGRAFIA", "demografia"),
}
ARMS = tuple(ARM_CONFIG)
PRIMARY = "WPTt_ANOVA_pessoa"
COMPARISONS = (
    ("MFCC_baseline", "WPTt_pessoa-MFCC"),
    ("WPTt_ANOVA_gravacao", "WPTt_pessoa-WPTt_gravacao"),
    ("WPTglobal_ANOVA_pessoa", "WPTt_pessoa-WPTglobal"),
    ("DWT_ANOVA_pessoa", "WPTt_pessoa-DWT"),
)
HELPER_FILES = {
    "manifesto_coorte.csv", "pairs.csv", "pares.csv", "exclusoes_manifesto.csv",
    "PROTOCOLO_COORTE.txt", "assinatura_manifesto.json", "RELATORIO_COORTE.txt",
}


def stamp():
    return datetime.now(timezone.utc).astimezone(
        timezone(timedelta(hours=-3))).isoformat(timespec="seconds")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def json_write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def csv_write(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", float_format="%.17g")
    temporary.replace(path)


class Logger:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, message):
        line = f"[{stamp()}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def protocol_text(seeds=DEFAULT_SEEDS):
    return f"""EXPERIMENTO 16 - SVD: WAVELET E AVALIACAO POR PARTICIPANTE
Versao do executor: {VERSION}
Protocolo a congelar antes do primeiro fit SVD deste experimento.

HIPOTESE E ESCOPO
WPT temporal e ranking ANOVA por participante podem generalizar melhor que
MFCC e que ranking por gravacao na mesma coorte SVD. Hipotese de desenvolvimento,
sem ganho presumido. Estes CSVs/SVD ja foram explorados historicamente.
Nao e teste externo cego nem transferencia AVFAD: nao ha sym5 nesses atributos,
as tarefas sao vogais sustentadas e o front-end difere das frases AVFAD.

COORTE
Usar somente manifesto_coorte.csv auditado e gerado antes dos fits.
Uma sessao canonica por pessoa, escolhida por regra cronologica do helper;
exatamente a_n, i_n e u_n da mesma sessao, sem duplicacoes de arquivo.
MATCHED e a coorte primaria: sexo exato, diferenca idade<=3 anos e ano<=1,
pares fixados pelo helper sem usar atributos de voz ou desempenho.
FULL e analise secundaria da coorte canonica completa, sem escolher coorte
depois da BA. Preservar exclusoes e proveniencia do manifesto.

BRACOS FIXOS
WPTt_ANOVA_pessoa: wptt_,1536 atributos, ANOVA nas medias de pessoas do treino.
WPTt_ANOVA_gravacao: mesmos atributos, ANOVA nas linhas; ablacao do ranking.
WPTglobal_ANOVA_pessoa: wpt_,576 atributos, ranking por pessoa.
DWT_ANOVA_pessoa: db4_/sym8_/coif5_,180 atributos, ranking por pessoa.
MFCC_baseline: mfcc_,26 atributos; manter todos apos remover constantes.
Cinco bracos de voz: SVC linear C=1, limiar=0, sem probability/calibracao.
k=128 (limitado ao numero disponivel); MFCC preserva todos os atributos.
DEMOGRAFIA, auxiliar: idade, sexo(m=0,w=1), ano da sessao; sem feature de voz.
Imputacao mediana + StandardScaler + LogisticRegression(C=1,solver=liblinear,
max_iter=2000,class_weight=None). Limiar zero da margem. Nao e modelo Wavelet
nem comparacao primaria; verifica sinal demografico linear residual.

APRENDIZADO E PESOS
Somente treino de cada fold: imputacao mediana, remocao de constantes nos
bracos de voz, ranking quando aplicavel, StandardScaler e classificador.
ANOVA por pessoa muda apenas ranking, aplicado depois as gravacoes originais.
Treinamento por linha; sample_weight balanceia classes pela contagem de
pessoas e divide o peso de cada pessoa por seu numero de linhas. Normalizar
para media1 entre linhas do treino. class_weight=None: nao balancear duas vezes.
Sem SMOTE, tuning, busca de familia, selecao externa ou ajuste de limiar.
Hiperparametros fixos: nao e necessario criar CV interna.
Uma predicao por participante: media das margens das tres vogais, limiar0.

FOLDS
5 folds externos; mesmas pessoas e folds em todos os bracos de cada coorte.
MATCHED: KFold shuffle nos pares; duas pessoas de cada par no mesmo fold.
FULL: StratifiedKFold shuffle na tabela de pessoas, nunca nas linhas.
Semente primaria20261105; estabilidade secundaria101..105.
Sementes efetivas desta execucao: {list(seeds)}.
Todas reportadas. Repeticoes reutilizam pessoas, nao sao novas coortes.

DESFECHOS
Primario: delta BA WPTt_ANOVA_pessoa-MFCC_baseline em MATCHED, seed20261105.
Secundarios: WPTt_pessoa-WPTt_gravacao, WPTt_pessoa-WPTglobal e WPTt_pessoa-DWT;
estabilidade nas seeds101..105; FULL reportada separadamente.
Todos os bracos: BA/sensibilidade/especificidade/matriz por participante.
AUC por fold e media ponderada por pares normal-patologico; concatenada
somente exploratoria porque margens entre modelos nao sao calibradas.
Bootstrap pareado estratificado por classe,2000 amostras, cluster participante:
mesma pessoa reamostrada em todas suas seeds. Condicional aos modelos/folds,
sem refazer treino, pareamento ou selecao e sem corrigir tentativas historicas.
DEMOGRAFIA auxiliar: resultado baixo nao prova ausencia de todos confundidores.
Nao selecionar melhor seed, coorte ou braco externo como estimativa imparcial.

RASTREABILIDADE E PARADA
Hashes SHA256 dos dois CSVs, manifesto, protocolo, helper, seletor e executor;
versoes do ambiente, particoes, configuracoes fixas, atributos selecionados,
predicoes por pessoa, metricas por fold/seed, ICs, tempos, log e TXT.
Checkpoints por coorte/seed/fold com hashes e assinatura da execucao.
Nenhuma escrita em C:, nenhuma nova extracao ou alteracao de CSV antigo.
Nao ha tabela de candidatos: os modelos sao fixos; CONFIGURACOES_FIXAS.json.
Ao terminar, documentar resultados/limites sem conclusao final do TG.
A comparacao envolve pipelines e bases previamente exploradas, nao isola
transformadas nem estabelece generalizacao externa de um modelo AVFAD.
"""


def validate_manifest(meta):
    needed = {"source_file", "source_row", "filename", "sample_id", "participant_id",
              "session_id", "task", "y", "label", "age", "sex", "year", "matched", "pair_id"}
    if not needed.issubset(meta.columns):
        raise ValueError(f"Manifesto sem colunas: {needed-set(meta.columns)}")
    m = meta.copy().reset_index(drop=True)
    for col in ("sample_id", "participant_id", "session_id", "filename", "task", "source_file"):
        m[col] = m[col].astype(str).str.strip()
        if m[col].eq("").any() or m[col].isna().any():
            raise ValueError(f"Identificador vazio: {col}")
    for col in ("y", "source_row"):
        values = pd.to_numeric(m[col], errors="raise")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Valor nao inteiro em {col}")
        m[col] = values.astype(int)
    if set(m.y) != {0, 1} or (m.source_row < 0).any():
        raise ValueError("Rotulos ou indices fora do dominio")
    flags = m.matched.astype(str).str.strip().str.lower()
    if not flags.isin({"true", "false", "1", "0"}).all():
        raise ValueError("Flag matched invalida")
    m["matched"] = flags.isin({"true", "1"})
    m["pair_id"] = m.pair_id.fillna("").astype(str).str.strip()
    m["sex"] = m.sex.astype(str).str.lower()
    if not m.sex.isin({"m", "w"}).all():
        raise ValueError("Sexo fora do mapeamento fixo m/w")
    for col in ("age", "year"):
        m[col] = pd.to_numeric(m[col], errors="coerce")
    if m.sample_id.duplicated().any() or m.duplicated(["source_file", "source_row"]).any():
        raise ValueError("Linha de origem ou sample_id duplicado")
    if m.duplicated(["participant_id", "task"]).any():
        raise ValueError("Vogal duplicada por participante")
    for pid, group in m.groupby("participant_id"):
        if (len(group) != 3 or set(group.task) != TASKS or group.session_id.nunique() != 1
                or group.y.nunique() != 1 or group.matched.nunique() != 1
                or group.pair_id.nunique() != 1 or group.sex.nunique() != 1):
            raise ValueError(f"Pessoa sem sessao/vogais/rotulo canonicos: {pid}")
        for col in ("age", "year"):
            if group[col].nunique(dropna=False) != 1:
                raise ValueError(f"Metadado inconsistente: {pid} {col}")
    people = m.drop_duplicates("participant_id")
    if people.y.value_counts().min() < 5:
        raise ValueError("Coorte completa pequena para cinco folds")
    paired = people[people.matched]
    if paired.pair_id.eq("").any() or (~people.matched & people.pair_id.ne("")).any():
        raise ValueError("Pair_id inconsistente com matched")
    for pair_id, pair in paired.groupby("pair_id"):
        if len(pair) != 2 or set(pair.y) != {0, 1} or pair.sex.nunique() != 1:
            raise ValueError(f"Par invalido: {pair_id}")
        if (not np.isfinite(pair[["age", "year"]].to_numpy(dtype=float)).all()
                or pair.age.max()-pair.age.min() > 3 or pair.year.max()-pair.year.min() > 1):
            raise ValueError(f"Par fora de limites demograficos: {pair_id}")
    if paired.pair_id.nunique() < 5:
        raise ValueError("Menos de cinco pares")
    return m


def read_manifest(path):
    return validate_manifest(pd.read_csv(path, keep_default_na=False, dtype={
        name: str for name in ("sample_id", "participant_id", "session_id", "task",
                              "source_file", "filename", "label", "pair_id", "sex")}))


def feature_groups(names):
    patterns = {
        "WPTt": ("wptt_",), "WPTglobal": ("wpt_",),
        "DWT": ("db4_", "sym8_", "coif5_"), "MFCC": ("mfcc_",),
    }
    return {key: np.flatnonzero(np.fromiter(
        (n.startswith(prefixes) for n in names), dtype=bool, count=len(names)))
        for key, prefixes in patterns.items()}


def load_features(source_root, meta):
    paths = [source_root / name for name in SOURCES]
    headers = [pd.read_csv(path, nrows=0).columns.tolist() for path in paths]
    if headers[0] != headers[1]:
        raise ValueError("Cabecalhos dos CSVs divergem")
    groups = feature_groups(np.asarray(headers[0], dtype=str))
    if {k: len(v) for k, v in groups.items()} != {
            "WPTt": 1536, "WPTglobal": 576, "DWT": 180, "MFCC": 26}:
        raise ValueError("Dimensoes dos grupos divergem do protocolo")
    columns = [headers[0][i] for i in sorted(np.concatenate(list(groups.values())))]
    names = np.asarray(columns, dtype=str)
    groups = feature_groups(names)
    source_names = [str(Path(value).resolve()) for value in meta.source_file]
    allowed = {str(path.resolve()) for path in paths}
    if not set(source_names).issubset(allowed):
        raise ValueError("Manifesto aponta arquivo fora dos dois CSVs autorizados")
    matrix = np.empty((len(meta), len(columns)), dtype=np.float32)
    seen = np.zeros(len(meta), dtype=bool)
    for path in paths:
        selected = np.flatnonzero(np.asarray(source_names) == str(path.resolve()))
        wanted_rows = meta.loc[selected, "source_row"].to_numpy()
        offset = 0
        for chunk in pd.read_csv(path, usecols=columns+["filename", "label"],
                                 dtype={**dict.fromkeys(columns, np.float32),
                                        "filename": str, "label": str},
                                 keep_default_na=True, chunksize=2048):
            take = np.flatnonzero((wanted_rows >= offset) & (wanted_rows < offset+len(chunk)))
            if len(take):
                positions = selected[take]
                rows = chunk.iloc[wanted_rows[take]-offset]
                actual = rows.filename.astype(str).str.replace("\\", "/", regex=False).str.strip().to_numpy()
                expected = meta.loc[positions, "filename"].str.replace("\\", "/", regex=False).str.strip().to_numpy()
                if (not np.array_equal(actual, expected)
                        or not np.array_equal(rows.label.astype(str).to_numpy(),
                                              meta.loc[positions, "label"].astype(str).to_numpy())):
                    raise ValueError("source_row nao corresponde ao filename/label do manifesto")
                matrix[positions] = rows[columns].to_numpy(dtype=np.float32)
                seen[positions] = True
            offset += len(chunk)
    if not seen.all():
        raise ValueError("Indices de origem ausentes dos CSVs")
    matrix[~np.isfinite(matrix)] = np.nan
    demographic = np.column_stack((meta.age, meta.sex.map({"m": 0, "w": 1}), meta.year)).astype(np.float32)
    start = matrix.shape[1]
    matrix = np.column_stack((matrix, demographic))
    names = np.r_[names, ["DEMOGRAFIA_age", "DEMOGRAFIA_sex_m0_w1", "DEMOGRAFIA_year"]]
    groups["DEMOGRAFIA"] = np.arange(start, start+3)
    return matrix, names, groups


def prepare(source_root, manifest, out, seeds):
    protocol = out / "PROTOCOLO.txt"
    if not protocol.is_file():
        raise FileNotFoundError("Congelar PROTOCOLO.txt antes de verificar/executar; usar protocol_text().")
    if protocol.read_text(encoding="utf-8") != protocol_text(seeds):
        raise ValueError("PROTOCOLO.txt difere da configuracao efetiva; preservar e revisar")
    meta = read_manifest(manifest)
    files = [source_root/name for name in SOURCES]
    files += [manifest, protocol, Path(__file__), HERE/"wavelet_participant_selection.py",
              HERE/"svd_cohort_manifest.py"]
    files += [out/name for name in ("PROTOCOLO_COORTE.txt", "assinatura_manifesto.json")
              if (out/name).is_file()]
    before = {str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
    identity = {"version": VERSION, "files": {str(p.resolve()): sha(p) for p in files},
                "seeds": list(seeds), "outer_folds": 5, "arms": ARM_CONFIG,
                "voice_model": "SVC_linear_C1_class_weight_None", "threshold": 0,
                "demographic_model": "LogisticRegression_C1_liblinear_class_weight_None",
                "selection_k": 128, "bootstrap": 2000,
                "environment": {"python": sys.version, "executable": sys.executable,
                                "platform": platform.platform(), "numpy": np.__version__,
                                "pandas": pd.__version__, "scipy": scipy.__version__,
                                "sklearn": sklearn.__version__}}
    x, names, groups = load_features(source_root, meta)
    for path in files:
        stat = path.stat()
        if before[str(path.resolve())] != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"Arquivo alterou durante a leitura: {path}")
    return x, meta, names, groups, json.loads(json.dumps(identity, sort_keys=True))


def splits(meta, cohort, seed):
    current = meta[meta.matched] if cohort == "MATCHED" else meta
    people = current.drop_duplicates("participant_id").sort_values("participant_id")
    if cohort == "MATCHED":
        units = np.array(sorted(people.pair_id.unique()))
        splitter = KFold(5, shuffle=True, random_state=seed)
        divisions = ((set(people[people.pair_id.isin(units[tr])].participant_id),
                      set(people[people.pair_id.isin(units[te])].participant_id))
                     for tr, te in splitter.split(units))
    elif cohort == "FULL":
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        divisions = ((set(people.iloc[tr].participant_id), set(people.iloc[te].participant_id))
                     for tr, te in splitter.split(people.participant_id, people.y))
    else:
        raise ValueError("Coorte desconhecida")
    for train_ids, test_ids in divisions:
        if train_ids & test_ids:
            raise ValueError("Pessoa compartilhada entre treino e teste")
        tr = current.index[current.participant_id.isin(train_ids)].to_numpy()
        te = current.index[current.participant_id.isin(test_ids)].to_numpy()
        if cohort == "MATCHED" and set(meta.loc[tr, "pair_id"]) & set(meta.loc[te, "pair_id"]):
            raise ValueError("Par compartilhado entre treino e teste")
        yield tr, te


def fold_manifest(meta, seeds):
    rows = []
    for cohort in COHORTS:
        for seed in seeds:
            for fold, (_, te) in enumerate(splits(meta, cohort, seed), 1):
                people = meta.loc[te].drop_duplicates("participant_id")
                rows.extend({"cohort": cohort, "seed": seed, "fold": fold,
                             "participant_id": row.participant_id, "y": row.y,
                             "pair_id": row.pair_id, "n_gravacoes": 3}
                            for row in people.itertuples())
    frame = pd.DataFrame(rows)
    if frame.duplicated(["cohort", "seed", "participant_id"]).any():
        raise ValueError("Manifesto de folds duplicado")
    return frame


def row_weights(meta):
    people = meta[["participant_id", "y"]].drop_duplicates()
    counts = meta.groupby("participant_id").size()
    class_counts = people.y.value_counts()
    w = np.array([len(people)/(2*class_counts[y]*counts[pid])
                  for pid, y in zip(meta.participant_id, meta.y)], dtype=float)
    return w * (len(w)/w.sum())


def make_model(arm, n_features):
    _, method = ARM_CONFIG[arm]
    if method == "demografia":
        return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scale", StandardScaler()),
                         ("classifier", LogisticRegression(C=1, solver="liblinear", max_iter=2000,
                                                           class_weight=None, random_state=20261105))])
    selector = (ParticipantMeanANOVA(k=min(128, n_features)) if method == "pessoa"
                else SelectKBest(f_classif, k="all" if method == "todos" else min(128, n_features)))
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("remove_constants", VarianceThreshold()),
        ("select", selector), ("scale", StandardScaler()),
        ("classifier", SVC(kernel="linear", C=1, class_weight=None, cache_size=300))])


def fit_model(arm, x, meta):
    model = make_model(arm, x.shape[1])
    kwargs = {"classifier__sample_weight": row_weights(meta)}
    if ARM_CONFIG[arm][1] == "pessoa":
        kwargs["select__groups"] = meta.participant_id.to_numpy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model.fit(x, meta.y.to_numpy(), **kwargs)
    return model


def aggregate(meta, scores):
    if len(scores) != len(meta) or not np.isfinite(scores).all():
        raise ValueError("Scores por gravacao invalidos")
    frame = meta[["participant_id", "y"]].copy()
    frame["score"] = scores
    result = frame.groupby("participant_id", as_index=False).agg(
        y=("y", "first"), score=("score", "mean"), n_gravacoes=("score", "size"))
    result["predicao"] = (result.score >= 0).astype(int)
    return result.sort_values("participant_id").reset_index(drop=True)


def validate_predictions(frame, meta, cohort, seed, fold):
    if (set(frame.arm) != set(ARMS) or set(frame.cohort) != {cohort}
            or set(frame.seed) != {seed} or set(frame.fold) != {fold}):
        raise ValueError("Braco/coorte/seed/fold inesperado")
    expected = meta.drop_duplicates("participant_id").set_index("participant_id").y
    counts = meta.groupby("participant_id").size()
    for arm, sub in frame.groupby("arm"):
        if (len(sub) != len(expected) or sub.participant_id.duplicated().any()
                or set(sub.participant_id) != set(expected.index)):
            raise ValueError(f"OOF incompleto/duplicado em {arm}")
        if (not np.array_equal(sub.participant_id.map(expected), sub.y)
                or not np.array_equal(sub.participant_id.map(counts), sub.n_gravacoes)
                or not np.isfinite(sub.score).all() or not (sub.limiar == 0).all()
                or not np.array_equal((sub.score >= 0).astype(int), sub.predicao)):
            raise ValueError("Rotulos, scores, contagem ou decisao invalidos")


def metric(frame):
    tn, fp, fn, tp = confusion_matrix(frame.y, frame.predicao, labels=[0, 1]).ravel()
    return {"n_participantes": len(frame),
            "BA": float((tn/(tn+fp)+tp/(tp+fn))/2),
            "sensibilidade": float(tp/(tp+fn)), "especificidade": float(tn/(tn+fp)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "AUC": float(roc_auc_score(frame.y, frame.score))}


def evaluate_fold(x, meta, names, groups, cohort, seed, fold, tr, te, log):
    train, test = meta.loc[tr].copy(), meta.loc[te].copy()
    if set(train.participant_id) & set(test.participant_id):
        raise ValueError("Vazamento de participantes")
    predictions, selected, metrics = [], [], []
    for arm, (representation, method) in ARM_CONFIG.items():
        cols = groups[representation]
        model = fit_model(arm, x[tr][:, cols], train)
        result = aggregate(test, model.decision_function(x[te][:, cols]))
        if method == "demografia":
            selected_names = names[cols]
        else:
            keep = model.named_steps["remove_constants"].get_support()
            selected_names = names[cols][keep][model.named_steps["select"].get_support()]
        for key, value in {"cohort": cohort, "seed": seed, "fold": fold, "arm": arm,
                           "limiar": 0, "input_features": len(cols),
                           "selected_features": len(selected_names)}.items():
            result[key] = value
        predictions.append(result)
        selected.extend({"cohort": cohort, "seed": seed, "fold": fold, "arm": arm, "feature": name}
                        for name in selected_names)
        mm = metric(result)
        metrics.append({"cohort": cohort, "seed": seed, "fold": fold, "arm": arm, **mm})
        log(f"{cohort} seed={seed} fold={fold} {arm}: BA={mm['BA']:.6f}; k={len(selected_names)}")
    frame = pd.concat(predictions, ignore_index=True)
    validate_predictions(frame, test, cohort, seed, fold)
    return {"predicoes.csv": frame, "atributos.csv": pd.DataFrame(selected),
            "metricas.csv": pd.DataFrame(metrics)}


def bootstrap_pairs(pred, seeds, left_arm, right_arm, iterations=2000):
    group = pred[pred.seed.isin(seeds)]
    ids = sorted(group.participant_id.unique())
    labels = group.drop_duplicates("participant_id").set_index("participant_id").loc[ids].y.to_numpy()
    differences = []
    for seed in seeds:
        left = group[(group.seed == seed) & (group.arm == left_arm)].set_index("participant_id").loc[ids]
        right = group[(group.seed == seed) & (group.arm == right_arm)].set_index("participant_id").loc[ids]
        if not np.array_equal(left.y, labels) or not np.array_equal(right.y, labels):
            raise ValueError("Rotulos divergentes no bootstrap pareado")
        differences.append((left.predicao.to_numpy() == labels).astype(float)
                           - (right.predicao.to_numpy() == labels).astype(float))
    delta = np.mean(differences, axis=0)
    normal, pathological = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    rng = np.random.default_rng(16052026)
    sampled = (delta[rng.choice(normal, (iterations, len(normal)), replace=True)].mean(axis=1)
               + delta[rng.choice(pathological, (iterations, len(pathological)), replace=True)].mean(axis=1))/2
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {"delta_BA": float((delta[normal].mean()+delta[pathological].mean())/2),
            "IC95_low": float(low), "IC95_high": float(high)}


def consolidate(out, meta, seeds, log):
    paths = [out/"checkpoints"/f"{cohort}_seed_{seed}_fold_{fold:02d}"
             for cohort in COHORTS for seed in seeds for fold in range(1, 6)]
    pred = pd.concat([pd.read_csv(path/"predicoes.csv", dtype={"participant_id": str}) for path in paths],
                     ignore_index=True)
    attributes = pd.concat([pd.read_csv(path/"atributos.csv") for path in paths], ignore_index=True)
    for cohort in COHORTS:
        current = meta[meta.matched] if cohort == "MATCHED" else meta
        for seed in seeds:
            sub = pred[(pred.cohort == cohort) & (pred.seed == seed)]
            if set(sub.arm) != set(ARMS):
                raise ValueError("Falta braco na consolidacao")
            for arm, arm_pred in sub.groupby("arm"):
                if (len(arm_pred) != current.participant_id.nunique()
                        or arm_pred.participant_id.duplicated().any()
                        or set(arm_pred.participant_id) != set(current.participant_id)):
                    raise ValueError(f"OOF incompleto: {cohort} {seed} {arm}")
    fold_rows = [{"cohort": c, "seed": int(s), "arm": a, "fold": int(f), **metric(sub)}
                 for (c, s, a, f), sub in pred.groupby(["cohort", "seed", "arm", "fold"])]
    fold_metrics = pd.DataFrame(fold_rows)
    summary_rows = []
    for (cohort, seed, arm), sub in pred.groupby(["cohort", "seed", "arm"]):
        mm = metric(sub)
        mm["AUC_OOF_exploratoria"] = mm.pop("AUC")
        fm = fold_metrics[(fold_metrics.cohort == cohort) & (fold_metrics.seed == seed) & (fold_metrics.arm == arm)]
        mm["AUC_media_por_fold"] = float(np.average(fm.AUC, weights=(fm.TN+fm.FP)*(fm.TP+fm.FN)))
        summary_rows.append({"cohort": cohort, "seed": int(seed), "arm": arm, **mm})
    summary = pd.DataFrame(summary_rows)
    scopes = ([("seed_primaria", [20261105])] if 20261105 in seeds else [])
    secondary = [seed for seed in seeds if seed != 20261105]
    if secondary:
        scopes.append(("estabilidade_secundaria", secondary))
    comparisons = []
    for cohort in COHORTS:
        for scope, selected_seeds in scopes:
            for other, label in COMPARISONS:
                designation = ("primaria" if cohort == "MATCHED" and scope == "seed_primaria"
                               and other == "MFCC_baseline" else "secundaria")
                comparisons.append({"cohort": cohort, "escopo": scope, "comparacao": label,
                                    "designacao": designation, "seeds": str(selected_seeds),
                                    **bootstrap_pairs(pred[pred.cohort == cohort], selected_seeds, PRIMARY, other)})
    csv_write(out/"predicoes_oof_participante.csv", pred)
    csv_write(out/"atributos_selecionados.csv", attributes)
    csv_write(out/"metricas_por_fold.csv", fold_metrics)
    csv_write(out/"metricas_por_seed.csv", summary)
    csv_write(out/"comparacoes_pareadas.csv", pd.DataFrame(comparisons))
    report = ["EXPERIMENTO 16 - RESULTADOS DE DESENVOLVIMENTO SVD",
              f"Consolidado: {stamp()}", "",
              "Protocolo de seis seeds completo." if list(seeds) == list(DEFAULT_SEEDS)
              else "Configuracao de seeds parcial/diferente; nao chamar de protocolo completo.",
              "Desfecho primario: MATCHED, seed20261105, WPTt_ANOVA_pessoa menos MFCC.",
              "FULL e todas as demais comparacoes sao secundarias.",
              "Modelos fixos: nao houve busca de candidatos nem CV interna.", ""]
    for cohort in COHORTS:
        current = meta[meta.matched] if cohort == "MATCHED" else meta
        people = current.drop_duplicates("participant_id")
        report += [f"COORTE {cohort}: {len(people)} pessoas; {len(current)} vogais; "
                   f"classes={people.y.value_counts().sort_index().to_dict()}"]
        for row in summary[summary.cohort == cohort].itertuples():
            report.append(f"seed={row.seed} {row.arm}: BA={row.BA:.6f}; sens={row.sensibilidade:.6f}; "
                          f"esp={row.especificidade:.6f}; AUC_fold={row.AUC_media_por_fold:.6f}; "
                          f"TN/FP/FN/TP={row.TN}/{row.FP}/{row.FN}/{row.TP}")
        report += [""]
    report += ["DIFERENCAS PAREADAS - IC95 CONDICIONAL"]
    for row in comparisons:
        report.append(f"{row['designacao']} {row['cohort']} {row['escopo']} {row['comparacao']}: "
                      f"deltaBA={row['delta_BA']:+.6f}; [{row['IC95_low']:+.6f}, {row['IC95_high']:+.6f}]")
    if secondary:
        report += ["", "ESTABILIDADE SECUNDARIA - MESMAS PESSOAS, DP NAO E IC"]
        for (cohort, arm), group in summary[summary.seed.isin(secondary)].groupby(["cohort", "arm"]):
            report.append(f"{cohort} {arm}: BA media={group.BA.mean():.6f}; min={group.BA.min():.6f}; "
                          f"max={group.BA.max():.6f}; DP={group.BA.std():.6f}")
    report += ["", "LIMITACOES E RASTREABILIDADE",
               "SVD ja explorada: desenvolvimento, nao teste externo cego de modelo congelado.",
               "Nao houve transferencia AVFAD; familias Wavelet, tarefas e front-end diferem.",
               "Nao isola efeito de transformada entre pipelines com extracao diferente.",
               "Pareamento usa apenas sexo/idade/ano; nao elimina confundimento nao medido.",
               "DEMOGRAFIA e controle auxiliar linear; BA baixa nao prova ausencia de confundidores.",
               "Bootstrap reamostra pessoas mantendo seeds; nao refaz pareamento, treino ou selecao.",
               "Nao escolher melhor seed, coorte ou braco para reivindicar desempenho imparcial.",
               "Pesos de treino equilibram classes por pessoa e dividem por quantidade de vogais.",
               "AUC por fold preserva escala de cada modelo; concatenada e exploratoria.",
               "Manifesto, CSVs de origem, codigos e ambiente vinculados por SHA256.",
               "Nenhum CSV de C: alterado; nenhum audio reextraido.",
               "Ver PROTOCOLO.txt, PROTOCOLO_COORTE.txt, assinatura_execucao.json e particoes_participantes.csv.",
               "Este documento nao constitui conclusao final do TG.", ""]
    atomic_text(out/"RELATORIO_RESULTADOS.txt", "\n".join(report))
    log("Consolidacao concluida; resultados TXT e CSV gravados.")


def run(x, meta, names, groups, out, seeds, identity):
    signature = out/"assinatura_execucao.json"
    if signature.exists():
        if json.loads(signature.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Identidade diferente; preservar resultados e usar novo experimento")
    else:
        allowed = HELPER_FILES | {"PROTOCOLO.txt"}
        if any(path.name not in allowed for path in out.iterdir()):
            raise RuntimeError("Pasta sem assinatura contem arquivos nao previstos; preservar e revisar")
        json_write(signature, identity)
    digest = identity_digest(identity)
    log = Logger(out/"LOG_EXECUCAO.txt")
    log(f"Inicio/retomada{VERSION}; seeds={list(seeds)}; sem novas extracoes; {len(ARMS)} bracos fixos.")
    csv_write(out/"particoes_participantes.csv", fold_manifest(meta, seeds))
    json_write(out/"CONFIGURACOES_FIXAS.json", {
        "arms": ARM_CONFIG, "k_voice": 128, "mfcc": "all_after_variance",
        "threshold": 0, "weight": "participant_class_balanced_divided_by_n_rows_normalized_mean1",
        "SVC": {"kernel": "linear", "C": 1, "class_weight": None},
        "DEMOGRAFIA": {"model": "LogisticRegression", "C": 1, "solver": "liblinear", "max_iter": 2000}})
    (out/"checkpoints").mkdir(exist_ok=True)
    expected_files = {"predicoes.csv", "atributos.csv", "metricas.csv"}
    for cohort in COHORTS:
        for seed in seeds:
            for fold, (tr, te) in enumerate(splits(meta, cohort, seed), 1):
                destination = out/"checkpoints"/f"{cohort}_seed_{seed}_fold_{fold:02d}"
                marker = destination/"CONCLUIDO.json"
                if marker.is_file():
                    saved = json.loads(marker.read_text(encoding="utf-8"))
                    if saved.get("run_signature") != digest or set(saved.get("sha256", {})) != expected_files:
                        raise RuntimeError(f"Assinatura/checkpoint incompleto: {destination}")
                    for filename, previous_sha in saved["sha256"].items():
                        if sha(destination/filename) != previous_sha:
                            raise RuntimeError(f"Checkpoint alterado: {destination/filename}")
                    validate_predictions(pd.read_csv(destination/"predicoes.csv", dtype={"participant_id": str}),
                                         meta.loc[te], cohort, seed, fold)
                    log(f"{cohort} seed={seed} fold={fold}: checkpoint integro, sem novo fit.")
                    continue
                if destination.exists() and any(destination.iterdir()):
                    raise RuntimeError(f"Checkpoint parcial, preservar e revisar: {destination}")
                destination.mkdir(exist_ok=True)
                start = time.monotonic()
                frames = evaluate_fold(x, meta, names, groups, cohort, seed, fold, tr, te, log)
                for filename, frame in frames.items():
                    csv_write(destination/filename, frame)
                json_write(marker, {"run_signature": digest, "finished": stamp(),
                                    "elapsed_seconds": time.monotonic()-start,
                                    "sha256": {filename: sha(destination/filename) for filename in frames}})
                log(f"{cohort} seed={seed} fold={fold}: concluido em {time.monotonic()-start:.1f}s")
    consolidate(out, meta, seeds, log)


def synthetic():
    rng = np.random.default_rng(1605)
    rows = []
    for p in range(56):
        paired = p < 40
        y = p % 2 if paired else int(p >= 44)
        pair = f"PAR{p//2:03d}" if paired else ""
        for task in sorted(TASKS):
            rows.append({"source_file": "synthetic.csv", "source_row": len(rows),
                         "filename": f"S{p:03d}/{task}.nsp", "sample_id": f"S{p:03d}__{task}",
                         "participant_id": f"S{p:03d}", "session_id": f"R{p:03d}", "task": task,
                         "y": y, "label": "healthy" if y == 0 else "pathology",
                         "age": 25+y, "sex": "m" if (p//2) % 2 == 0 else "w",
                         "year": 2010+y, "matched": paired, "pair_id": pair})
    meta = validate_manifest(pd.DataFrame(rows))
    groups, names, position = {}, [], 0
    for group, prefix, dimension in (("WPTt", "wptt_", 160), ("WPTglobal", "wpt_", 144),
                                     ("DWT", "db4_", 140), ("MFCC", "mfcc_", 26)):
        groups[group] = np.arange(position, position+dimension)
        names.extend(f"{prefix}{i}" for i in range(dimension))
        position += dimension
    x = rng.normal(size=(len(meta), position)).astype(np.float32)
    x[:, :10] += meta.y.to_numpy()[:, None] * 0.25
    x[0, 0] = np.nan
    x[:, 1] = 0
    demographics = np.column_stack((meta.age, meta.sex.map({"m": 0, "w": 1}), meta.year))
    groups["DEMOGRAFIA"] = np.arange(position, position+3)
    x = np.column_stack((x, demographics)).astype(np.float32)
    return x, meta, np.asarray(names+["DEMOGRAFIA_age", "DEMOGRAFIA_sex_m0_w1", "DEMOGRAFIA_year"]), groups


def autotest():
    x, meta, names, groups = synthetic()
    seeds = [20261105]
    for cohort in COHORTS:
        tested = []
        for tr, te in splits(meta, cohort, seeds[0]):
            assert not set(meta.loc[tr, "participant_id"]) & set(meta.loc[te, "participant_id"])
            if cohort == "MATCHED":
                assert not set(meta.loc[tr, "pair_id"]) & set(meta.loc[te, "pair_id"])
            tested.extend(meta.loc[te, "participant_id"].unique())
        current = meta[meta.matched] if cohort == "MATCHED" else meta
        assert len(tested) == len(set(tested)) == current.participant_id.nunique()
    weights = pd.DataFrame({"participant_id": meta.participant_id, "y": meta.y, "weight": row_weights(meta)})
    summed = weights.groupby(["participant_id", "y"]).weight.sum().reset_index()
    assert np.isclose(weights.weight.mean(), 1)
    assert np.isclose(summed.groupby("y").weight.sum().iloc[0], summed.groupby("y").weight.sum().iloc[1])
    for _, group in summed.groupby("y"):
        assert np.allclose(group.weight, group.weight.iloc[0])
    tr, te = next(splits(meta, "MATCHED", seeds[0]))
    before = evaluate_fold(x, meta, names, groups, "MATCHED", seeds[0], 1, tr, te, lambda _: None)
    contaminated_x, contaminated_meta = x.copy(), meta.copy()
    contaminated_x[te] *= 100000
    contaminated_meta.loc[te, "y"] = 1-contaminated_meta.loc[te, "y"]
    after = evaluate_fold(contaminated_x, contaminated_meta, names, groups, "MATCHED",
                          seeds[0], 1, tr, te, lambda _: None)
    pd.testing.assert_frame_equal(before["atributos.csv"], after["atributos.csv"])
    for arm, (representation, _) in ARM_CONFIG.items():
        cols = groups[representation]
        model = fit_model(arm, x[tr][:, cols], meta.loc[tr])
        learned = model.named_steps["scale"].mean_.copy()
        model.decision_function(contaminated_x[te][:, cols])
        np.testing.assert_array_equal(learned, model.named_steps["scale"].mean_)
        assert model.named_steps["classifier"].class_weight is None
    with TemporaryDirectory(prefix="svd_exp16_test_") as temporary:
        out = Path(temporary)
        atomic_text(out/"PROTOCOLO.txt", protocol_text(seeds))
        identity = {"synthetic": True, "version": VERSION, "seeds": seeds}
        run(x, meta, names, groups, out, seeds, identity)
        pred = pd.read_csv(out/"predicoes_oof_participante.csv")
        first = bootstrap_pairs(pred[pred.cohort == "FULL"], seeds, PRIMARY, "MFCC_baseline")
        duplicate = pred[pred.cohort == "FULL"].copy()
        duplicate["seed"] = 101
        repeated = bootstrap_pairs(pd.concat((pred[pred.cohort == "FULL"], duplicate)),
                                   [seeds[0], 101], PRIMARY, "MFCC_baseline")
        assert first == repeated
        resumed = False
        original_fit = globals()["fit_model"]
        def forbidden_fit(*args, **kwargs):
            raise AssertionError("Retomada tentou treinar novamente")
        globals()["fit_model"] = forbidden_fit
        try:
            run(x, meta, names, groups, out, seeds, identity)
            resumed = True
        finally:
            globals()["fit_model"] = original_fit
        assert resumed
        checkpoint = next((out/"checkpoints").glob("*/predicoes.csv"))
        checkpoint.write_text(checkpoint.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        try:
            run(x, meta, names, groups, out, seeds, identity)
        except RuntimeError as exc:
            assert "Checkpoint alterado" in str(exc)
        else:
            raise AssertionError("Corrupcao de checkpoint nao detectada")
    print("AUTOTESTE OK: folds pessoa/par, pesos, isolamento de teste, escala, bootstrap cluster, "
          "roundtrip, retomada sem fit e corrupcao; sem ler SVD real nem escrever em C:.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonte", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifesto", type=Path)
    parser.add_argument("--saida", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verificar", action="store_true")
    action.add_argument("--executar", action="store_true")
    action.add_argument("--autoteste", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        autotest()
        return
    seeds = list(args.seeds)
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        parser.error("Sementes distintas e nao negativas")
    out = args.saida.resolve()
    if args.executar and out.drive.upper() == "C:":
        raise ValueError("Saida em C: proibida; preservar fontes e usar pasta D: do projeto")
    manifest = args.manifesto.resolve() if args.manifesto else out/"manifesto_coorte.csv"
    x, meta, names, groups, identity = prepare(args.fonte.resolve(), manifest, out, seeds)
    partitions = fold_manifest(meta, seeds)
    print(f"VERIFICADO: {meta.participant_id.nunique()} pessoas, {len(meta)} vogais, "
          f"matched={meta.loc[meta.matched, 'participant_id'].nunique()}, "
          f"grupos={ {k:len(v) for k,v in groups.items()} }, folds={len(partitions)} linhas.", flush=True)
    if not args.executar:
        print("Sem treinamento; nenhum arquivo gravado.", flush=True)
        return
    run(x, meta, names, groups, out, seeds, identity)


if __name__ == "__main__":
    main()
