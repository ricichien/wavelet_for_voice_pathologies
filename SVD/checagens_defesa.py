# -*- coding: utf-8 -*-
"""
Checagens para a defesa -- saudavel vs patologico (SVD).

NAO reextrai audio: usa os dois CSVs que o pipeline ja gerou
  - svd_features_mfcc_wavelet_temporal.csv   (9 patologias)
  - svd_features_healthy.csv                 (550 falantes saudaveis)

O que ele responde (cada item e uma pergunta que a banca pode fazer):

 1) "Foi so sorte desse sorteio de teste?"
    -> validacao cruzada ESTRATIFICADA POR FALANTE (5 folds): media e desvio
       da acuracia de cada representacao, e a diferenca WPT - MFCC fold a fold.

 2) "A WPT ganha so por ter 2112 atributos contra 27?"
    -> WPT com apenas os 27 melhores atributos, escolhidos DENTRO de cada fold
       (sem vazar informacao do teste).

 3) "O ganho vem do volume/tamanho da gravacao, e nao da voz?"
    -> WPT 'invariante a ganho': so entropia e ZCR (global) e energia relativa,
       ZCR e delta (temporal), que nao dependem do volume absoluto.

 4) "Qual parte da WPT carrega o ganho?"
    -> WPT global, WPT temporal e as duas juntas, separadas.

 5) "Saudaveis e patologicos diferem em sexo, idade ou duracao?"
    -> estatisticas por grupo e um classificador que usa SO sexo/idade/duracao.
       Se ele ja acerta bem, parte da tarefa nao e sobre a voz.

COMO RODAR (na raiz do projeto, com o venv ativo):
    python checagens_defesa.py

Pode demorar horas (varios treinos x 5 folds). Os resultados sao salvos a
cada treino em checagens_defesa_folds.csv: se o PC dormir, nada se perde.
"""

import os
import io
import sys
import csv
import time
import zipfile
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURACAO
# ---------------------------------------------------------------------------
PATHO_CSV = "svd_features_mfcc_wavelet_temporal.csv"
HEALTHY_CSV = "svd_features_healthy.csv"

N_SPLITS = 5
RANDOM_STATE = 42

# modelos usados na validacao cruzada. CatBoost foi o melhor em alguns casos,
# mas e bem mais lento: ligue so se tiver tempo (USAR_CATBOOST = True).
MODELOS = ["logistic_regression", "svm_rbf", "random_forest"]
USAR_CATBOOST = False

K_MELHORES = 27          # mesmo numero de atributos do MFCC (controle de dimensionalidade)
OUT_FOLDS = "checagens_defesa_folds.csv"
OUT_RELATORIO = "checagens_defesa_relatorio.txt"

# zips (so para ler o overview.csv: sexo, nascimento, data da gravacao).
# Se o import falhar, a parte demografica e pulada.
SRC_DIR = "src"
SCRIPT_MODULES = ["svd_pipeline_wpt_temporal_HEALTHY", "svd_pipeline_wpt_temporal"]


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
RELATORIO = []


def say(msg: str = ""):
    print(msg, flush=True)
    RELATORIO.append(msg)


def carregar_dados() -> pd.DataFrame:
    for p in (PATHO_CSV, HEALTHY_CSV):
        if not os.path.exists(p):
            sys.exit(f"ERRO: {p} nao encontrado. Rode este script na pasta onde estao os CSVs.")

    patho = pd.read_csv(PATHO_CSV)
    healthy = pd.read_csv(HEALTHY_CSV)
    patho["grupo"] = "patologico"
    healthy["grupo"] = "saudavel"
    df = pd.concat([patho, healthy], ignore_index=True)
    df["y"] = (df["grupo"] == "patologico").astype(int)
    say(f"Dados: {len(df)} gravacoes | {df['speaker_id'].nunique()} falantes")
    say(df.groupby("grupo")["speaker_id"].agg(["size", "nunique"])
        .rename(columns={"size": "gravacoes", "nunique": "falantes"}).to_string())
    return df


def conjuntos_de_colunas(df: pd.DataFrame) -> dict:
    cols = [c for c in df.columns if c not in ("label", "speaker_id", "filename", "grupo", "y")]
    mfcc = [c for c in cols if c.startswith("mfcc_") or c == "duration_sec"]
    wpt_g = [c for c in cols if c.startswith("wpt_")]
    wpt_t = [c for c in cols if c.startswith("wptt_")]
    # atributos que nao dependem do volume absoluto da gravacao:
    #  global: entropia (energia normalizada) e ZCR; temporal: energia relativa,
    #  ZCR e delta da log-energia (a diferenca entre frames cancela o ganho)
    wpt_inv = ([c for c in wpt_g if c.endswith("_entropy") or c.endswith("_zcr")] +
               [c for c in wpt_t if "_relenergy_" in c or "_zcr_" in c or "_delta_" in c])
    return {
        "MFCC": (mfcc, None),
        "WPT global": (wpt_g, None),
        "WPT temporal": (wpt_t, None),
        "WPT completa (global+temporal)": (wpt_g + wpt_t, None),
        "WPT invariante a ganho": (wpt_inv, None),
        f"WPT completa -> {K_MELHORES} melhores (dentro do fold)": (wpt_g + wpt_t, K_MELHORES),
        "Hibrido MFCC + WPT completa": (mfcc + wpt_g + wpt_t, None),
    }


def fabrica_modelo(nome: str):
    if nome == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if nome == "svm_rbf":
        return SVC(kernel="rbf", class_weight="balanced")
    if nome == "random_forest":
        return RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight="balanced",
                                      random_state=RANDOM_STATE)
    if nome == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(verbose=0, thread_count=4, allow_writing_files=False,
                                  random_seed=RANDOM_STATE, auto_class_weights="Balanced")
    raise ValueError(nome)


def montar_pipeline(nome_modelo: str, k_melhores):
    passos = [("sc", StandardScaler())]
    if k_melhores:
        passos.append(("sel", SelectKBest(f_classif, k=k_melhores)))
    passos.append(("clf", fabrica_modelo(nome_modelo)))
    return Pipeline(passos)


# ---------------------------------------------------------------------------
# 1-4) validacao cruzada por falante
# ---------------------------------------------------------------------------
def validacao_cruzada(df: pd.DataFrame):
    sets = conjuntos_de_colunas(df)
    modelos = list(MODELOS) + (["catboost"] if USAR_CATBOOST else [])

    y = df["y"].values
    groups = df["speaker_id"].values
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(np.zeros(len(df)), y, groups))
    say(f"\nValidacao cruzada: {N_SPLITS} folds, estratificada e agrupada por falante")
    for i, (tr, te) in enumerate(splits):
        say(f"  fold {i + 1}: treino={len(tr)} teste={len(te)} "
            f"falantes_teste={len(set(groups[te]))} %patologico_teste={y[te].mean():.2f}")

    if os.path.exists(OUT_FOLDS):
        feitos = pd.read_csv(OUT_FOLDS)
        say(f"\n[retomando] {len(feitos)} resultados ja salvos em {OUT_FOLDS}")
    else:
        feitos = pd.DataFrame(columns=["representacao", "modelo", "fold", "acc", "bal_acc", "n_atributos"])
        feitos.to_csv(OUT_FOLDS, index=False)

    ja = set(zip(feitos["representacao"], feitos["modelo"], feitos["fold"]))

    for rep, (cols, k) in sets.items():
        if not cols:
            say(f"\n[AVISO] '{rep}': nenhuma coluna encontrada, pulando.")
            continue
        X = df[cols].values.astype(np.float32)
        n_attr = k if k else len(cols)
        for mod in modelos:
            for f, (tr, te) in enumerate(splits):
                if (rep, mod, f) in ja:
                    continue
                t0 = time.time()
                pipe = montar_pipeline(mod, k)
                pipe.fit(X[tr], y[tr])
                pred = pipe.predict(X[te])
                row = {"representacao": rep, "modelo": mod, "fold": f,
                       "acc": accuracy_score(y[te], pred),
                       "bal_acc": balanced_accuracy_score(y[te], pred),
                       "n_atributos": n_attr}
                pd.DataFrame([row]).to_csv(OUT_FOLDS, mode="a", header=False, index=False)
                print(f"  {rep[:42]:<42} {mod:<20} fold {f + 1}/{N_SPLITS} "
                      f"acc={row['acc']:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    return pd.read_csv(OUT_FOLDS)


def resumo_cv(res: pd.DataFrame):
    say("\n" + "=" * 70)
    say("RESULTADO DA VALIDACAO CRUZADA POR FALANTE (acuracia, media +- desvio entre folds)")
    say("=" * 70)

    g = res.groupby(["representacao", "modelo"])["acc"].agg(["mean", "std", "count"]).reset_index()
    ordem = list(dict.fromkeys(res["representacao"]))
    for mod in dict.fromkeys(res["modelo"]):
        say(f"\n-- {mod} --")
        sub = g[g["modelo"] == mod].set_index("representacao").reindex(ordem).dropna(how="all")
        for rep, r in sub.iterrows():
            say(f"  {rep:<58} {100 * r['mean']:.2f}% +- {100 * r['std']:.2f}   (folds={int(r['count'])})")

    say("\n" + "-" * 70)
    say("DIFERENCA PAREADA POR FOLD (representacao - MFCC), em pontos percentuais")
    say("-" * 70)
    piv = res.pivot_table(index=["modelo", "fold"], columns="representacao", values="acc")
    if "MFCC" in piv.columns:
        for rep in [c for c in piv.columns if c != "MFCC"]:
            say(f"\n{rep}")
            for mod in dict.fromkeys(res["modelo"]):
                if mod not in piv.index.get_level_values(0):
                    continue
                d = (piv.loc[mod][rep] - piv.loc[mod]["MFCC"]).dropna() * 100
                if len(d) == 0:
                    continue
                say(f"  {mod:<20} media {d.mean():+.2f} pp | desvio {d.std():.2f} | "
                    f"melhor que MFCC em {int((d > 0).sum())}/{len(d)} folds")


# ---------------------------------------------------------------------------
# 5) confundidores: sexo, idade, duracao
# ---------------------------------------------------------------------------
def carregar_zips_do_pipeline():
    sys.path.insert(0, SRC_DIR)
    for nome in SCRIPT_MODULES:
        try:
            mod = __import__(nome)
            pats = list(getattr(mod, "PATHOLOGY_ZIPS", {}).keys())
            hz = getattr(mod, "HEALTHY_ZIP", None)
            return pats, hz
        except Exception as e:  # noqa: BLE001
            print(f"  (nao consegui importar {nome}: {type(e).__name__})")
    return [], None


def ler_overviews(zip_paths: list) -> pd.DataFrame:
    rows = []
    for zp in zip_paths:
        if not zp or not os.path.exists(zp):
            continue
        with zipfile.ZipFile(zp) as zf:
            cand = [n for n in zf.namelist() if n.endswith("overview.csv")]
            if not cand:
                continue
            with zf.open(cand[0]) as f:
                for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                    rows.append({"AufnahmeID": str(r.get("AufnahmeID", "")).strip(),
                                 "Geschlecht": str(r.get("Geschlecht", "")).strip(),
                                 "Geburtsdatum": r.get("Geburtsdatum", ""),
                                 "AufnahmeDatum": r.get("AufnahmeDatum", "")})
    ov = pd.DataFrame(rows)
    return ov.drop_duplicates("AufnahmeID") if len(ov) else ov


def anexar_demografia(df: pd.DataFrame, ov: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AufnahmeID"] = df["filename"].map(lambda n: os.path.basename(str(n)).split("-")[0])
    df = df.merge(ov, on="AufnahmeID", how="left")
    nasc = pd.to_datetime(df["Geburtsdatum"], errors="coerce")
    grav = pd.to_datetime(df["AufnahmeDatum"], errors="coerce")
    df["idade"] = (grav - nasc).dt.days / 365.25
    df["feminino"] = (df["Geschlecht"] == "w").astype(float)
    df.loc[~df["Geschlecht"].isin(["m", "w"]), "feminino"] = np.nan
    df["ano_gravacao"] = grav.dt.year
    return df


def checagem_confundidores(df: pd.DataFrame):
    say("\n" + "=" * 70)
    say("CONFUNDIDORES: sexo, idade, ano da gravacao e duracao")
    say("=" * 70)

    patos, hz = carregar_zips_do_pipeline()
    ov = ler_overviews(patos + [hz])
    if ov.empty:
        say("  (overview.csv nao encontrado; pulando sexo/idade. Duracao abaixo.)")
        d = df
    else:
        d = anexar_demografia(df, ov)
        sem = d["idade"].isna().mean()
        say(f"  gravacoes sem idade/sexo apos o cruzamento: {100 * sem:.1f}%")

    say("\n-- por grupo (nivel de gravacao) --")
    agg = {"duration_sec": ["mean", "median"]}
    if "idade" in d:
        agg.update({"idade": ["mean", "median"], "feminino": ["mean"], "ano_gravacao": ["mean"]})
    say(d.groupby("grupo").agg(agg).round(2).to_string())

    if "idade" in d:
        say("\n-- por grupo (nivel de falante, 1 linha por pessoa) --")
        sp = d.groupby(["grupo", "speaker_id"]).agg(idade=("idade", "mean"),
                                                    feminino=("feminino", "max")).reset_index()
        say(sp.groupby("grupo").agg(falantes=("speaker_id", "nunique"), idade_media=("idade", "mean"),
                                    pct_feminino=("feminino", "mean")).round(2).to_string())

    say("\n-- classificador que usa SO esses atributos (CV por falante) --")
    say("   (se acertar bem acima de 50%, parte da tarefa nao e sobre a voz)")
    y = d["y"].values
    groups = d["speaker_id"].values
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(sgkf.split(np.zeros(len(d)), y, groups))
    testes = {"so duracao": ["duration_sec"]}
    if "idade" in d:
        testes["so sexo + idade"] = ["feminino", "idade"]
        testes["sexo + idade + duracao"] = ["feminino", "idade", "duration_sec"]
        testes["so ano da gravacao"] = ["ano_gravacao"]
    for nome, cols in testes.items():
        X = d[cols].copy()
        X = X.fillna(X.median())
        accs = []
        for tr, te in splits:
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=4,
                                         class_weight="balanced", random_state=RANDOM_STATE)
            clf.fit(X.iloc[tr], y[tr])
            accs.append(accuracy_score(y[te], clf.predict(X.iloc[te])))
        say(f"  {nome:<26} {100 * np.mean(accs):.2f}% +- {100 * np.std(accs, ddof=1):.2f}")


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    df = carregar_dados()
    checagem_confundidores(df)
    res = validacao_cruzada(df)
    resumo_cv(res)
    say(f"\nTempo total: {(time.time() - t0) / 60:.1f} min")
    with open(OUT_RELATORIO, "w", encoding="utf-8") as f:
        f.write("\n".join(RELATORIO))
    print(f"\nRelatorio salvo em {OUT_RELATORIO} e folds em {OUT_FOLDS}")


if __name__ == "__main__":
    main()
