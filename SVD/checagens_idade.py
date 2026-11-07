# -*- coding: utf-8 -*-
"""
Checagem de IDADE e comparacao com grupos PAREADOS por idade e sexo.

Contexto: na rodada anterior (checagens_defesa.py) descobrimos que, em
saudavel vs patologico, saudaveis tem ~28 anos e patologicos ~53, e que um
classificador que usa SO sexo+idade acerta ~79% (mais que qualquer modelo de
voz). Este script:

  TAREFA 1  saudavel vs patologico
  TAREFA 2  neurologico vs nao-neurologico (so patologias)
            -> o grupo neurologico tem vox_senilis e rekurrensparese, entao
               o mesmo vies de idade pode existir aqui tambem.

Para cada tarefa:
  a) mostra idade/sexo/ano por grupo e o que um classificador acerta usando
     SO idade, sexo, ano da gravacao ou duracao (baseline demografico);
  b) PAREIA os falantes: em cada faixa de idade (e sexo), sorteia o mesmo
     numero de falantes de cada grupo. Depois do pareamento o baseline de
     idade deve cair para ~50%. Falantes que aparecem nos dois grupos sao
     excluidos;
  c) refaz a validacao cruzada por falante (MFCC, WPT, WPT invariante a ganho,
     WPT reduzida a 27 componentes por PCA, hibrido) NO CONJUNTO PAREADO.
  d) na tarefa 2, roda tambem no conjunto original, para comparar.

Usa os mesmos CSVs e os helpers de checagens_defesa.py (deixe os dois
arquivos na mesma pasta). Nao reextrai audio.

COMO RODAR (raiz do projeto, venv ativo):
    python checagens_idade.py
Os resultados sao salvos a cada treino em checagens_idade_folds.csv; se for
interrompido, ao rodar de novo ele continua de onde parou.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checagens_defesa as base  # noqa: E402  (helpers: overview, modelos, colunas)

# ---------------------------------------------------------------------------
# CONFIGURACAO
# ---------------------------------------------------------------------------
PATHO_CSV = base.PATHO_CSV
HEALTHY_CSV = base.HEALTHY_CSV

N_SPLITS = 5
SEED = 42
AGE_BIN_YEARS = 5            # largura da faixa de idade usada no pareamento
PAREAR_ANO_ANOS = None       # ex.: 3 -> tambem pareia por periodo de gravacao (perde mais dados)
MODELOS = ["logistic_regression", "svm_rbf", "random_forest"]
K_PCA = 27                   # mesmo numero de atributos do MFCC
MIN_FALANTES_POR_GRUPO = 40  # abaixo disso a validacao cruzada nao e confiavel

OUT_FOLDS = "checagens_idade_folds.csv"
OUT_RELATORIO = "checagens_idade_relatorio.txt"

LABEL_TO_GROUP = {
    "dysarthrophonie": "neurologico",
    "rekurrensparese": "neurologico",
    "spasmodische_dysphonie": "neurologico",
    "vox_senilis": "neurologico",
    "zentral_laryngale": "neurologico",
    "psychogene_dysphonie": "nao_neurologico",
    "laryngitis": "nao_neurologico",
    "funktionelle_dysphonie": "nao_neurologico",
    "balbuties": "nao_neurologico",
}

RELATORIO = []
RESUMO_BASELINES = []   # (tarefa, versao, nome, media)


def say(msg: str = ""):
    print(msg, flush=True)
    RELATORIO.append(msg)


# ---------------------------------------------------------------------------
# dados
# ---------------------------------------------------------------------------
def carregar() -> pd.DataFrame:
    for p in (PATHO_CSV, HEALTHY_CSV):
        if not os.path.exists(p):
            sys.exit(f"ERRO: {p} nao encontrado. Rode na pasta onde estao os CSVs.")
    patho = pd.read_csv(PATHO_CSV)
    healthy = pd.read_csv(HEALTHY_CSV)
    patho["grupo"] = "patologico"
    healthy["grupo"] = "saudavel"
    df = pd.concat([patho, healthy], ignore_index=True)

    patos, hz = base.carregar_zips_do_pipeline()
    ov = base.ler_overviews(patos + [hz])
    if ov.empty:
        sys.exit("ERRO: nao consegui ler os overview.csv dos zips (preciso de sexo e idade). "
                 "Confira os caminhos em PATHOLOGY_ZIPS / HEALTHY_ZIP no script de src.")
    df = base.anexar_demografia(df, ov)
    df["y"] = 0
    say(f"Dados: {len(df)} gravacoes | {df['speaker_id'].nunique()} falantes | "
        f"sem idade/sexo: {100 * df['idade'].isna().mean():.1f}%")
    return df


def tabela_falantes(d: pd.DataFrame) -> pd.DataFrame:
    return d.groupby("speaker_id").agg(
        grupo=("grupo", "first"), n_grupos=("grupo", "nunique"),
        idade=("idade", "mean"), feminino=("feminino", "max"),
        ano=("ano_gravacao", "mean"), n_grav=("grupo", "size")).reset_index()


def demografia(sp: pd.DataFrame, titulo: str):
    say(f"\n-- demografia por grupo ({titulo}; 1 linha por falante) --")
    t = sp.groupby("grupo").agg(falantes=("speaker_id", "nunique"), idade_media=("idade", "mean"),
                                idade_mediana=("idade", "median"), pct_feminino=("feminino", "mean"),
                                ano_medio=("ano", "mean")).round(2)
    say(t.to_string())


# ---------------------------------------------------------------------------
# pareamento
# ---------------------------------------------------------------------------
def parear(sp: pd.DataFrame, g_pos: str, g_neg: str):
    s = sp[(sp["n_grupos"] == 1) & sp["idade"].between(0, 105) & sp["feminino"].notna()].copy()
    s["faixa_idade"] = (s["idade"] // AGE_BIN_YEARS).astype(int)
    chaves = ["faixa_idade", "feminino"]
    if PAREAR_ANO_ANOS:
        s = s[s["ano"].notna()].copy()
        s["faixa_ano"] = (s["ano"] // PAREAR_ANO_ANOS).astype(int)
        chaves.append("faixa_ano")

    escolhidos = []
    for _, sub in s.groupby(chaves):
        a = sub[sub["grupo"] == g_pos]
        b = sub[sub["grupo"] == g_neg]
        n = min(len(a), len(b))
        if n:
            escolhidos += list(a.sample(n, random_state=SEED)["speaker_id"])
            escolhidos += list(b.sample(n, random_state=SEED)["speaker_id"])

    say(f"\n-- pareamento: faixas de {AGE_BIN_YEARS} anos x sexo"
        f"{' x periodo' if PAREAR_ANO_ANOS else ''} --")
    kept = s[s["speaker_id"].isin(escolhidos)]
    tab = pd.DataFrame({
        f"{g_pos} (antes)": s[s["grupo"] == g_pos].groupby("faixa_idade").size(),
        f"{g_neg} (antes)": s[s["grupo"] == g_neg].groupby("faixa_idade").size(),
        "mantidos por grupo": kept[kept["grupo"] == g_pos].groupby("faixa_idade").size(),
    }).fillna(0).astype(int)
    tab.index = [f"{int(i) * AGE_BIN_YEARS}-{int(i) * AGE_BIN_YEARS + AGE_BIN_YEARS - 1}" for i in tab.index]
    say(tab.to_string())
    return set(escolhidos)


# ---------------------------------------------------------------------------
# baselines demograficos
# ---------------------------------------------------------------------------
def splits_por_falante(d: pd.DataFrame):
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(sgkf.split(np.zeros(len(d)), d["y"].values, d["speaker_id"].values))


def baselines(d: pd.DataFrame, tarefa: str, versao: str):
    say(f"\n-- baseline demografico ({versao}): classificador com SO estes atributos "
        f"(CV por falante) --")
    say("   perto de 50% = o atributo nao separa os grupos; bem acima = vies")
    if versao == "PAREADO":
        say("   (no conjunto pareado, valores ATE ~10 pontos abaixo de 50% sao esperados e "
            "significam 'sem sinal':")
        say("    ao tirar falantes de uma faixa para o teste, a maioria restante dela no treino "
            "fica invertida)")
    y = d["y"].values
    splits = splits_por_falante(d)
    testes = {
        "so idade": ["idade"],
        "so sexo + idade": ["feminino", "idade"],
        "so ano da gravacao": ["ano_gravacao"],
        "so duracao": ["duration_sec"],
        "sexo + idade + ano + duracao": ["feminino", "idade", "ano_gravacao", "duration_sec"],
    }
    for nome, cols in testes.items():
        X = d[cols].copy()
        X = X.fillna(X.median())
        accs = []
        for tr, te in splits:
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=4,
                                         class_weight="balanced", random_state=SEED)
            clf.fit(X.iloc[tr], y[tr])
            accs.append(accuracy_score(y[te], clf.predict(X.iloc[te])))
        m = 100 * np.mean(accs)
        RESUMO_BASELINES.append((tarefa, versao, nome, m))
        say(f"  {nome:<30} {m:.2f}% +- {100 * np.std(accs, ddof=1):.2f}")


# ---------------------------------------------------------------------------
# validacao cruzada
# ---------------------------------------------------------------------------
def montar(nome_modelo: str, reducao):
    passos = [("sc", StandardScaler())]
    if reducao == "pca":
        passos += [("pca", PCA(n_components=K_PCA, random_state=SEED)), ("sc2", StandardScaler())]
    passos.append(("clf", base.fabrica_modelo(nome_modelo)))
    return Pipeline(passos)


def validacao_cruzada(d: pd.DataFrame, tag: str):
    cs = base.conjuntos_de_colunas(d)
    sets = {
        "MFCC": (cs["MFCC"][0], None),
        "WPT completa": (cs["WPT completa (global+temporal)"][0], None),
        "WPT invariante a ganho": (cs["WPT invariante a ganho"][0], None),
        f"WPT completa -> PCA {K_PCA}": (cs["WPT completa (global+temporal)"][0], "pca"),
        "Hibrido MFCC + WPT": (cs["Hibrido MFCC + WPT completa"][0], None),
    }
    y = d["y"].values
    splits = splits_por_falante(d)

    if os.path.exists(OUT_FOLDS):
        feitos = pd.read_csv(OUT_FOLDS)
    else:
        feitos = pd.DataFrame(columns=["tag", "representacao", "modelo", "fold", "acc", "bal_acc", "n_atributos"])
        feitos.to_csv(OUT_FOLDS, index=False)
    ja = set(zip(feitos["tag"], feitos["representacao"], feitos["modelo"], feitos["fold"]))

    say(f"\nValidacao cruzada [{tag}]: {len(d)} gravacoes, {d['speaker_id'].nunique()} falantes, "
        f"{N_SPLITS} folds por falante")
    for rep, (cols, red) in sets.items():
        if not cols:
            say(f"  [AVISO] '{rep}': nenhuma coluna encontrada, pulando.")
            continue
        X = d[cols].values.astype(np.float32)
        for mod in MODELOS:
            for f, (tr, te) in enumerate(splits):
                if (tag, rep, mod, f) in ja:
                    continue
                t0 = time.time()
                pipe = montar(mod, red)
                pipe.fit(X[tr], y[tr])
                pred = pipe.predict(X[te])
                row = {"tag": tag, "representacao": rep, "modelo": mod, "fold": f,
                       "acc": accuracy_score(y[te], pred), "bal_acc": balanced_accuracy_score(y[te], pred),
                       "n_atributos": K_PCA if red else len(cols)}
                pd.DataFrame([row]).to_csv(OUT_FOLDS, mode="a", header=False, index=False)
                print(f"  [{tag}] {rep[:26]:<26} {mod:<20} fold {f + 1}/{N_SPLITS} "
                      f"acc={row['acc']:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    res = pd.read_csv(OUT_FOLDS)
    res = res[res["tag"] == tag]
    resumo_cv(res, tag)


def resumo_cv(res: pd.DataFrame, tag: str):
    say("\n" + "=" * 70)
    say(f"RESULTADO [{tag}] acuracia por falante, media +- desvio entre folds")
    say("=" * 70)
    ordem = list(dict.fromkeys(res["representacao"]))
    g = res.groupby(["representacao", "modelo"])["acc"].agg(["mean", "std"]).reset_index()
    for mod in dict.fromkeys(res["modelo"]):
        say(f"\n-- {mod} --")
        sub = g[g["modelo"] == mod].set_index("representacao").reindex(ordem).dropna(how="all")
        for rep, r in sub.iterrows():
            say(f"  {rep:<34} {100 * r['mean']:.2f}% +- {100 * r['std']:.2f}")

    say("\nDiferenca pareada por fold (representacao - MFCC), em pontos percentuais")
    piv = res.pivot_table(index=["modelo", "fold"], columns="representacao", values="acc")
    if "MFCC" in piv.columns:
        for rep in [c for c in piv.columns if c != "MFCC"]:
            say(f"\n{rep}")
            for mod in dict.fromkeys(res["modelo"]):
                if mod not in piv.index.get_level_values(0):
                    continue
                dif = (piv.loc[mod][rep] - piv.loc[mod]["MFCC"]).dropna() * 100
                if len(dif):
                    say(f"  {mod:<20} media {dif.mean():+.2f} pp | desvio {dif.std():.2f} | "
                        f"melhor que MFCC em {int((dif > 0).sum())}/{len(dif)} folds")


# ---------------------------------------------------------------------------
# uma tarefa
# ---------------------------------------------------------------------------
def rodar_tarefa(nome: str, d: pd.DataFrame, g_pos: str, g_neg: str, tambem_original: bool):
    say("\n" + "#" * 70)
    say(f"# TAREFA: {nome}   (classe positiva = {g_pos})")
    say("#" * 70)
    d = d.copy()
    d["y"] = (d["grupo"] == g_pos).astype(int)

    sp = tabela_falantes(d)
    mistos = int((sp["n_grupos"] > 1).sum())
    say(f"Falantes com gravacoes nos dois grupos (serao excluidos do pareamento): {mistos}")
    demografia(sp, "original")
    baselines(d, nome, "ORIGINAL")

    escolhidos = parear(sp, g_pos, g_neg)
    dm = d[d["speaker_id"].isin(escolhidos)].copy()
    spm = tabela_falantes(dm)
    demografia(spm, "PAREADO")
    n_min = int(spm.groupby("grupo")["speaker_id"].nunique().min()) if len(spm) else 0
    say(f"\nConjunto pareado: {len(dm)} gravacoes | {dm['speaker_id'].nunique()} falantes "
        f"({n_min} por grupo)")
    if n_min < MIN_FALANTES_POR_GRUPO:
        say(f"[AVISO] menos de {MIN_FALANTES_POR_GRUPO} falantes por grupo apos o pareamento: "
            f"a validacao cruzada nao seria confiavel. Tente AGE_BIN_YEARS maior (ex.: 10).")
        return
    baselines(dm, nome, "PAREADO")
    cfg = f"k{N_SPLITS}"
    cfg_par = f"{cfg}_idade{AGE_BIN_YEARS}" + (f"_ano{PAREAR_ANO_ANOS}" if PAREAR_ANO_ANOS else "")
    validacao_cruzada(dm, f"{nome}|pareado|{cfg_par}")
    if tambem_original:
        validacao_cruzada(d, f"{nome}|original|{cfg}")


def main():
    t0 = time.time()
    df = carregar()

    rodar_tarefa("saudavel_vs_patologico", df, "patologico", "saudavel", tambem_original=False)

    dn = df[df["grupo"] == "patologico"].copy()
    dn["grupo"] = dn["label"].map(LABEL_TO_GROUP)
    dn = dn.dropna(subset=["grupo"])
    rodar_tarefa("neuro_vs_nao_neuro", dn, "neurologico", "nao_neurologico", tambem_original=True)

    say("\n" + "=" * 70)
    say("RESUMO: baseline demografico (CV por falante), original vs pareado")
    say("=" * 70)
    tab = pd.DataFrame(RESUMO_BASELINES, columns=["tarefa", "versao", "atributo", "acc"])
    say(tab.pivot_table(index=["tarefa", "atributo"], columns="versao", values="acc").round(1).to_string())

    say(f"\nTempo total: {(time.time() - t0) / 60:.1f} min")
    with open(OUT_RELATORIO, "w", encoding="utf-8") as f:
        f.write("\n".join(RELATORIO))
    print(f"\nRelatorio salvo em {OUT_RELATORIO}")


if __name__ == "__main__":
    main()
