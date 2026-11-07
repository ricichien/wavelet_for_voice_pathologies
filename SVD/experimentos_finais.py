# -*- coding: utf-8 -*-
"""
Experimentos finais -- saudavel vs patologico com controle de idade, sexo e
PERIODO de gravacao, em varios recortes.

Roda em sequencia (nao reextrai audio; usa os dois CSVs ja gerados):

  1) todas_patologias   : as 9 patologias juntas contra saudaveis
  2) vogal_a_normal     : so a vogal /a/ em tom normal (-a_n), em vez de misturar
                          vogais, tons e frases
  2b) vogal_a_4_tons, vogais_sustentadas: composicao de gravacoes igual nos dois grupos
  3) rekurrensparese, laryngitis, funktionelle_dysphonie, psychogene_dysphonie:
                          UMA patologia por vez contra saudaveis

Em todos:
  - falantes pareados por faixa de idade, sexo e periodo da gravacao (se nao
    sobrar gente suficiente, o script afrouxa o pareamento em etapas e AVISA
    qual usou);
  - baselines demograficos (idade, ano, duracao) para provar que o viés caiu;
  - validacao cruzada por falante (MFCC, WPT, WPT invariante a ganho, WPT com
    PCA de 27 componentes, hibrido);
  - alem da acuracia por gravacao, a acuracia por FALANTE (media das previsoes
    de todas as gravacoes da pessoa), que e como uma ferramenta de verdade
    decidiria, com intervalo de confianca de 95%.

No fim imprime um QUADRO FINAL com tudo lado a lado.

Arquivos necessarios na mesma pasta: checagens_defesa.py, checagens_idade.py,
os dois CSVs de atributos e a pasta src (para achar os zips).

COMO RODAR:  python experimentos_finais.py   (ou o rodar_experimentos_finais.bat)
Os resultados sao salvos a cada treino em experimentos_finais_folds.csv: se o
PC dormir ou for interrompido, ao rodar de novo ele continua de onde parou.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checagens_idade as ci  # noqa: E402
base = ci.base

# ---------------------------------------------------------------------------
# CONFIGURACAO
# ---------------------------------------------------------------------------
N_SPLITS = 5
SEED = 42
MODELOS = ["logistic_regression", "svm_rbf", "random_forest"]
MIN_FALANTES = 40   # minimo de falantes por grupo depois do pareamento

# (largura da faixa de idade, largura do periodo de gravacao em anos).
# Tenta da mais rigorosa para a mais frouxa; usa a primeira que deixar
# MIN_FALANTES por grupo. Na rodada anterior, faixas de 3 anos NAO bastaram:
# o ano da gravacao sozinho ainda acertava 64-73%. Por isso agora comeca em 1 ano.
# Se nenhuma configuracao deixar gente suficiente, o experimento e PULADO (em
# vez de seguir sem controle de periodo).
CONFIGS_PAREAMENTO = [(5, 1), (5, 2), (5, 3)]

OUT_FOLDS = "experimentos_finais_folds.csv"
OUT_RELATORIO = "experimentos_finais_relatorio.txt"

EXPERIMENTOS = [
    {"nome": "todas_patologias"},
    {"nome": "vogal_a_normal", "filtro_arquivo": "-a_n.nsp"},
    # composicao de gravacoes igual nos dois grupos: so vogais sustentadas
    # (a, i, u nos tons n/l/h/lhl), sem frases e sem a sequencia "iau"
    {"nome": "vogal_a_4_tons", "filtro_regex": r"-a_(n|l|h|lhl)\.nsp"},
    {"nome": "vogais_sustentadas", "filtro_regex": r"-[aiu]_(n|l|h|lhl)\.nsp"},
    {"nome": "rekurrensparese", "patologia": "rekurrensparese"},
    {"nome": "laryngitis", "patologia": "laryngitis"},
    {"nome": "funktionelle_dysphonie", "patologia": "funktionelle_dysphonie"},
    {"nome": "psychogene_dysphonie", "patologia": "psychogene_dysphonie"},
]

RELATORIO = []
QUADRO = []


def say(msg: str = ""):
    print(msg, flush=True)
    RELATORIO.append(msg)


# ---------------------------------------------------------------------------
# pareamento (idade x sexo x periodo), com afrouxamento automatico
# ---------------------------------------------------------------------------
def parear_silencioso(sp: pd.DataFrame, g_pos: str, g_neg: str, bin_idade: int, bin_ano):
    s = sp[(sp["n_grupos"] == 1) & sp["idade"].between(0, 105) & sp["feminino"].notna()].copy()
    s["fi"] = (s["idade"] // bin_idade).astype(int)
    chaves = ["fi", "feminino"]
    if bin_ano:
        s = s[s["ano"].notna()].copy()
        s["fa"] = (s["ano"] // bin_ano).astype(int)
        chaves.append("fa")
    escolhidos = []
    for _, sub in s.groupby(chaves):
        a = sub[sub["grupo"] == g_pos]
        b = sub[sub["grupo"] == g_neg]
        n = min(len(a), len(b))
        if n:
            escolhidos += list(a.sample(n, random_state=SEED)["speaker_id"])
            escolhidos += list(b.sample(n, random_state=SEED)["speaker_id"])
    return set(escolhidos)


def escolher_pareamento(sp: pd.DataFrame, g_pos: str, g_neg: str):
    ultimo_n = 0
    for bin_idade, bin_ano in CONFIGS_PAREAMENTO:
        esc = parear_silencioso(sp, g_pos, g_neg, bin_idade, bin_ano)
        n = int(sp[sp["speaker_id"].isin(esc)].groupby("grupo").size().min()) if esc else 0
        ultimo_n = n
        desc = f"idade {bin_idade} anos x sexo" + (f" x periodo {bin_ano} anos" if bin_ano else " (SEM controle de periodo)")
        say(f"   tentativa: {desc} -> {n} falantes por grupo")
        if n >= MIN_FALANTES:
            return esc, desc, n, bin_ano
    return None, None, ultimo_n, None


# ---------------------------------------------------------------------------
# decisao por falante e utilidades estatisticas
# ---------------------------------------------------------------------------
def escores(pipe, X):
    """Escore com limiar natural em 0 (positivo = classe 1)."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1] - 0.5
    return pipe.decision_function(X)


def decisao_por_falante(spk_te, y_te, sc):
    t = pd.DataFrame({"spk": spk_te, "y": y_te, "s": sc}).groupby("spk").agg(y=("y", "first"), s=("s", "mean"))
    pred = (t["s"] > 0).astype(int)
    return int((pred == t["y"]).sum()), int(len(t))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    marg = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centro - marg, centro + marg


# ---------------------------------------------------------------------------
# validacao cruzada
# ---------------------------------------------------------------------------
def cv_completo(d: pd.DataFrame, tag: str) -> pd.DataFrame:
    cs = base.conjuntos_de_colunas(d)
    sets = {
        "MFCC": (cs["MFCC"][0], None),
        "WPT completa": (cs["WPT completa (global+temporal)"][0], None),
        "WPT invariante a ganho": (cs["WPT invariante a ganho"][0], None),
        f"WPT completa -> PCA {ci.K_PCA}": (cs["WPT completa (global+temporal)"][0], "pca"),
        "Hibrido MFCC + WPT": (cs["Hibrido MFCC + WPT completa"][0], None),
    }
    y = d["y"].values
    spk = d["speaker_id"].values
    splits = ci.splits_por_falante(d)

    cols_csv = ["tag", "representacao", "modelo", "fold", "acc", "bal_acc", "spk_ok", "spk_n", "n_atributos"]
    if os.path.exists(OUT_FOLDS):
        feitos = pd.read_csv(OUT_FOLDS)
    else:
        feitos = pd.DataFrame(columns=cols_csv)
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
                pipe = ci.montar(mod, red)
                pipe.fit(X[tr], y[tr])
                pred = pipe.predict(X[te])
                ok, n = decisao_por_falante(spk[te], y[te], escores(pipe, X[te]))
                row = {"tag": tag, "representacao": rep, "modelo": mod, "fold": f,
                       "acc": accuracy_score(y[te], pred), "bal_acc": balanced_accuracy_score(y[te], pred),
                       "spk_ok": ok, "spk_n": n, "n_atributos": ci.K_PCA if red else len(cols)}
                pd.DataFrame([row], columns=cols_csv).to_csv(OUT_FOLDS, mode="a", header=False, index=False)
                print(f"  [{tag[:38]}] {rep[:24]:<24} {mod:<19} fold {f + 1}/{N_SPLITS} "
                      f"acc={row['acc']:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    res = pd.read_csv(OUT_FOLDS)
    return res[res["tag"] == tag]


def resumo(res: pd.DataFrame, tag: str) -> dict:
    say("\n" + "=" * 70)
    say(f"RESULTADO [{tag}]")
    say("=" * 70)
    ordem = list(dict.fromkeys(res["representacao"]))
    out = {}
    for mod in dict.fromkeys(res["modelo"]):
        say(f"\n-- {mod} --   (por gravacao: media +- desvio dos folds | por falante: acuracia agrupada [IC 95%])")
        for rep in ordem:
            r = res[(res["modelo"] == mod) & (res["representacao"] == rep)]
            if r.empty:
                continue
            k, n = int(r["spk_ok"].sum()), int(r["spk_n"].sum())
            lo, hi = wilson(k, n)
            say(f"  {rep:<30} {100 * r['acc'].mean():.2f}% +- {100 * r['acc'].std():.2f}   |   "
                f"{100 * k / n:.1f}% [{100 * lo:.0f}-{100 * hi:.0f}] ({n} falantes)")
            out.setdefault(rep, []).append((r["acc"].mean(), k / n))

    say("\nDiferenca pareada por fold (representacao - MFCC), por gravacao, em pontos percentuais")
    piv = res.pivot_table(index=["modelo", "fold"], columns="representacao", values="acc")
    if "MFCC" in piv.columns:
        for rep in [c for c in piv.columns if c != "MFCC"]:
            linhas = []
            for mod in dict.fromkeys(res["modelo"]):
                if mod not in piv.index.get_level_values(0):
                    continue
                dif = (piv.loc[mod][rep] - piv.loc[mod]["MFCC"]).dropna() * 100
                if len(dif):
                    linhas.append(f"    {mod:<20} {dif.mean():+.2f} pp | melhor que MFCC em "
                                  f"{int((dif > 0).sum())}/{len(dif)} folds")
            say(f"  {rep}")
            for l in linhas:
                say(l)
    return {rep: (np.mean([a for a, _ in v]), np.mean([b for _, b in v])) for rep, v in out.items()}


# ---------------------------------------------------------------------------
# um experimento
# ---------------------------------------------------------------------------
def composicao_por_tipo(dm: pd.DataFrame):
    """Gravacoes por falante, por tipo de arquivo (a_n, i_l, phrase...), em cada grupo.
    Se os grupos tiverem composicoes diferentes, o modelo pode estar aprendendo o TIPO
    de gravacao e nao a patologia."""
    tipo = dm["filename"].astype(str).str.lower().str.extract(r"-([a-z0-9_]+)\.nsp", expand=False).fillna("outro")
    t = pd.crosstab(dm["grupo"], tipo)
    t = t.div(dm.groupby("grupo")["speaker_id"].nunique(), axis=0).round(2)
    say("\n-- composicao: gravacoes por falante, por tipo de gravacao --")
    say(t.to_string())


def rodar_experimento(df: pd.DataFrame, exp: dict):
    nome = exp["nome"]
    say("\n" + "#" * 70)
    say(f"# EXPERIMENTO: {nome}")
    say("#" * 70)

    d = df.copy()
    if exp.get("filtro_arquivo"):
        d = d[d["filename"].astype(str).str.lower().str.contains(exp["filtro_arquivo"], regex=False)]
    if exp.get("filtro_regex"):
        d = d[d["filename"].astype(str).str.lower().str.contains(exp["filtro_regex"], regex=True)]
    if exp.get("patologia"):
        d = d[(d["label"] == exp["patologia"]) | (d["grupo"] == "saudavel")]
    d = d.copy()
    d["y"] = (d["grupo"] == "patologico").astype(int)

    cont = d.groupby("grupo")["speaker_id"].agg(gravacoes="size", falantes="nunique")
    say(cont.to_string())
    if len(cont) < 2:
        say("[PULADO] um dos grupos ficou vazio apos o recorte. Se for o filtro de arquivo, confira se o "
            "nome dos arquivos do grupo que sumiu tem o mesmo padrao (ex.: ...-a_n.nsp).")
        QUADRO.append({"experimento": nome, "pareamento": "pulado (grupo vazio)"})
        return
    if cont["falantes"].min() < MIN_FALANTES:
        say(f"[PULADO] falantes insuficientes (minimo {MIN_FALANTES} por grupo) antes do pareamento.")
        QUADRO.append({"experimento": nome, "pareamento": "pulado (poucos falantes)"})
        return

    sp = ci.tabela_falantes(d)
    say(f"\n-- procurando pareamento (falantes nos dois grupos sao excluidos: "
        f"{int((sp['n_grupos'] > 1).sum())}) --")
    esc, desc, n_min, bin_ano = escolher_pareamento(sp, "patologico", "saudavel")
    if esc is None:
        say(f"[PULADO] nenhum pareamento deixou {MIN_FALANTES} falantes por grupo (melhor: {n_min}).")
        QUADRO.append({"experimento": nome, "pareamento": f"pulado (melhor: {n_min}/grupo)"})
        return
    say(f"-> pareamento usado: {desc} ({n_min} falantes por grupo)")

    dm = d[d["speaker_id"].isin(esc)].copy()
    ci.demografia(ci.tabela_falantes(dm), "PAREADO")
    say(f"\nConjunto pareado: {len(dm)} gravacoes | {dm['speaker_id'].nunique()} falantes")
    composicao_por_tipo(dm)
    ci.N_SPLITS = N_SPLITS
    ci.baselines(dm, nome, "PAREADO")

    tag = f"{nome}|k{N_SPLITS}|{desc}"
    res = cv_completo(dm, tag)
    r = resumo(res, tag)

    base_idade = next((m for t, v, a, m in ci.RESUMO_BASELINES if t == nome and v == "PAREADO" and a == "so idade"), np.nan)
    base_ano = next((m for t, v, a, m in ci.RESUMO_BASELINES if t == nome and v == "PAREADO" and a == "so ano da gravacao"), np.nan)
    linha = {"experimento": nome, "pareamento": desc, "falantes/grupo": n_min,
             "base idade": round(base_idade, 1), "base ano": round(base_ano, 1)}
    for rep, chave in [("MFCC", "MFCC"), ("WPT completa", "WPT"), ("Hibrido MFCC + WPT", "Hibrido")]:
        if rep in r:
            linha[f"{chave} grav"] = round(100 * r[rep][0], 1)
            linha[f"{chave} falante"] = round(100 * r[rep][1], 1)
    if "MFCC" in r and "WPT completa" in r:
        linha["WPT-MFCC (pp)"] = round(100 * (r["WPT completa"][0] - r["MFCC"][0]), 1)
    QUADRO.append(linha)


def main():
    t0 = time.time()
    ci.say = say   # demografia e baselines (checagens_idade) passam a entrar no relatorio
    ci.N_SPLITS = N_SPLITS
    ci.MODELOS = MODELOS
    df = ci.carregar()
    for exp in EXPERIMENTOS:
        rodar_experimento(df, exp)

    say("\n" + "=" * 70)
    say("QUADRO FINAL (acuracias medias dos 3 modelos; 'grav' = por gravacao, 'falante' = por falante)")
    say("'base idade' e 'base ano': o que um classificador acerta usando SO idade / SO ano da gravacao")
    say("(perto de 50% = controlado; as acuracias de voz so valem se ficarem acima disso)")
    say("=" * 70)
    say(pd.DataFrame(QUADRO).to_string(index=False))
    say(f"\nTempo total: {(time.time() - t0) / 60:.1f} min")
    with open(OUT_RELATORIO, "w", encoding="utf-8") as f:
        f.write("\n".join(RELATORIO))
    print(f"\nRelatorio salvo em {OUT_RELATORIO}")


if __name__ == "__main__":
    main()
