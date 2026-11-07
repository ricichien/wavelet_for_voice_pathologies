# -*- coding: utf-8 -*-
"""Manifesto SVD pre-avaliacao: identidades, sessao canonica e pareamento.

Le somente colunas de metadados dos CSVs e overview/diretorio central dos ZIPs.
Nao importa os pipelines historicos, nao le audio e nao treina classificadores.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

VERSION = "16-cohort-20261105-v1"
SOURCE_ROOT = Path(os.environ.get("AVFAD_SVD_SOURCE_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "SVD")))
HEALTHY_ZIP = Path(os.environ.get("AVFAD_SVD_HEALTHY_ZIP", str(Path(__file__).resolve().parents[1] / "data" / "SVD" / "healthy.zip")))
OUT_ROOT = Path(__file__).resolve().parents[1] / "results/experiments/resultados_16_svd_wavelet_participante"
PATHOLOGY_PREFIXES = {
    "balbuties": "balbuties", "dysarthrophonie": "dysarthrophonie",
    "funktionelle_dysphonie": "funktionelle_dysphonie", "laryngitis": "laryngitis",
    "psychogene_dysphonie": "psychogene_dysphonie", "rekurrensparese": "rekurrensparese",
    "spasmodische_dysphonie": "spasmodische_dysphonie", "vox_senilis": "vox_senilis",
    "zentral-laryngale": "zentral_laryngale",
}
EXPECTED_MIXED = {value.strip() for value in os.environ.get("AVFAD_EXPECTED_MIXED_IDS", "").split(",") if value.strip()}
NORMAL_VOWELS = ("a_n", "i_n", "u_n")
MAX_AGE_DIFF = 3.0
MAX_YEAR_DIFF = 1
MIN_PRIMARY_PAIRS = 50


def _task_and_session(filenames):
    names = filenames.astype(str).str.replace("\\", "/", regex=False).str.rsplit("/", n=1).str[-1]
    extracted = names.str.extract(r"^(\d+)-(.+)\.nsp$", flags=0)
    if extracted.isna().any().any():
        raise ValueError("filename NSP nao permite recuperar sessao/tarefa")
    return extracted[0].astype(str), extracted[1].str.lower()


def _find_pathology_zips(source_root):
    available = list((Path(source_root) / "dataset").glob("*.zip"))
    sources = []
    for prefix, label in PATHOLOGY_PREFIXES.items():
        matches = [p for p in available if p.stem.casefold().startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(f"Esperava um ZIP para {label}, encontrei {len(matches)}")
        sources.append((matches[0], label))
    return sources


def _source_metadata(source_root):
    frames = []
    info = []
    for filename, y in (("svd_features_mfcc_wavelet_temporal.csv", 1), ("svd_features_healthy.csv", 0)):
        path = (Path(source_root) / filename).resolve()
        frame = pd.read_csv(path, usecols=["label", "speaker_id", "filename"], dtype=str)
        if frame.isna().any().any():
            raise ValueError(f"Metadados ausentes em {filename}")
        for col in ("label", "speaker_id", "filename"):
            frame[col] = frame[col].str.strip()
            if frame[col].eq("").any():
                raise ValueError(f"Metadados vazios em {filename}:{col}")
        expected = set(PATHOLOGY_PREFIXES.values()) if y else {"healthy"}
        if set(frame.label) != expected:
            raise ValueError(f"Conjunto de rotulos inesperado em {filename}")
        if frame[["label", "filename"]].duplicated().any():
            raise ValueError(f"Par label+filename duplicado em {filename}")
        frame["source_file"] = str(path)
        frame["source_row"] = np.arange(len(frame))
        frame["row_index"] = frame.source_row
        frame["y"] = y
        frame["participant_id"] = frame.speaker_id
        frame["session_id"], frame["task"] = _task_and_session(frame.filename)
        frames.append(frame)
        st = path.stat()
        payload = frame[["label", "speaker_id", "filename"]].to_csv(index=False).encode("utf-8")
        info.append({"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
                     "metadata_sha256": hashlib.sha256(payload).hexdigest(), "rows": len(frame)})
    return pd.concat(frames, ignore_index=True), info


def _raw_metadata(source_root, healthy_zip):
    import zipfile
    frames = []
    sources = []
    for path, label in [*_find_pathology_zips(source_root), (Path(healthy_zip), "healthy")]:
        with zipfile.ZipFile(path) as z:
            overviews = [n for n in z.namelist() if n.endswith("overview.csv")]
            if len(overviews) != 1:
                raise ValueError(f"overview.csv ausente ou ambiguo em {path}")
            raw_bytes = z.read(overviews[0])
            overview = pd.read_csv(io.BytesIO(raw_bytes), dtype=str)
            required = ["AufnahmeID", "SprecherID", "AufnahmeDatum", "Geburtsdatum", "Geschlecht"]
            if not set(required).issubset(overview):
                raise ValueError(f"Schema do overview inesperado em {path}")
            overview = overview[required].rename(columns={
                "AufnahmeID": "session_id", "SprecherID": "participant_zip", "AufnahmeDatum": "session_date",
                "Geburtsdatum": "birth_date", "Geschlecht": "sex",
            })
            if overview.session_id.duplicated().any():
                raise ValueError(f"ID de sessao duplicado em {path}")
            nsp = [v for v in z.infolist() if v.filename.lower().endswith(".nsp")]
            frame = pd.DataFrame({"filename": [v.filename for v in nsp],
                                  "zip_crc": [v.CRC for v in nsp], "zip_nbytes": [v.file_size for v in nsp]})
            if frame.filename.duplicated().any():
                raise ValueError(f"NSP duplicado no diretorio central de {path}")
            frame["session_id"], _ = _task_and_session(frame.filename)
            frame = frame.merge(overview, on="session_id", how="left", validate="many_to_one")
            if frame.participant_zip.isna().any():
                raise ValueError(f"NSP sem identidade de pessoa no overview de {path}")
            frame["label"] = label
            frames.append(frame)
            sources.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                            "overview_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                            "overview_sessions": len(overview), "nsp_files": len(frame),
                            "nsp_people": int(frame.participant_zip.nunique())})
    raw = pd.concat(frames, ignore_index=True)
    for col in ("participant_zip", "session_date", "birth_date", "sex"):
        if raw.groupby("session_id")[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Overviews discordam sobre {col} da mesma sessao")
    return raw, sources


def canonical_sessions(rows):
    """Escolhe a sessao mais antiga com vogal normal, sem usar atributos de voz."""
    rows = rows[rows.task.isin(NORMAL_VOWELS)].copy()
    rows["session_date"] = pd.to_datetime(rows.session_date, errors="coerce", format="%Y-%m-%d")
    if rows.session_date.isna().any():
        raise ValueError("Data de sessao ausente/invalida em candidato a sessao canonica")
    if not rows.session_id.str.fullmatch(r"\d+").all():
        raise ValueError("ID de sessao nao numerico")
    rows["_session_number"] = rows.session_id.astype(np.int64)
    ordered = rows.sort_values(["session_date", "_session_number"], kind="stable")
    first = ordered.drop_duplicates("participant_id").set_index("participant_id").session_id
    selected = rows.session_id.eq(rows.participant_id.map(first))
    chosen = rows.loc[selected].drop(columns="_session_number").copy()
    removed = rows.loc[~selected].drop(columns="_session_number").copy()
    birth = pd.to_datetime(chosen.birth_date, errors="coerce", format="%Y-%m-%d")
    chosen["age"] = (chosen.session_date - birth).dt.days / 365.25
    if chosen.age.isna().any() or not chosen.age.between(0, 105).all():
        raise ValueError("Idade ausente/invalida na sessao canonica (faixa admitida 0..105)")
    chosen["sex"] = chosen.sex.str.strip().str.lower()
    if not chosen.sex.isin(["m", "w"]).all():
        raise ValueError("Sexo ausente/invalido na sessao canonica; esperado m/w")
    chosen["year"] = chosen.session_date.dt.year.astype(int)
    chosen["session_date"] = chosen.session_date.dt.strftime("%Y-%m-%d")
    if chosen.groupby("participant_id").session_id.nunique().ne(1).any():
        raise AssertionError("Mais de uma sessao canonica por pessoa")
    return chosen, removed


def match_participants(people):
    """Maximiza pares e minimiza custo |idade1-idade0|/3 + |ano1-ano0|."""
    required = ["participant_id", "y", "age", "sex", "year"]
    people = people[required].copy()
    if people.participant_id.duplicated().any():
        raise ValueError("Tabela de pareamento precisa ter uma linha por pessoa")
    if people[required].isna().any().any() or not people.y.isin([0, 1]).all():
        raise ValueError("Pareamento exige metadados completos e y binario")
    if not people.participant_id.astype(str).str.fullmatch(r"\d+").all():
        raise ValueError("Pareamento exige IDs numericos")
    people["_numeric_id"] = people.participant_id.astype(np.int64)
    positive = people[people.y.eq(1)].sort_values("_numeric_id").reset_index(drop=True)
    negative = people[people.y.eq(0)].sort_values("_numeric_id").reset_index(drop=True)
    columns = ["pair_id", "positive_id", "negative_id", "age_positive", "age_negative",
               "sex", "year_positive", "year_negative", "age_difference", "year_difference", "matching_cost"]
    if positive.empty or negative.empty:
        return pd.DataFrame(columns=columns)
    age_diff = np.abs(positive.age.to_numpy()[:, None] - negative.age.to_numpy()[None, :])
    year_diff = np.abs(positive.year.to_numpy()[:, None] - negative.year.to_numpy()[None, :])
    allowed = ((positive.sex.to_numpy()[:, None] == negative.sex.to_numpy()[None, :])
               & (age_diff <= MAX_AGE_DIFF) & (year_diff <= MAX_YEAR_DIFF))
    valid_cost = age_diff / MAX_AGE_DIFF + year_diff / MAX_YEAR_DIFF
    # Cada aresta valida custa <=2; a penalidade dummy supera a soma maxima
    # de todos os custos validos. Uma aresta adicional sempre tem prioridade.
    dummy_penalty = 10.0 * (len(positive) + len(negative) + 1)
    cost = np.full((len(positive), len(negative) + len(positive)), dummy_penalty)
    cost[:, :len(negative)] = np.where(allowed, valid_cost, dummy_penalty * 10)
    rr, cc = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rr, cc) if c < len(negative) and allowed[r, c]]
    records = []
    for pair_number, (r, c) in enumerate(matches, 1):
        a, b = positive.iloc[r], negative.iloc[c]
        records.append({"pair_id": f"SVD_PAIR_{pair_number:04d}", "positive_id": a.participant_id,
                        "negative_id": b.participant_id, "age_positive": a.age, "age_negative": b.age,
                        "sex": a.sex, "year_positive": a.year, "year_negative": b.year,
                        "age_difference": float(age_diff[r, c]), "year_difference": int(year_diff[r, c]),
                        "matching_cost": float(valid_cost[r, c])})
    pairs = pd.DataFrame(records, columns=columns)
    if pairs.positive_id.duplicated().any() or pairs.negative_id.duplicated().any():
        raise AssertionError("Pessoa reutilizada em mais de um par")
    return pairs


def build_manifest(source_root=SOURCE_ROOT, healthy_zip=HEALTHY_ZIP):
    """Retorna manifesto, proveniencia, exclusoes e pares; sem escrever arquivos."""
    source, source_info = _source_metadata(source_root)
    raw, zip_info = _raw_metadata(source_root, healthy_zip)
    rows = source.merge(raw, on=["label", "filename", "session_id"], how="left", validate="one_to_one")
    if rows.participant_zip.isna().any() or rows.participant_id.ne(rows.participant_zip).any():
        raise ValueError("CSV sem origem NSP ou com identidade divergente do overview")
    mixed = set(rows.groupby("participant_id").y.nunique().loc[lambda s: s.gt(1)].index)
    if not EXPECTED_MIXED:
        raise ValueError("Defina AVFAD_EXPECTED_MIXED_IDS no ambiente local para validar a coorte.")
    if mixed != EXPECTED_MIXED:
        raise ValueError("A coorte difere da lista privada de verificacao configurada.")
    removed = []
    def remove(mask, reason):
        nonlocal rows
        part = rows.loc[mask].copy()
        part["reason"] = reason
        removed.append(part)
        rows = rows.loc[~mask].copy()
    remove(rows.participant_id.isin(mixed), "mixed_participant_binary_labels")
    duplicate = rows[rows.filename.duplicated(keep=False)]
    for col in ("zip_crc", "zip_nbytes", "participant_id", "session_id", "y"):
        if duplicate.groupby("filename")[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Arquivo repetido diverge em {col}; nao deduplicar silenciosamente")
    diagnoses = rows.groupby("filename").label.agg(lambda s: "|".join(sorted(set(s))))
    rows["diagnoses"] = rows.filename.map(diagnoses)
    remove(rows.filename.duplicated(keep="first"), "duplicate_filename_across_diagnoses")
    remove(~rows.task.isin(NORMAL_VOWELS), "task_outside_normal_vowels")
    rows, later_sessions = canonical_sessions(rows)
    later_sessions["reason"] = "later_session_than_canonical"
    removed.append(later_sessions)
    people = rows[["participant_id", "y", "age", "sex", "year"]].drop_duplicates()
    if people.participant_id.duplicated().any() or rows.groupby("participant_id").y.nunique().ne(1).any():
        raise AssertionError("Pessoa com metadados/rotulos divergentes apos selecao de sessao")
    pairs = match_participants(people)
    assignment = {}
    for pair in pairs.itertuples(index=False):
        assignment[pair.positive_id] = pair.pair_id
        assignment[pair.negative_id] = pair.pair_id
    rows["pair_id"] = rows.participant_id.map(assignment).fillna("")
    rows["matched"] = rows.pair_id.ne("")
    rows["sample_id"] = "SVD__" + rows.session_id + "__" + rows.task
    if rows.sample_id.duplicated().any() or rows[["participant_id", "task"]].duplicated().any():
        raise AssertionError("Arquivo/tarefa duplicado na sessao canonica")
    out_columns = ["source_file", "source_row", "row_index", "filename", "sample_id", "participant_id",
                   "speaker_id", "session_id", "task", "y", "label", "diagnoses", "age", "sex", "year",
                   "session_date", "matched", "pair_id"]
    rows["_person_number"] = rows.participant_id.astype(np.int64)
    manifest = rows.sort_values(["_person_number", "task"])[out_columns].reset_index(drop=True)
    exclusion_columns = ["source_file", "source_row", "filename", "participant_id", "session_id", "task", "y", "label", "reason"]
    exclusions = pd.concat(removed, ignore_index=True)[exclusion_columns]
    if len(manifest) + len(exclusions) != len(source):
        raise AssertionError("Linhas de origem nao foram inteiramente contabilizadas")
    counts = manifest.groupby("y").agg(rows=("sample_id", "size"), people=("participant_id", "nunique")).to_dict("index")
    matched_counts = manifest[manifest.matched].groupby("y").agg(rows=("sample_id", "size"), people=("participant_id", "nunique")).to_dict("index")
    provenance = {"version": VERSION, "sources": source_info, "zip_metadata": zip_info,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "mixed_participants": sorted(mixed, key=int), "source_rows": len(source),
                  "complete_counts": counts, "matched_counts": matched_counts,
                  "pairs": len(pairs), "primary_eligible": bool(len(pairs) >= MIN_PRIMARY_PAIRS),
                  "exclusion_counts": exclusions.reason.value_counts().to_dict(),
                  "matching": {"sex": "exact", "max_age_difference": MAX_AGE_DIFF,
                               "max_year_difference": MAX_YEAR_DIFF, "min_pairs": MIN_PRIMARY_PAIRS,
                               "cost": "abs(age1-age0)/3 + abs(year1-year0)",
                               "objective": "maximum_cardinality_then_minimum_demographic_cost"},
                  "numeric_duplicate_check": "preexisting audit 06: all 406 repeated float32 vectors equal",
                  "limitations": "SVD previously explored; not a blind independent test. No model trained here."}
    return manifest, provenance, exclusions, pairs


def cohort_protocol(provenance):
    return f"""EXPERIMENTO 16 - PROTOCOLO DE COORTE SVD ANTES DE NOVA BA
Versao: {VERSION}
Somente metadados de CSV e overview/diretorio central dos ZIPs. Sem audio/treino.
Alvo: saudavel (y=0) versus qualquer das nove patologias com atributos (y=1).
Parkinson nao foi adicionado. SVD ja explorada, portanto estudo de desenvolvimento/
replicacao metodologica; nao teste cego nem transferencia direta do modelo AVFAD.

Exclusao anterior a desempenho: pessoas com sessoes nos dois estados binarios.
IDs: {', '.join(provenance['mixed_participants'])}.
Deduplicacao de filename entre diagnosticos conserva a primeira source_row e
lista diagnosticos em diagnoses. CRC/tamanho/identidade foram checados neste
script; igualdade de todos os vetores float32 foi verificada no inventario 06.
Tarefas somente a_n, i_n, u_n. Uma sessao por pessoa: a mais antiga por data do
overview entre sessoes com ao menos uma dessas vogais; empate menor AufnahmeID
numerico. Nao se escolhe sessao por qualidade/desempenho de classificador.
Idade em anos=(data sessao-data nascimento).days/365.25; sexo m/w; ano da sessao.
Ausencia/invalidade de data, nascimento ou sexo impede prosseguir, sem descarte
silencioso. Pessoas com uma ou duas vogais normais permanecem no manifesto.

Primaria: sexo exato, diferenca de idade <=3 anos e ano da sessao <=1 ano;
sem reposicao. Hungarian maximiza o numero de pares e, nessa cardinalidade,
minimiza abs(delta idade)/3 + abs(delta ano). Matriz ordenada por ID numerico;
solucao deterministica para mesma versao scipy/dados. Nenhuma feature acustica
ou BA participa do pareamento. Minimo predefinido de 50 pares; nao relaxar
calipers para melhorar quantidade ou desempenho. IDs do mesmo par devem ficar
no mesmo fold no runner (KFold em pares); este script NAO realiza treino.
Secundaria: coorte completa de pessoas puras apos mesma regra de sessao/tarefa,
com e sem pareamento indicados pelo flag matched, sem alterar o alvo.

Colunas source_file/source_row (zero-based, excluido cabecalho) ligam cada linha
ao CSV original; row_index e alias de source_row. sample_id e sessao+tarefa.
manifesto_coorte.csv contem a coorte completa e matched/pair_id da primaria.
pairs.csv documenta os deltas/custo; exclusoes_manifesto.csv documenta TODAS as
linhas removidas e motivos. Nenhum arquivo de origem e modificado.

CONTAGENS CONGELADAS DESTA CONSTRUCAO:
Linhas de origem: {provenance['source_rows']}
Coorte completa por y: {provenance['complete_counts']}
Coorte pareada por y: {provenance['matched_counts']}
Pares: {provenance['pairs']}; elegivel primaria >=50: {provenance['primary_eligible']}
Exclusoes por motivo: {provenance['exclusion_counts']}

Assinatura de metadados/overviews/codigo em assinatura_manifesto.json. Nao e
hash do audio nem dos valores de todos os atributos; o runner deve assinar os
arquivos numericos e seu proprio codigo antes de avaliar. Nao existe BA nova
neste documento. O protocolo de modelos precisa ser congelado separadamente.
"""


def write_manifest(out, manifest, provenance, exclusions, pairs):
    out = Path(out)
    payloads = {"manifesto_coorte.csv": manifest.to_csv(index=False).encode("utf-8-sig"),
                "pairs.csv": pairs.to_csv(index=False).encode("utf-8-sig"),
                "exclusoes_manifesto.csv": exclusions.to_csv(index=False).encode("utf-8-sig"),
                "PROTOCOLO_COORTE.txt": cohort_protocol(provenance).encode("utf-8"),
                "assinatura_manifesto.json": json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")}
    for name, content in payloads.items():
        if (out / name).exists() and (out / name).read_bytes() != content:
            raise ValueError(f"Artefato existente diverge; nao sobrescrever: {out / name}")
    out.mkdir(parents=True, exist_ok=True)
    for name, content in payloads.items():
        target = out / name
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonte", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--healthy-zip", type=Path, default=HEALTHY_ZIP)
    parser.add_argument("--saida", type=Path, default=OUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verificar", action="store_true", help="So valida/construi em memoria")
    mode.add_argument("--executar", action="store_true", help="Grava manifesto/protocolo; sem treino")
    args = parser.parse_args()
    manifest, provenance, exclusions, pairs = build_manifest(args.fonte, args.healthy_zip)
    print(cohort_protocol(provenance), flush=True)
    if args.executar:
        write_manifest(args.saida, manifest, provenance, exclusions, pairs)
        print(f"Manifesto salvo: {args.saida}")
    else:
        print("VERIFICACAO SEM ESCRITA. Nenhum treino ou audio processado.")


if __name__ == "__main__":
    main()
