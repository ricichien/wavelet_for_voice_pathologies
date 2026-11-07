# -*- coding: utf-8 -*-
"""Auditoria de metadados e nomes de arquivos; nenhum áudio/modelo é processado."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "results/experiments/documentacao_wav2vec"
BASE = ROOT / "data/metadata/AVFAD_01_00_00_1_README"
COHORT = ROOT / "results/experiments/resultados_23_wpt_rms_sequencias/metadata_wpt_rms.csv"
VALIDATION = ROOT / "data/processed/validacao_tarefa_002.csv"
WORKBOOK = BASE / "AVFAD_01_00_00.xlsx"
TASKS = {"001": "vogal /i/", "002": "vogal /a/", "003": "vogal /u/",
         "004": "CAPE-V frase 1", "005": "CAPE-V frase 2", "006": "CAPE-V frase 3",
         "007": "CAPE-V frase 4", "008": "CAPE-V frase 5", "009": "CAPE-V frase 6",
         "010": "texto O Vento Norte e o Sol", "011": "fala espontânea"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish(path, data):
    """Não substituir documentos distintos sem uma decisão explícita."""
    data = data.encode("utf8") if isinstance(data, str) else data
    if path.exists():
        require(path.read_bytes() == data, f"Saída existente diferente, preservada: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(data)


def main():
    source = pd.read_csv(COHORT, dtype={"sample_id": str, "participant_id": str, "task": str})
    people = source.drop_duplicates("participant_id").sort_values("participant_id").copy()
    require(len(people) == 411 and people.y.value_counts().to_dict() == {1: 329, 0: 82}, "Coorte diferente")
    require(source.groupby("participant_id").y.nunique().eq(1).all(), "Rótulo inconsistente")
    ids = set(people.participant_id)
    clinical = pd.read_excel(WORKBOOK, sheet_name="AVFAD", engine="openpyxl")
    clinical = clinical.dropna(subset=["File ID"]).copy()
    clinical["File ID"] = clinical["File ID"].astype(str).str.strip().str.upper()
    place_whitespace = clinical.loc[clinical["Visit place"].astype(str).ne(clinical["Visit place"].astype(str).str.strip()), "File ID"].tolist()
    clinical["Visit place"] = clinical["Visit place"].astype(str).str.strip()
    require(not clinical["File ID"].duplicated().any(), "IDs clínicos duplicados")
    validation = pd.read_csv(VALIDATION, dtype={"participant_id": str})
    require(not validation.participant_id.duplicated().any(), "IDs de validação duplicados")
    require(validation.label.value_counts().to_dict() == {"Normal": 363, "Pathological": 346}, "Classes completas inesperadas")
    classifications = pd.read_excel(WORKBOOK, sheet_name="Classifications", engine="openpyxl")
    class_codes = pd.to_numeric(classifications.iloc[:, 0], errors="coerce")
    class_counts = pd.to_numeric(classifications.iloc[:, 2], errors="coerce")
    require(class_counts[class_codes.eq(0)].sum() == 363 and class_counts[class_codes.gt(0)].sum() == 346,
            "Totais do dicionário Classifications divergem")
    require(ids <= set(clinical["File ID"]) and ids <= set(validation.participant_id), "IDs faltantes")
    codecol = "CMVD-I Dimension 1 (numeric system)"
    namecol = "CMVD-I Dimension 1 (word system)"
    clinical[codecol] = pd.to_numeric(clinical[codecol], errors="raise")
    selected = people.merge(clinical, left_on="participant_id", right_on="File ID", validate="one_to_one")
    selected = selected.merge(validation[["participant_id", "label", "numeric_code", "clinical_label"]],
                              on="participant_id", suffixes=("", "_validation"), validate="one_to_one")
    require(selected[codecol].notna().all(), "Código clínico ausente na coorte")
    require(np.array_equal(selected.y, selected[codecol].ne(0).astype(int)), "Y diverge do código original")
    require(np.array_equal(selected.y, selected.numeric_code.ne(0).astype(int)), "Y diverge da validação")
    require(np.array_equal(selected.label, selected.label_validation), "Labels divergentes")
    require(np.array_equal(selected.age, selected.Age), "Idade divergente")
    require(np.array_equal(selected.sex, selected.Sex), "Sexo divergente")
    require(np.array_equal(selected.visit_place, selected["Visit place"]), "Local divergente")
    require(np.array_equal(selected.visit_year, pd.to_datetime(selected["Visit date"]).dt.year), "Ano divergente")

    indexed = defaultdict(list)
    prt = defaultdict(list)
    supporting = []
    original_metadata = {}
    zip_stats = []
    pattern = re.compile(r"^([A-Z]+)(\d{3})\.wav$", re.I)
    for archive in sorted(ROOT.glob("AVFAD_01_00_00_*.zip")):
        stat = archive.stat()
        zip_stats.append({"archive": archive.name, "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns})
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                base = info.filename.replace("\\", "/").split("/")[-1]
                match = pattern.fullmatch(base)
                if match:
                    pid, task = match.group(1).upper(), match.group(2)
                    indexed[(pid, task)].append({"archive": archive.name, "audio_path": info.filename,
                                                "uncompressed_bytes": info.file_size,
                                                "compressed_bytes": info.compress_size, "CRC32": f"{info.CRC:08x}"})
                if base.lower().endswith("002.prt"):
                    pid = base[:-7].upper()
                    prt[pid].append({"archive": archive.name, "path": info.filename})
                if Path(base).suffix.lower() in {".xlsx", ".xls", ".csv", ".txt", ".textgrid", ".psc"}:
                    supporting.append({"archive": archive.name, "path": info.filename, "bytes": info.file_size})
                if base.lower() in {"avfad_01_00_00.xlsx", "radialgraphs_01_00_00.xlsx"}:
                    payload = z.read(info.filename)
                    workbook = pd.ExcelFile(BytesIO(payload), engine="openpyxl")
                    cells_with_scale_terms = []
                    for sheet in workbook.sheet_names:
                        values = pd.read_excel(workbook, sheet_name=sheet, header=None)
                        for column in values.columns:
                            found = values[column].astype(str).str.contains(r"GRBAS|CAPE.?V|severity|gravidade|percept", case=False, regex=True)
                            for row_index in values.index[found]:
                                cells_with_scale_terms.append({"sheet": sheet, "row_1_based": int(row_index)+1,
                                                               "column_1_based": int(column)+1, "value": str(values.at[row_index, column])})
                    original_metadata[base] = {"archive": archive.name, "path": info.filename,
                                               "sha256": hashlib.sha256(payload).hexdigest(),
                                               "sheets": workbook.sheet_names, "scale_terms_in_cells": cells_with_scale_terms}
    require(original_metadata["AVFAD_01_00_00.xlsx"]["sha256"] == sha(WORKBOOK), "Planilha local difere da original ZIP")
    require(len(zip_stats) == 5, "Esperava cinco ZIPs")
    tasks = []
    inventory = []
    for task, meaning in TASKS.items():
        present = set()
        duplicate = set()
        zero = set()
        for row in people.itertuples():
            entries = indexed[(row.participant_id, task)]
            if entries:
                present.add(row.participant_id)
                if len(entries) > 1:
                    duplicate.add(row.participant_id)
                if any(entry["uncompressed_bytes"] == 0 for entry in entries):
                    zero.add(row.participant_id)
                for entry in entries:
                    inventory.append({"participant_id": row.participant_id, "y": int(row.y), "task": task,
                                      "task_definition": meaning, **entry})
        normal = set(people.loc[people.y == 0, "participant_id"])
        pathological = ids - normal
        tasks.append({"task": task, "definition": meaning, "participants_present": len(present),
                      "normal_present": len(present & normal), "pathological_present": len(present & pathological),
                      "missing_ids": ";".join(sorted(ids-present)), "duplicate_ids": ";".join(sorted(duplicate)),
                      "zero_byte_ids": ";".join(sorted(zero)),
                      "all_database_participants_present": len({pid for (pid,t), e in indexed.items() if t == task and e})})
    tasks = pd.DataFrame(tasks)
    availability = selected[["participant_id", "y", "label", "age", "sex", "visit_year", "visit_place",
                             "numeric_code", "clinical_label", "matched_age_sex", "matched_age_sex_year"]].copy()
    for task in TASKS:
        availability[f"n_files_{task}"] = availability.participant_id.map(lambda pid: len(indexed[(pid, task)]))
    availability["n_prt002"] = availability.participant_id.map(lambda pid: len(prt[pid]))
    availability["all_three_vowels_present"] = availability[[f"n_files_{t}" for t in ("001", "002", "003")]].ge(1).all(axis=1)
    availability["all_three_vowels_unique"] = availability[[f"n_files_{t}" for t in ("001", "002", "003")]].eq(1).all(axis=1)
    vowel2_legacy = validation.set_index("participant_id").loc[people.participant_id].reset_index()
    for row in vowel2_legacy.itertuples():
        entries = indexed[(row.participant_id, "002")]
        require(len(entries) == 1 and entries[0]["archive"] == row.audio_archive and entries[0]["audio_path"] == row.audio_path,
                "Caminho da vogal002 diverge do inventário ZIP")
    conditions = (selected.groupby(["numeric_code", "y"], as_index=False)
                  .agg(participants=("participant_id", "nunique"),
                       clinical_label=("clinical_label", lambda s: " | ".join(sorted(set(s))))))
    conditions = conditions.sort_values(["y", "participants", "numeric_code"], ascending=[True, False, True])
    full_counts = (clinical.groupby(["Visit place", codecol], dropna=False).size().rename("participants").reset_index())
    full_counts["label"] = np.where(full_counts[codecol].eq(0), "Normal", "Pathological")
    locations = full_counts.groupby(["Visit place", "label"], as_index=False).participants.sum()
    demographics = []
    for y, group in selected.groupby("y"):
        demographics.append({"y": int(y), "label": "Normal" if y == 0 else "Pathological", "participants": len(group),
                             "age_mean": group.age.mean(), "age_median": group.age.median(),
                             "age_min": group.age.min(), "age_max": group.age.max(),
                             "female": int(group.sex.eq("F").sum()), "male": int(group.sex.eq("M").sum()),
                             "year_2013": int(group.visit_year.eq(2013).sum()), "year_2014": int(group.visit_year.eq(2014).sum()),
                             "year_2015": int(group.visit_year.eq(2015).sum()),
                             "matched_age_sex": int(group.matched_age_sex.sum()),
                             "matched_age_sex_year": int(group.matched_age_sex_year.sum())})
    demographics = pd.DataFrame(demographics)
    acoustic_columns = clinical.columns[16:35].tolist()
    metrics = []
    for c in acoustic_columns:
        vals = pd.to_numeric(selected[c], errors="coerce")
        metrics.append({"field": c, "finite_HSA": int(np.isfinite(vals).sum()), "missing_HSA": int((~np.isfinite(vals)).sum()),
                        "finite_normal": int(np.isfinite(vals[selected.y.eq(0)]).sum()),
                        "finite_pathological": int(np.isfinite(vals[selected.y.eq(1)]).sum())})
    metadata_fields = pd.DataFrame({"column_1_based": np.arange(1, 36), "field": clinical.columns[:35],
                                   "role_in_original_table": ["clinical/demographic"]*16 + ["Praat acoustic"]*19})
    require(len(metadata_fields) == 35, "Colunas inesperadas na planilha")
    DOC.mkdir(parents=True, exist_ok=True)
    outputs = {"27_TAREFAS_DISPONIVEIS.csv": tasks, "27_INVENTARIO_TAREFAS_HSA.csv": pd.DataFrame(inventory),
               "27_DISPONIBILIDADE_POR_PARTICIPANTE.csv": availability, "27_DISTRIBUICAO_DIAGNOSTICOS.csv": conditions,
               "27_DEMOGRAFIA_HSA.csv": demographics, "27_LOCAIS_BASE_COMPLETA.csv": locations,
               "27_CAMPOS_PLANILHA.csv": metadata_fields, "27_COMPLETUDE_PRAAT_ORIGINAL.csv": pd.DataFrame(metrics),
               "27_ARQUIVOS_SUPORTE_ZIPS.csv": pd.DataFrame(supporting), "27_HEADERS_VOGAL_002_LEGADO.csv": vowel2_legacy}
    for name, frame in outputs.items():
        publish(DOC / name, frame.to_csv(index=False, lineterminator="\n"))
    lines = ["AUDITORIA 27 — ALVO CLÍNICO E TAREFAS DISPONÍVEIS NA AVFAD/HSA", "Data: 06/11/2026.", "",
             "Escopo: inspeção de metadados e do diretório central dos cinco ZIPs.",
             "Nenhum WAV foi aberto, descomprimido ou decodificado; nenhum atributo/modelo foi ajustado.",
             "As execuções 25/26 e seus arquivos/fontes não foram modificados.", "",
             "ALVO E RASTREABILIDADE", "",
             f"Planilha original: {WORKBOOK.relative_to(ROOT)}, aba AVFAD, linhas 2–{len(clinical)+1}, 35 colunas A:AI.",
             "A1=File ID; O1=CMVD-I Dimension 1 (numeric system); P1=CMVD-I Dimension 1 (word system).",
             "A:P contém 16 campos clínicos/demográficos; Q:AI contém 19 parâmetros acústicos Praat.",
             "O alvo binário atual equivale exatamente a CMVD-I=0 (Normal) versus CMVD-I diferente de 0 (Pathological).",
             "Essa equivalência foi conferida nas 411 pessoas contra o CSV de validação e os metadados23.",
             "Também conferidos idade, sexo, local (espaços externos removidos) e ano da visita; nenhuma divergência semântica encontrada.",
             f"Local original com espaço externo: {','.join(place_whitespace) or 'nenhum'}; ISC aparece como 'HSA ' na planilha e HSA no manifesto.",
             "O código do extrator CAPE-V filtra HSA e Normal/Pathological; fonte src/extrair_features_capev_compare_avfad.py:95–133.",
             "As colunas originais não contêm escores perceptivos CAPE-V/GRBAS ou um campo de gravidade vocal.",
             "CAPE-V 004–009 identifica aqui as frases lidas; não demonstra a existência de notas perceptivas.",
             "O README cita pesquisas com GRBAS/CAPE-V, mas não fornece escores individuais em sua tabela distribuída.",
             "As quatro abas do XLSX principal e ambas as abas Male/Female de RadialGraphs foram inspecionadas por termos de escala; nenhum campo desses escores foi encontrado.",
             "RadialGraphs contém médias/DP de referência e exemplos de parâmetros acústicos (FAX/MLY), não uma tabela de avaliação perceptiva da coorte.",
             "Não se deve usar diagnóstico clínico como sinônimo automático de uma gravidade acústica/perceptiva conhecida.", "",
             "COORTE E POSSÍVEIS CONFUNDIDORES", "",
             f"Planilha AVFAD: {len(clinical)} pessoas; CMVD-I zero={int(clinical[codecol].eq(0).sum())}, não zero={int(clinical[codecol].ne(0).sum())}.",
             "Classifications!A2:C2 confirma código0=Normal com363 casos; os códigos nãozero nessa aba somam346.",
             "O CSV validacao_tarefa_002.csv também contém363 Normal e346 Pathological nos709 IDs.",
             "Há divergência de descrição com MVP (arXiv2505.20050v1), seção3.1: o artigo informa346 saudáveis/363 patológicos e vogais/a,e,o/.",
             "A distribuição local verificada é363 normais/346 patológicos e README:11–13 define/i,a,u/.",
             "Não foi obtido um manifesto individual dos autores que permita conciliar versão, rótulos e arquivos escolhidos;",
             "registrar a divergência sem concluir que houve inversão efetiva no código ou erro experimental dos autores.",
             "Fonte primária consultada: https://arxiv.org/html/2505.20050v1#S3.SS1 . Não comparar seus percentuais diretamente comHSA411.",
             "Coorte corrente preservada: 411 HSA, 82 normais e 329 patológicos, 2.465 frases já avaliadas.",
             "Tabela 27_LOCAIS_BASE_COMPLETA.csv mostra a composição por local na base inteira;", 
              "ampliar a coorte para outro local mudaria aquisição/composição e exige protocolo separado.",
              "Na base inteira, HSA tem 329 patológicos/82 normais; ESSUA tem 274 normais e nenhum patológico; HSJ tem 8/2 e ULSM 9/5 (patológicos/normais).",
              "Exemplo aritmético pós-hoc, sem áudio ou treinamento: prever patologia apenas quando o local é HSA dá TP329/FN17/TN281/FP82,",
              f"acurácia={610/709:.6f} e BA={(329/346+281/363)/2:.6f} nos 709. Essa regra NÃO é um modelo vocal válido ou validação externa.",
              "O exemplo demonstra associação entre local e rótulo: porcentagens de 80–90 na base inteira podem refletir confundimento,",
              "sem demonstrar que algum artigo específico dependa desse mecanismo. No HSA, essa regra colapsa para BA=0,5.", ""]
    for row in demographics.itertuples():
        lines.append(f"{row.label}: n={row.participants}, idade média={row.age_mean:.4f}, mediana={row.age_median:.1f}, faixa={row.age_min}–{row.age_max}; F={row.female}, M={row.male}; 2013={row.year_2013}, 2014={row.year_2014}, 2015={row.year_2015}.")
    lines += ["", "Pareamentos legados são flags, não a coorte principal: 81+81 por idade/sexo; 68+68 por idade/sexo/ano.",
              "Todos os 411 continuam na proposta de tarefas adicionais. Não selecionar o subconjunto mais fácil para elevar BA.",
              f"Patológicos na coorte: {conditions.loc[conditions.y.eq(1), 'numeric_code'].nunique()} códigos distintos.",
              "Distribuição completa em 27_DISTRIBUICAO_DIAGNOSTICOS.csv; códigos e nomes vêm do CSV clínico original.",
              "Nomes diferentes do mesmo código são listados juntos, sem separar a classe: código1220 tem29 pessoas e duas grafias do apóstrofo.", ""]
    for row in conditions.head(12).itertuples():
        lines.append(f"Código {row.numeric_code}, {row.clinical_label}: n={row.participants}, y={row.y}.")
    lines += ["", "Esses grupos documentam heterogeneidade do alvo; não provam que casos difíceis estejam rotulados incorretamente.",
              "Não foram feitos novos modelos/subgrupos nem relabeling/exclusões. Não inferir gravidade que não está registrada.", "",
              "TAREFAS ORIGINAIS E COMPLETUDE", "",
              "Fonte: README_01_00_00.txt, linhas 6 e 11–22. Cada WAV pode conter três repetições;",
              "a existência de um arquivo não confirma três emissões válidas, segmentação ou qualidade acústica.",
              "001=/i/, 002=/a/, 003=/u/; 004–009=seis frases CAPE-V; 010=texto; 011=fala espontânea.", ""]
    for row in tasks.itertuples():
        lines.append(f"{row.task} ({row.definition}): {row.participants_present}/411, normal={row.normal_present}/82, patológico={row.pathological_present}/329; ausentes={row.missing_ids or 'nenhum'}; duplicados={row.duplicate_ids or 'nenhum'}; vazios={row.zero_byte_ids or 'nenhum'}.")
    lines += ["", f"Três vogais presentes: {int(availability.all_three_vowels_present.sum())}/411; três vogais únicas: {int(availability.all_three_vowels_unique.sum())}/411.",
              f"Arquivo002.prt presente: {int(availability.n_prt002.ge(1).sum())}/411; extensão corresponde a áudio+anotação Praat binária segundo README:22.",
              "As seis frases existem nominalmente para411 pessoas (2.466 arquivos), mas o pipeline mantém2.465 gravações: CCO004 já foi rejeitado por duração0,093s.",
              "Fonte dessa exclusão legada: results/legacy/erros_capev_compare_avfad.csv:2. Não alterar essa regra neste inventário.",
              "Nomes, tamanhos, CRC armazenado e caminhos em 27_INVENTARIO_TAREFAS_HSA.csv. CRC não foi recalculado.",
              "Nenhum header de áudio foi lido nesta auditoria. Não afirmar duração/taxa/canais utilizáveis apenas pelo inventário.",
              "O CSV legado validacao_tarefa_002.csv já possui duração/taxa/canais para /a/, mas novos /i,u/ precisam de validação de leitura.", "",
              "No CSV legado, os411/a/constam como encontrados, todosmono/48kHz/16bits; duração5,671–45,0583s, mediana15,1757s.",
              "Os caminhos de todos os411/a/no CSV foram conferidos contra os nomes dos ZIPs. Esses headers não foram relidos aqui.",
              "27_HEADERS_VOGAL_002_LEGADO.csv mantém esses metadados e deixa explícito que não são uma nova inspeção dos WAVs.", "",
              "DECISÃO TÉCNICA QUE ESTES ARQUIVOS APOIAM", "",
              "É possível planejar uma comparação frases versus vogal /a/ versus frases+/a/ com a mesma coorte,",
              "desde que os arquivos inventariados passem por validação de leitura. /i,u/ são candidatos adicionais predefinidos.",
              "As vogais podem oferecer informação de fonação complementar à fala conectada; isso permanece hipótese, não ganho medido.",
              "Usar as mesmas partições externas por pessoa, todas as tarefas da pessoa no mesmo fold, normalização/seleção dentro do treino.",
              "Manter referência somente frases e testar Wavelet, encoder e fusão em entradas correspondentes;",
              "não atribuir ao método Wavelet um ganho causado apenas pela inclusão de novas tarefas.",
              "Adicionar gravações melhora a observação de cada pessoa, mas o tamanho amostral independente permanece 411.",
              "Os 19 parâmetros Praat originais são de análise de /a/ anotada segundo README:6;",
              "são dados acústicos disponíveis, não diagnóstico, gravidade, notas CAPE-V ou informação que substitua folds sem vazamento.",
              "Se usados como baseline, sua origem/segmento/falhas devem ser declarados e a avaliação deve permanecer por pessoa.",
              "Esta auditoria não determina que BA de 80–90% seja alcançável nesta tarefa/coorte; nenhuma métrica de classificador vocal foi produzida.", "",
              "FONTES E LIMITES", "",
              "27_CAMPOS_PLANILHA.csv enumera todos os 35 campos da aba AVFAD; 27_COMPLETUDE_PRAAT_ORIGINAL.csv documenta finitos.",
              "27_ARQUIVOS_SUPORTE_ZIPS.csv lista XLSX/CSV/TXT/TextGrid/PSC encontrados em todos os ZIPs;",
              "não foram encontrados escores individuais CAPE-V/GRBAS nas duas planilhas distribuídas inspecionadas.",
              "Não foi aberta cada anotação binária PRT nem procurado material externo à cópia local do corpus.",
              "O escopo é AVFAD; a SVD não foi treinada, extraída nem validada aqui.",
              "Scripts e resultados anteriores permanecem intactos; os novos arquivos são metadados de auditoria.", ""]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in (COHORT, VALIDATION, WORKBOOK, BASE / "README_01_00_00.txt")}
    lines += [f"SHA256 {name}: {digest}" for name, digest in inputs.items()]
    lines += [f"Auditor: src/{Path(__file__).name}; SHA256={sha(__file__)}", ""]
    report = DOC / "27_AUDITORIA_ALVO_TAREFAS.txt"
    publish(report, "\n".join(lines))
    output_hashes = {name: sha(DOC / name) for name in outputs}
    output_hashes[report.name] = sha(report)
    manifest = {"scope": "metadata_and_zip_directory_only_no_audio_no_model", "cohort_people": 411,
                "cohort_recordings_current": 2465, "vowels_complete": int(availability.all_three_vowels_unique.sum()),
                "inputs_sha256": inputs, "zip_metadata_only": zip_stats, "small_metadata_read_from_zip": original_metadata,
                "outputs_sha256": output_hashes,
                "auditor_sha256": sha(__file__)}
    publish(DOC / "27_MANIFESTO_AUDITORIA.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"people": 411, "vowels_complete": int(availability.all_three_vowels_unique.sum()),
                      "tasks": tasks.to_dict("records"), "clinical_columns": clinical.columns[:35].tolist(),
                      "locations": locations.to_dict("records"), "report": str(report)}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
