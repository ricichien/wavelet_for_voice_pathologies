# -*- coding: utf-8 -*-
"""Auditoria do experimento 24: estatísticas e previsões, sem ajustar modelos.

Reutiliza as operações aritméticas independentes do auditor 22. Nenhum método
fit é chamado. A única saída é um relatório novo, após todas as verificações.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


EXP24 = module("exp24_for_audit", "24_avaliar_normalizacao_wpt_avfad.py")
AUDIT = module("audit22_helpers_for_24", "verificar_wpt_wav2vec22.py")
require, sha, read_csv = AUDIT.require, AUDIT.sha, AUDIT.read_csv


def audit(out, report):
    started = time.perf_counter()
    marker = out / "ARTEFATOS_COMPLETOS.json"
    require(marker.is_file(), "Experimento 24 ainda não concluído; nenhum relatório escrito")
    exports = json.loads(marker.read_text(encoding="utf8"))
    for name, digest in exports.items():
        require(Path(name).name == name and sha(out / name) == digest, f"Exportação alterada: {name}")
    identity = json.loads((out / "assinatura_execucao.json").read_text(encoding="utf8"))
    for category in ("code", "input_files"):
        for name, digest in identity[category].items():
            require(sha(name) == digest, f"Fonte/insumo alterado: {name}")
    actual_protocol = hashlib.sha256((out / "PROTOCOLO.txt").read_text(encoding="utf8").encode("utf8")).hexdigest()
    require(actual_protocol == identity["protocol_sha256"], "Protocolo diferente da assinatura")
    seeds = tuple(identity["configuration"]["seeds"])
    require(seeds == EXP24.SEEDS, "Auditoria final exige as seis sementes previstas")
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    matrices, meta, names = EXP24.load_inputs(ROOT, ROOT / EXP24.DATA23_REL)
    y = meta.y.to_numpy()
    r13, r17 = EXP24.references(ROOT, ROOT / EXP24.REF17_REL, meta, seeds)
    raw = read_csv(out / "predicoes_oof_participante.csv")
    summary = read_csv(out / "metricas_por_seed.csv")
    require(len(raw) == 7398 and len(summary) == 18, "Contagens OOF/seed incorretas")
    require(not raw.duplicated(["seed", "arm", "participant_id"]).any(), "OOF duplicado")
    pd.testing.assert_frame_equal(read_csv(out / "particoes_participantes.csv"), EXP24.EXP19.fold_table(meta, seeds))
    totals = {"outer_folds": 0, "classificadores": 0, "checkpoint_hashes": 0,
              "preprocessadores_treino": 0, "margens_teste_reconstruidas": 0,
              "margens_treino_reconstruidas": 0, "max_margin_error": 0.,
              "metricas_fold_recalculadas": 0}
    all_attributes, all_metrics = [], []
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP24.EXP13.row_splits(meta, 5, seed), 1):
            require(not set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id), "Pessoa compartilhada")
            destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            frames, arrays = EXP24.load_checkpoint(destination, signature)
            EXP24.validate(frames, arrays, meta, train, test, names, seed, fold, r13, r17)
            totals["outer_folds"] += 1
            totals["checkpoint_hashes"] += len(json.loads((destination / "CONCLUIDO.json").read_text(encoding="utf8"))["sha256"])
            for arm in EXP24.ARMS:
                prefix = f"{arm}__"
                AUDIT.verify_preprocessor(matrices[arm], y, train, arrays, prefix, 128)
                totals["preprocessadores_treino"] += 1
                np.testing.assert_array_equal(arrays[prefix + "classes"], [0, 1])
                counts = np.bincount(y[train], minlength=2)
                np.testing.assert_allclose(arrays[prefix + "class_weight"], len(train) / (2 * counts), rtol=0, atol=1e-14)
                for selected, filename, counter in ((test, "predicoes_teste_gravacao.csv", "margens_teste_reconstruidas"),
                        (train[np.unique(np.linspace(0, len(train)-1, 3, dtype=int))], "predicoes_treino_gravacao.csv", "margens_treino_reconstruidas")):
                    values = AUDIT.transform(matrices[arm], selected, arrays, prefix).astype(np.float64)
                    margin = (values @ arrays[prefix + "coef"].T).ravel() + arrays[prefix + "intercept"][0]
                    observed = frames[filename].query("arm == @arm").set_index("sample_id").loc[meta.iloc[selected].sample_id].score.to_numpy()
                    error = float(np.max(np.abs(margin - observed)))
                    require(error < 1e-7, f"Margens diferentes: {arm}, erro={error}")
                    totals["max_margin_error"] = max(totals["max_margin_error"], error)
                    totals[counter] += len(selected)
                totals["classificadores"] += 1
            filenames = {("teste", "participante"): "predicoes.csv", ("treino", "participante"): "predicoes_treino_participante.csv",
                         ("teste", "gravacao"): "predicoes_teste_gravacao.csv", ("treino", "gravacao"): "predicoes_treino_gravacao.csv"}
            for row in frames["metricas.csv"].itertuples():
                AUDIT.compare_numbers(AUDIT.metrics(frames[filenames[row.phase, row.unit]].query("arm == @row.arm")), row)
                totals["metricas_fold_recalculadas"] += 1
            pd.testing.assert_frame_equal(raw[(raw.seed == seed) & (raw.fold == fold)].reset_index(drop=True), frames["predicoes.csv"], check_dtype=False)
            all_attributes.append(frames["atributos.csv"])
            all_metrics.append(frames["metricas.csv"])
    pd.testing.assert_frame_equal(read_csv(out / "atributos_selecionados.csv"), pd.concat(all_attributes, ignore_index=True), check_dtype=False)
    pd.testing.assert_frame_equal(read_csv(out / "metricas_por_fold.csv"), pd.concat(all_metrics, ignore_index=True), check_dtype=False)
    for row in summary.itertuples():
        group = raw[(raw.seed == row.seed) & (raw.arm == row.arm)]
        require(len(group) == 411 and group.participant_id.nunique() == 411, "OOF incompleto")
        measured = AUDIT.metrics(group)
        measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        AUDIT.compare_numbers(measured, row)
        aucs, weights = [], []
        for _, fold_group in group.groupby("fold"):
            aucs.append(AUDIT.metrics(fold_group)["AUC"])
            weights.append(int((fold_group.y == 0).sum() * (fold_group.y == 1).sum()))
        require(abs(np.average(aucs, weights=weights) - row.AUC_media_folds_ponderada_pares) < 1e-14, "AUC ponderada incorreta")
    contrasts = read_csv(out / "comparacoes_pareadas.csv")
    require(len(contrasts) == 6, "Contrastes incompletos")
    ci_error = 0.
    for row in contrasts.itertuples():
        selected = [20261105] if row.escopo == "principal_seed20261105" else [101, 102, 103, 104, 105]
        expected = AUDIT.paired_interval(raw, selected, EXP24.CONTRASTS[row.comparacao])
        actual = [row.delta_BA, row.IC95_inferior, row.IC95_superior]
        np.testing.assert_allclose(expected, actual, atol=1e-14, rtol=0)
        ci_error = max(ci_error, float(np.max(np.abs(expected - actual))))
    require(totals["classificadores"] == 90 and totals["preprocessadores_treino"] == 90 and totals["checkpoint_hashes"] == 210, "Auditoria incompleta")
    require(totals["margens_teste_reconstruidas"] == 44370, "Nem todas as margens de teste foram reconstruídas")
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")
    lines = ["AUDITORIA FINAL DO EXPERIMENTO 24 — APROVADA", f"Verificado em: {now}",
             f"Auditor: {Path(__file__).resolve()}", f"SHA256 auditor: {sha(__file__)}",
             f"SHA256 operações aritméticas compartilhadas com auditor22: {sha(HERE / 'verificar_wpt_wav2vec22.py')}", "",
             "Nenhum modelo foi ajustado e nenhum áudio foi relido. O único arquivo criado é este relatório.",
             "Hashes dos exports, fontes e insumos, partições e checkpoints confirmados.",
             "Mediana, variabilidade, ANOVA, ranking e escala recalculados somente nas linhas de treino.",
             "O cálculo F usa f_classif; os demais cálculos são aritméticos, sem estimador.fit.",
             "Margens de TODAS as gravações de teste e três gravações de treino por ajuste reconstruídas",
             "com normalizadores e coeficientes exportados; decisões por pessoa conferidas pela média das frases.",
             "Pesos de classe do SVC verificados pelas gravações de treino; a avaliação principal é por pessoa.",
             "Controle A validado contra as referências 13/17, sem escolher a melhor semente.", ""]
    lines.extend(f"{key}: {value}" for key, value in totals.items())
    lines += [f"OOF_participante: {len(raw)}", f"resultados_seed_braco: {len(summary)}", f"contrastes_IC_recalculados: {len(contrasts)}",
              f"erro_maximo_contrastes_IC: {ci_error}", f"segundos_auditoria: {time.perf_counter()-started:.3f}", "",
              "ICs condicionais às previsões; não refazem seleção/treino nem corrigem exploração histórica.",
              "A auditoria computacional não equivale a validação clínica ou avaliação externa na SVD.", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf8") as stream:
        stream.write("\n".join(lines))
    print("AUDITORIA24 APROVADA", json.dumps(totals, sort_keys=True), flush=True)
    print("Relatório:", report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", type=Path, default=ROOT / EXP24.OUT_REL)
    parser.add_argument("--relatorio", type=Path, default=ROOT / "results/experiments/documentacao_wav2vec/24_VERIFICACAO_INDEPENDENTE.txt")
    args = parser.parse_args()
    out, report = args.experimento.resolve(), args.relatorio.resolve()
    require(out.is_relative_to(ROOT / "results/experiments") and report.is_relative_to(ROOT / "results/experiments"), "Caminho fora dos experimentos")
    audit(out, report)


if __name__ == "__main__":
    main()
