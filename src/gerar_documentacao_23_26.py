# -*- coding: utf-8 -*-
"""Documentacao23–26 a partir de exports completos; nunca importa/ajusta modelos.

Uso: .venv\Scripts\python.exe -B src/gerar_documentacao_23_26.py
Sem ARTEFATOS_COMPLETOS, um experimento e PENDENTE: checkpoints e metricas
parciais nao sao usados. A fonte de cada numero e registrada em manifesto.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DOC_REL = Path("results/experiments/documentacao_wav2vec")
SEEDS = {20261105, 101, 102, 103, 104, 105}
SPECS = {
    23: ("resultados_23_wpt_rms_sequencias", ()),
    24: ("resultados_24_normalizacao_wpt", ("A_RAW", "B_RMS", "C_NO_LOG")),
    25: ("resultados_25_wavelet_cnn_tcn", ("A_CNN_RAW", "B_CNN_RMS", "C_TCN_RAW", "D_TCN_RMS", "E_MLP_RESUMOS_RMS")),
    26: ("resultados_26_hibrido_wavelet_temporal_wav2vec", ("A_WAV_MLP32_ONLY", "B_WAV_MLP64_BUDGET", "C_CNN_RMS_WAV", "D_TCN_RMS_WAV")),
}
SUPPORT = {
    23: ("test_wpt_extracao23.py", "test_verificar_extracao23.py", "verificar_extracao23.py"),
    24: ("test_wpt_normalizacao24.py", "verificar_normalizacao24.py"),
    25: ("test_wavelet_cnn_tcn25.py", "verificar_redes25_26.py"),
    26: ("test_wavelet_hibrido26.py", "verificar_redes25_26.py"),
}
AUDITS = ("23_VERIFICACAO_EXTRACAO.txt", "24_VERIFICACAO_INDEPENDENTE.txt",
          "25_26_VERIFICACAO_REDES.txt", "25_26_VERIFICACAO_REDES.json",
          "25_VERIFICACAO_REDES.txt", "26_VERIFICACAO_REDES.txt")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def complete_exports(folder):
    """Nenhum arquivo de resultado parcial e lido sem marcador final valido."""
    marker = folder / "ARTEFATOS_COMPLETOS.json"
    if not marker.is_file():
        return None
    mapping = json.loads(marker.read_text(encoding="utf8"))
    require(isinstance(mapping, dict) and mapping, f"Marcador invalido: {marker}")
    for name, expected in mapping.items():
        require(Path(name).name == name, f"Caminho invalido no marcador: {name}")
        require(sha(folder / name) == expected, f"Artefato completo alterado: {folder/name}")
    require("assinatura_execucao.json" in mapping, f"Assinatura ausente em {folder}")
    identity = json.loads((folder / "assinatura_execucao.json").read_text(encoding="utf8"))
    for name, expected in identity["code"].items():
        require(sha(name) == expected, f"Fonte diverge da assinatura: {name}")
    return {"marker_sha256": sha(marker), "files": mapping, "identity": identity,
            "folder": str(folder.relative_to(ROOT))}


def metrics(folder, expected_arms):
    frame = pd.read_csv(folder / "metricas_por_seed.csv", float_precision="round_trip")
    require(set(frame.seed) == SEEDS and set(frame.arm) == set(expected_arms), "Ciclo completo exige seis sementes e todos os bracos")
    require(len(frame) == len(SEEDS) * len(expected_arms) and not frame.duplicated(["seed", "arm"]).any(), "Metricas incompletas/duplicadas")
    require(frame.n.eq(411).all() and (frame.TN+frame.FP).eq(82).all()
            and (frame.TP+frame.FN).eq(329).all(), "Coorte de metricas alterada")
    oof = pd.read_csv(folder / "predicoes_oof_participante.csv", float_precision="round_trip")
    for row in frame.itertuples():
        group = oof[(oof.seed == row.seed) & (oof.arm == row.arm)]
        require(len(group) == 411 and not group.participant_id.duplicated().any(), "OOF por pessoa incompleto")
        require(group.y.value_counts().to_dict() == {1:329, 0:82}, "Classes OOF alteradas")
        require(np.array_equal(group.predicao, (group.score >= group.limiar).astype(int)), "Decisao OOF nao corresponde ao limiar")
        counts = {"TN": int(((group.y == 0) & (group.predicao == 0)).sum()),
                  "FP": int(((group.y == 0) & (group.predicao == 1)).sum()),
                  "FN": int(((group.y == 1) & (group.predicao == 0)).sum()),
                  "TP": int(((group.y == 1) & (group.predicao == 1)).sum())}
        require(all(counts[key] == getattr(row, key) for key in counts), "Matriz de confusao OOF diverge do resumo")
        ba = .5 * (counts["TN"]/82 + counts["TP"]/329)
        require(abs(ba-row.BA) < 1e-14, "BA do resumo diverge das predicoes OOF")
    return frame


def summary_rows(experiment, frame, arms, origin):
    rows = []
    for arm in arms:
        row = {"experimento": experiment, "arm": arm, "estado": "PENDENTE", "unidade": "participante",
               "BA_principal_20261105": None, "BA_estabilidade_media": None, "BA_estabilidade_DP": None,
               "BA_estabilidade_minimo": None, "BA_estabilidade_maximo": None,
               "n_sementes_estabilidade": 0, "sens_principal": None, "esp_principal": None,
               "acuracia_principal": None, "AUC_folds_principal": None, "fonte": origin}
        if frame is not None:
            subset = frame[frame.arm == arm]
            primary = subset[subset.seed == 20261105].iloc[0]
            stability = subset[subset.seed != 20261105]
            row.update(estado="COMPLETO", BA_principal_20261105=float(primary.BA),
                       BA_estabilidade_media=float(stability.BA.mean()), BA_estabilidade_DP=float(stability.BA.std(ddof=1)),
                       BA_estabilidade_minimo=float(stability.BA.min()), BA_estabilidade_maximo=float(stability.BA.max()),
                       n_sementes_estabilidade=len(stability), sens_principal=float(primary.sensibilidade),
                       esp_principal=float(primary.especificidade), acuracia_principal=float(primary.acuracia),
                       AUC_folds_principal=float(primary.AUC_media_folds_ponderada_pares))
        rows.append(row)
    return rows


def freeze_sources(documentation, completed):
    target = documentation / "codigo_congelado_23_26"
    records = []
    for experiment, state in completed.items():
        if state is None:
            continue
        signed = state["identity"]["code"]
        sources = [(Path(name), digest, True) for name, digest in signed.items()]
        sources += [(ROOT / "src" / name, None, False) for name in SUPPORT[experiment] if (ROOT / "src" / name).is_file()]
        for source, expected, is_signed in sources:
            digest = sha(source)
            require(expected is None or digest == expected, f"Fonte assinada mudou: {source}")
            destination = target / source.name
            if destination.exists():
                require(sha(destination) == digest, f"Copia congelada diferente preservada: {destination}")
            else:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            records.append({"experimento": experiment, "fonte": str(source.relative_to(ROOT)),
                            "copia": str(destination.relative_to(ROOT)), "sha256": digest,
                            "assinado_pela_execucao": is_signed,
                            "observacao": "" if is_signed else "Suporte copiado nesta data; nao integra a assinatura do treino"})
    return records


def fmt(value):
    return "PENDENTE" if pd.isna(value) else f"{float(value):.6f}"


def generate(documentation):
    require(documentation.is_relative_to(ROOT / "results/experiments"), "Documentacao precisa ficar dentro de results/experiments")
    previous = documentation / "MANIFESTO_DOCUMENTACAO_23_26.json"
    if previous.exists():
        old = json.loads(previous.read_text(encoding="utf8"))
        for name, digest in old["generated_files"].items():
            require(sha(documentation/name) == digest, f"Documento gerado foi editado; preservar: {name}")
    generated_names = ("RESUMO_TECNICO_23_26.txt", "TABELA_METRICAS_23_26.csv", "TABELA_CONTRASTES_23_26.csv", "FONTES_CONGELADAS_23_26.csv")
    if not previous.exists():
        require(not any((documentation/name).exists() for name in generated_names), "Documento existente sem manifesto; preservar")
    completed, rows, contrasts = {}, [], []
    for experiment, (folder_name, arms) in SPECS.items():
        folder = ROOT / "results/experiments" / folder_name
        state = completed[experiment] = complete_exports(folder)
        if experiment == 23:
            continue
        frame = metrics(folder, arms) if state else None
        rows += summary_rows(experiment, frame, arms, str((folder/"metricas_por_seed.csv").relative_to(ROOT)))
        if state:
            comparison = pd.read_csv(folder/"comparacoes_pareadas.csv", float_precision="round_trip")
            require(set(comparison.escopo) == {"principal_seed20261105", "estabilidade_secundaria"}, "Contrastes sem ambos os escopos")
            require(np.isfinite(comparison[["delta_BA", "IC95_inferior", "IC95_superior"]].to_numpy()).all(), "Contraste nao finito")
            comparison.insert(0, "experimento", experiment)
            comparison["fonte"] = str((folder/"comparacoes_pareadas.csv").relative_to(ROOT))
            contrasts += comparison.to_dict("records")
    # Contexto historico: apenas duas referencias21, nunca misturadas ao ciclo24–26.
    history_folder = ROOT / "results/experiments/resultados_21_wpt_wav2vec"
    history = complete_exports(history_folder)
    historical_rows = []
    if history:
        frame21 = metrics(history_folder, ("A_WPT", "B_WAV2VEC", "C_HIBRIDO", "D_WAV2VEC_256"))
        historical_rows = summary_rows(21, frame21, ("A_WPT", "D_WAV2VEC_256"), str((history_folder/"metricas_por_seed.csv").relative_to(ROOT)))
        rows += historical_rows
    frozen = freeze_sources(documentation, completed)
    audits = {name: sha(documentation/name) for name in AUDITS if (documentation/name).is_file()}
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["SÍNTESE TÉCNICA DOS EXPERIMENTOS 23–26 — AVFAD/HSA", f"Gerado em: {now}",
             "Fonte: exports finais com hashes válidos; resultados parciais não são usados.",
             "Este documento descreve desenvolvimento na AVFAD; não constitui conclusão clínica ou estado da arte.", "",
             "ESTADO DA CAMPANHA"]
    for experiment in SPECS:
        lines.append(f"Experimento {experiment}: {'COMPLETO' if completed[experiment] else 'PENDENTE — sem marcador final; nenhuma métrica parcial consolidada'}.")
    lines += ["", "COORTE E AVALIAÇÃO", "411 participantes HSA: 82 normais e 329 patológicos; 2.465 gravações CAPE-V.",
              "Participante é a unidade principal. Cinco folds externos com pessoas disjuntas.",
              "Semente 20261105 principal; 101–105 para estabilidade. Média e DP de estabilidade usam somente essas cinco sementes.",
              "Acurácia balanceada (BA)=média de sensibilidade e especificidade. Acurácia simples é secundária.",
              "Não escolher a melhor semente; DP entre sementes não é intervalo de confiança.", "", "EXTRAÇÃO 23"]
    if completed[23]:
        folder = ROOT / completed[23]["folder"]
        controls = pd.read_csv(folder/"CONTROLE_REFERENCIA_48KHZ.csv", float_precision="round_trip")
        quality = pd.read_csv(folder/"QUALIDADE_AUDIO.csv", float_precision="round_trip")
        meta = pd.read_csv(folder/"metadata_wpt_rms.csv", float_precision="round_trip")
        require(len(controls) == len(meta) == len(quality) == 2465 and controls.equivalent.eq(True).all(), "Controle23 incompleto")
        require(float(controls.max_abs_error.max()) == 0, "RAW23 deixou de reproduzir exatamente a referencia")
        lines += ["WPT sym5 a 48 kHz, níveis 5/6: 96 nós × 5 descritores = 480 atributos por janela.",
                  "Descritores: energia relativa, logenergia, entropia normalizada, curtose e ZCR.",
                  "Janelas de 400 ms, avanço de 200 ms. Resumos por gravação: média, DP, q10 e q90 = 1.920 atributos.",
                  "VAD aplicado antes do ganho; RMS ativo global alvo 0,1 por gravação, preservando a variação entre janelas.",
                  "Sem normalização individual por janela, remoção de gravações ou clipping artificial dos picos normalizados.",
                  f"Controle RAW: 2.465/2.465 aprovados; maior erro absoluto = {controls.max_abs_error.max():.1f}.",
                  f"Sequências: {int(meta.window_count.sum())} janelas; mínimo/mediana/máximo por gravação="
                  f"{int(meta.window_count.min())}/{meta.window_count.median():.1f}/{int(meta.window_count.max())}.",
                  f"Flags: {int((quality.channels_original>1).sum())} gravações estéreo;"
                  f" {int(quality.flag_duration_over_60s.sum())} acima de 60 s;"
                  f" {int(quality.flag_any_full_scale_sample.sum())} com amostras em escala máxima;"
                  f" {int(quality.flag_possible_channel_cancellation.sum())} suspeitas de cancelamento entre canais.",
                  "Essas flags não demonstram defeito, etiologia ou indicação para exclusão.",
                  "Auditoria independente 23: vetores, sequências, hashes, ordem/IDs e referência; ver laudo quando presente."]
    else:
        lines.append("PENDENTE: nenhum resultado de extração foi presumido.")
    lines += ["", "PIPELINES 24–26", "24 — A_RAW: 1.920 atributos WPT; B_RMS: 1.920 com RMS; C_NO_LOG: RAW sem os 384 atributos de logenergia.",
              "Imputação, remoção de constantes, ANOVA com 128 atributos e StandardScaler no treino; SVC linear, C=1, pesos balanceados.",
              "Decisão por pessoa pela média das margens das frases; limiar 0. Remover logenergia não equivale a RMS.",
              "25 — CNN e TCN compactas em sequências RAW/RMS; MLP com resumos RMS como controle estático.",
              "Interpolação linear da extensão completa para 128 passos: sem truncar início/fim, porém com perda possível de transientes.",
              "TCN offline não causal, padding simétrico. GroupNorm por exemplo; sem BatchNorm entre gravações.",
              "26 — CNN_RMS/TCN_RMS combinadas aos embeddings congelados XLS-R 300M do experimento 20.",
              "Apenas a projeção MLP e as redes Wavelet são treinadas. Há controle wav2vec isolado e controle de maior capacidade.",
              "Em 25/26: média dos logits das frases antes da BCE ponderada por participante.",
              "Escalas/seleção de época usam apenas treino/validação internos por pessoa; refit do zero no treino externo completo.",
              "O teste externo é avaliado depois desse refit. Limiar 0 fixo; não selecionar época pela BA externa.",
              "Opções exatas, parâmetros e ambiente pertencem aos respectivos PROTOCOLO.txt e assinatura_execucao.json.", "", "RESULTADOS POR PARTICIPANTE"]
    for row in rows:
        lines.append(f"Exp. {row['experimento']} {row['arm']}: {row['estado']}; BA principal={fmt(row['BA_principal_20261105'])}; "
                     f"BA estabilidade média={fmt(row['BA_estabilidade_media'])}; DP={fmt(row['BA_estabilidade_DP'])}.")
    if historical_rows:
        lines += ["", "CONTEXTO HISTÓRICO 21", "As referências do experimento 21 são resultados do protocolo fixo anterior, preservados em linhas próprias.",
                  "WPT A_WPT e wav2vec D_WAV2VEC_256 são referências de desenvolvimento; não são novos resultados de 24–26.",
                  "Diferenças descritivas em relação ao 21 não substituem um contraste pareado nem validam generalização externa."]
    lines += ["", "CONTRASTES PAREADOS", "ΔBA e IC95 são copiados dos exports finais verificados; sinal conforme o nome da comparação.",
              "Bootstrap estratificado por participante, 2.000 repetições; todas as repetições de cada pessoa permanecem no mesmo cluster."]
    for row in contrasts:
        lines.append(f"Exp. {row['experimento']} {row['escopo']} {row['comparacao']}: ΔBA={row['delta_BA']:+.6f}; "
                     f"IC95=[{row['IC95_inferior']:+.6f};{row['IC95_superior']:+.6f}].")
    lines += ["", "INTERPRETAÇÃO E LIMITES", "Nenhuma melhora é inferida de execuções pendentes ou de um único fold/semente favorável.",
              "Uma contribuição Wavelet híbrida exige comparação com wav2vec isolado e com o controle de capacidade, não apenas com Wavelet isolada.",
              "IC contendo zero não sustenta uma diferença positiva consistente no respectivo contraste condicional.",
              "Os ICs condicionais não refazem seleção/treinamento nem corrigem as múltiplas explorações históricas da AVFAD.",
              "Esta mesma coorte já foi explorada: resultados representam desenvolvimento, não teste cego independente.",
              "411 pessoas e apenas 82 controles limitam a precisão; seis frases de uma pessoa não são seis pacientes.",
              "A normalização RMS pode remover informação de intensidade; a sequência com 128 passos pode omitir transientes.",
              "Arquiteturas/representações têm capacidades distintas; comparações não isolam perfeitamente cada mecanismo.",
              "Não houve treinamento/validação na SVD nesta campanha, adaptação do encoder XLS-R ou comprovação clínica.",
              "Nenhum percentual deste ciclo pode ser comparado diretamente a outro artigo com coorte, unidade ou protocolo diferentes.",
              "Resultados negativos devem ser preservados; nenhum resultado implica diagnóstico automático confiável.", "", "RASTREABILIDADE"]
    interpretation = []
    for row in contrasts:
        if row["escopo"] != "estabilidade_secundaria":
            continue
        lower, upper = row["IC95_inferior"], row["IC95_superior"]
        if lower <= 0 <= upper:
            conclusion = "IC inclui zero: não foi demonstrado ganho consistente neste contraste condicional."
        elif lower > 0:
            conclusion = "IC acima de zero: diferença positiva neste contraste condicional; exige confirmação independente."
        else:
            conclusion = "IC abaixo de zero: diferença negativa neste contraste condicional."
        interpretation.append(f"Exp. {row['experimento']} {row['comparacao']}: {conclusion}")
    if interpretation:
        position = lines.index("INTERPRETAÇÃO E LIMITES") + 1
        lines[position:position] = interpretation + [""]
    for experiment, state in completed.items():
        if state:
            lines += [f"Exp. {experiment}: {state['folder']}/ARTEFATOS_COMPLETOS.json", f"SHA256 marcador: {state['marker_sha256']}"]
    for name, digest in audits.items():
        lines.append(f"Laudo: {name}; SHA256={digest}")
    lines += ["Cópias de código somente para experimentos completos: codigo_congelado_23_26.",
              "FONTES_CONGELADAS distingue fontes assinadas pela execução de testes/auditores auxiliares.",
              "TABELA_METRICAS e TABELA_CONTRASTES preservam experimento, braço, escopo e arquivo de origem.",
              f"Gerador: src/{Path(__file__).name}; SHA256={sha(__file__)}", ""]
    documentation.mkdir(parents=True, exist_ok=True)
    outputs = {"RESUMO_TECNICO_23_26.txt": "\n".join(lines),
               "TABELA_METRICAS_23_26.csv": pd.DataFrame(rows).to_csv(index=False, lineterminator="\n", float_format="%.17g"),
               "TABELA_CONTRASTES_23_26.csv": pd.DataFrame(contrasts, columns=["experimento", "escopo", "comparacao", "seeds", "delta_BA", "IC95_inferior", "IC95_superior", "fonte"]).to_csv(index=False, lineterminator="\n", float_format="%.17g"),
               "FONTES_CONGELADAS_23_26.csv": pd.DataFrame(frozen).to_csv(index=False, lineterminator="\n")}
    for name, value in outputs.items():
        (documentation/name).write_text(value, encoding="utf8")
    manifest = {"generated_at": now, "generator_sha256": sha(__file__), "experiments": {
                str(key): ({k:v for k,v in state.items() if k != "identity"} if state else {"status":"PENDENTE"})
                for key,state in completed.items()}, "historical21_marker_sha256": history["marker_sha256"] if history else None,
                "audit_reports_sha256": audits, "generated_files": {name:sha(documentation/name) for name in outputs}}
    previous.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    print("Documentacao23–26 gerada; estados:", {key:"COMPLETO" if value else "PENDENTE" for key,value in completed.items()}, flush=True)
    print(documentation / "RESUMO_TECNICO_23_26.txt", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documentacao", type=Path, default=ROOT/DOC_REL)
    args = parser.parse_args()
    generate(args.documentacao.resolve())


if __name__ == "__main__":
    main()
