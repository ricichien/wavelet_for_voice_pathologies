# -*- coding: utf-8 -*-
"""Continuação autorizada: aguarda 25/26, extrai27 e avalia/audita28.

Não inicia nem altera a fila anterior. Comandos e fontes são vinculados por
hash antes da espera. Usa a trava e os recibos já testados do executor25/26.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("queue25_for_queue27", ROOT / "src/executar_campanha_25_26.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
OUT = ROOT / "results/experiments/campanha_27_28_execucao"
DOC = ROOT / "results/experiments/documentacao_wav2vec"
PREVIOUS = ROOT / "results/experiments/campanha_25_26_execucao"
DATA27 = ROOT / "results/experiments/resultados_27_camadas_ssl_extracao"
PILOT27 = ROOT / "results/experiments/resultados_27_camadas_ssl_piloto_3"
DATA28 = ROOT / "results/experiments/resultados_28_camadas_ssl_wavelet"
SEEDS = [20261105, 101, 102, 103, 104, 105]


def exports(folder):
    marker = folder / "ARTEFATOS_COMPLETOS.json"
    values = BASE.read_json(marker)
    BASE.require(isinstance(values, dict) and bool(values), f"Marcador inválido: {marker}")
    for name, digest in values.items():
        BASE.require(Path(name).name == name and name not in (".", ".."), "Caminho inválido no marcador")
        BASE.require(BASE.sha(folder / name) == digest, f"Export alterado: {folder / name}")
    return BASE.sha(marker)


def configuration(wait_hours, poll):
    gpu = ROOT / ".venv_wav2vec/Scripts/python.exe"
    cpu = ROOT / ".venv/Scripts/python.exe"
    commands = {
        "verificar27": [str(gpu), "-B", "src/27_extrair_camadas_ssl_avfad.py", "--verificar"],
        "piloto27": [str(gpu), "-B", "src/27_extrair_camadas_ssl_avfad.py", "--piloto", "3"],
        "extrair27": [str(gpu), "-B", "src/27_extrair_camadas_ssl_avfad.py", "--executar"],
        "verificar28": [str(cpu), "-B", "src/28_avaliar_camadas_ssl_wavelet_avfad.py", "--verificar"],
        "avaliar28": [str(cpu), "-B", "src/28_avaliar_camadas_ssl_wavelet_avfad.py", "--executar"],
        "auditar28": [str(cpu), "-B", "src/verificar_camadas_ssl28.py"],
    }
    names = [Path(__file__), ROOT / "src/executar_campanha_25_26.py",
             ROOT / "src/27_extrair_camadas_ssl_avfad.py", ROOT / "src/28_avaliar_camadas_ssl_wavelet_avfad.py",
             ROOT / "src/verificar_camadas_ssl28.py", gpu, cpu,
             PREVIOUS / "assinatura_fila.json", DOC / "27_30_DECISOES_PROXIMA_ETAPA.txt"]
    for name in ("13_otimizar_wpt_sym5.py", "17_robustez_distribuicao_wpt_avfad.py",
                 "18_extrair_wpt_16khz_avfad.py", "19_avaliar_wpt_16khz_avfad.py",
                 "20_extrair_wav2vec_avfad.py", "21_avaliar_wpt_wav2vec_avfad.py",
                 "22_regularizar_wpt_wav2vec_avfad.py"):
        names.append(ROOT / "src" / name)
    for path in names:
        BASE.require(path.is_file(), f"Pré-requisito não finalizado: {path}")
    return {"version": "1.0", "root": str(ROOT), "wait_hours": wait_hours,
            "poll_seconds": poll, "protected_sha256": {str(p.resolve()): BASE.sha(p) for p in names},
            "commands": commands, "never_launches25_or26": True,
            "previous_queue_signature_sha256": BASE.sha(PREVIOUS / "assinatura_fila.json")}


class Continuation(BASE.Queue):
    def wait_previous(self):
        deadline = time.monotonic() + self.config["wait_hours"] * 3600
        while True:
            self.protected()
            state = BASE.read_json(PREVIOUS / "STATUS.json")
            if state["phase"] == "completo":
                break
            BASE.require(state["phase"] != "erro", "Fila25/26 registrou erro; execução anterior preservada")
            remaining = deadline - time.monotonic()
            BASE.require(remaining > 0, "Espera por25/26 esgotada; nenhum novo processo GPU iniciado")
            self.status("aguardando25_26", previous_phase=state["phase"], seconds_remaining=round(remaining, 1))
            time.sleep(min(self.config["poll_seconds"], remaining))
        BASE.verify_complete(BASE.EXP25, BASE.ARMS25)
        BASE.verify_complete(BASE.EXP26, BASE.ARMS26)
        audit = BASE.read_json(DOC / "25_26_VERIFICACAO_REDES.json")
        BASE.require({row["experiment"] for row in audit["experiments"]} == {25, 26}, "Auditoria anterior incompleta")
        integrity = BASE.read_json(DOC / "23_26_INTEGRIDADE_FINAL.json")
        BASE.require(not integrity["failures"], "Integridade anterior não aprovada")
        self.status("25_26_concluidos_e_auditados")

    def summary(self):
        identity = BASE.read_json(DATA28 / "assinatura_execucao.json")
        source_map = identity.get("code", {})
        BASE.require(bool(source_map), "Assinatura28 sem fontes")
        BASE.check_file_map(source_map)
        with (DATA28 / "metricas_por_seed.csv").open(encoding="utf8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        groups = {}
        for row in rows:
            key = (int(row["seed"]), row["arm"])
            BASE.require(key not in groups and math.isfinite(float(row["BA"])), "Métrica duplicada/inválida")
            groups[key] = row
        arms = sorted({row["arm"] for row in rows})
        BASE.require(len(arms) == 9 and len(groups) == 54, "Ciclo28 não tem nove braços/seis sementes")
        lines = ["EXPERIMENTOS27–28 — CAMADAS SSL E CONTRIBUIÇÃO WAVELET",
                 f"Consolidado após auditoria: {BASE.now()}",
                 "411 pessoas HSA, 82 normais/329 patológicos; avaliação principal por participante.",
                 "Camadas5/12/16 fixadas antes dos ajustes; camada24 do cache20 como referência.",
                 "Semente20261105 principal; estabilidade101–105. Não escolher a melhor divisão.", "",
                 "BA POR PARTICIPANTE — PRINCIPAL / MÉDIA E DP DE ESTABILIDADE"]
        for arm in arms:
            BASE.require(all((seed, arm) in groups for seed in SEEDS), f"Sementes incompletas: {arm}")
            principal = float(groups[(20261105, arm)]["BA"])
            secondary = [float(groups[(seed, arm)]["BA"]) for seed in SEEDS[1:]]
            lines.append(f"{arm}: {principal:.6f} / {statistics.mean(secondary):.6f} ± {statistics.stdev(secondary):.6f}")
        lines += ["", "CONTRASTES PAREADOS PRÉ-DEFINIDOS"]
        with (DATA28 / "comparacoes_pareadas.csv").open(encoding="utf8", newline="") as stream:
            contrasts = list(csv.DictReader(stream))
        for row in contrasts:
            lines.append(f"{row['escopo']} {row['comparacao']}: deltaBA={float(row['delta_BA']):+.6f}; "
                         f"IC95=[{float(row['IC95_inferior']):+.6f};{float(row['IC95_superior']):+.6f}]")
        lines += ["", "LEITURA E LIMITES",
                  "Camada/C são selecionados exclusivamente dentro do treino externo. Todos os resultados são preservados.",
                  "H-G avalia o incremento Wavelet; ganhar apenas sobre Wavelet isolada não demonstra esse incremento.",
                  "G-F compara busca interna de camada à última camada com a mesma busca de C.",
                  "Examinar os ICs e os controles de dimensão, além dos valores médios. DP não é IC.",
                  "80–90% de BA é uma aspiração; nenhum ganho é presumido. Acurácia bruta não substitui BA.",
                  "AVFAD historicamente explorada: desenvolvimento, não teste cego; ICs condicionais não corrigem todas as tentativas.",
                  "27/28 mantêm somente frases: vogais foram inventariadas, não incluídas automaticamente nestas tabelas.",
                  "Não houve fine-tuning do encoder, treinamento SVD ou edição do artigo neste ciclo.",
                  "Detalhes completos: resultados_28_camadas_ssl_wavelet/RELATORIO_RESULTADOS.txt e laudo28.", ""]
        target = DOC / "RESUMO_TECNICO_27_28.txt"
        payload = "\n".join(lines)
        if target.exists():
            BASE.require(target.read_text(encoding="utf8") == payload or self.previous_summary_matches(target),
                         "Resumo existente não corresponde à fila; preservar")
        target.write_text(payload, encoding="utf8")
        return target

    def previous_summary_matches(self, target):
        receipt = self.folder / "DOCUMENTACAO_FINAL.json"
        return receipt.is_file() and BASE.read_json(receipt).get("summary_sha256") == BASE.sha(target)

    def run(self):
        self.wait_previous()
        self.stage("verificar27")
        self.stage("piloto27", (PILOT27 / "ARTEFATOS_COMPLETOS.json",))
        exports(PILOT27)
        self.stage("extrair27", (DATA27 / "ARTEFATOS_COMPLETOS.json",))
        marker27 = exports(DATA27)
        self.stage("verificar28")
        self.stage("avaliar28", (DATA28 / "ARTEFATOS_COMPLETOS.json",))
        marker28 = exports(DATA28)
        audit_txt = DOC / "28_VERIFICACAO_INDEPENDENTE.txt"
        audit_json = audit_txt.with_suffix(".json")
        self.stage("auditar28", (audit_txt, audit_json))
        audit = BASE.read_json(audit_json)
        BASE.require(audit.get("status") == "aprovado" and audit.get("n_folds") == 30
                     and audit.get("n_arms") == 9 and audit.get("n_outer") == 270,
                     "Laudo28 não confirma o ciclo completo aprovado")
        BASE.require(audit.get("marker28_sha256") == marker28
                     and audit.get("auditor_sha256") == BASE.sha(ROOT / "src/verificar_camadas_ssl28.py"),
                     "Laudo28 não corresponde aos resultados/fontes desta fila")
        self.protected()
        summary = self.summary()
        final = {"finished_at": BASE.now(), "queue_signature_sha256": self.signature,
                 "marker27_sha256": marker27, "marker28_sha256": marker28,
                 "summary_sha256": BASE.sha(summary), "audit_sha256": BASE.sha(audit_json)}
        BASE.atomic_json(self.folder / "DOCUMENTACAO_FINAL.json", final)
        self.status("completo", summary=str(summary), audit_report=str(audit_txt), **final)


def main():
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verificar", action="store_true")
    action.add_argument("--executar", action="store_true")
    parser.add_argument("--espera-horas", type=float, default=6.)
    parser.add_argument("--poll-segundos", type=float, default=30.)
    args = parser.parse_args()
    BASE.require(0 < args.espera_horas <= 24 and 1 <= args.poll_segundos <= 60, "Espera/poll fora dos limites")
    config = configuration(args.espera_horas, args.poll_segundos)
    if args.verificar:
        print(json.dumps({"readonly": True, "verified_at": BASE.now(), "configuration": config}, ensure_ascii=False, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with BASE.exclusive_lock(OUT / "FILA.lock"):
        state_file = OUT / "STATUS.json"
        previous = BASE.read_json(state_file) if state_file.is_file() else {}
        BASE.require(not BASE.pid_alive(previous.get("child_pid")), "Subprocesso anterior ativo; não duplicar")
        signature = OUT / "assinatura_fila.json"
        if signature.exists():
            BASE.require(BASE.read_json(signature) == config, "Assinatura da continuação mudou; preservar pasta")
        else:
            with signature.open("x", encoding="utf8") as stream:
                json.dump(config, stream, ensure_ascii=False, indent=2)
        queue = Continuation(OUT, config)
        queue.status("inicio", previous_phase=previous.get("phase"))
        try:
            queue.run()
        except BaseException as exc:
            import traceback
            prior = BASE.read_json(state_file)
            queue.status("erro", error_type=type(exc).__name__, error=str(exc),
                         child_pid=prior.get("child_pid"), prior_phase=prior.get("phase"), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
