# -*- coding: utf-8 -*-
"""Fila autorizada29/30: vogal/a/ e avaliação por participante.

Preserva25–28, usa fontes assinadas e nunca inicia modelos antigos. Só produz
síntese final depois do ciclo completo e da auditoria independente.
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
SPEC = importlib.util.spec_from_file_location("queue25_for_queue29", ROOT / "src/executar_campanha_25_26.py")
BASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASE)
OUT = ROOT / "results/experiments/campanha_29_30_execucao"
DOC = ROOT / "results/experiments/documentacao_wav2vec"
PREVIOUS = ROOT / "results/experiments/campanha_27_28_execucao"
DATA27 = ROOT / "results/experiments/resultados_27_camadas_ssl_extracao"
DATA28 = ROOT / "results/experiments/resultados_28_camadas_ssl_wavelet"
DATA29 = ROOT / "results/experiments/resultados_29_vogal_wavelet_ssl_extracao"
PILOT29 = ROOT / "results/experiments/resultados_29_vogal_wavelet_ssl_piloto_3"
DATA30 = ROOT / "results/experiments/resultados_30_frase_vogal_wavelet"
SEEDS = [20261105, 101, 102, 103, 104, 105]
ARMS = ["A_WPT_SENT128", "B_WPT_VOW128", "C_WPT_SENT_VOW256", "D_SSL_SENT128", "E_SSL_VOW128",
        "F_SSL_SENT_VOW256", "G_HYB_SENT_VOW512", "H_SSL_SENT_VOW512", "I_HYB_SENT256"]


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
        "verificar29": [str(gpu), "-B", "src/29_extrair_vogal_wavelet_ssl_avfad.py", "--verificar"],
        "piloto29": [str(gpu), "-B", "src/29_extrair_vogal_wavelet_ssl_avfad.py", "--piloto", "3"],
        "extrair29": [str(gpu), "-B", "src/29_extrair_vogal_wavelet_ssl_avfad.py", "--executar"],
        "verificar30": [str(cpu), "-B", "src/30_avaliar_frase_vogal_wavelet_avfad.py", "--verificar"],
        "avaliar30": [str(cpu), "-B", "src/30_avaliar_frase_vogal_wavelet_avfad.py", "--executar"],
        "auditar30": [str(cpu), "-B", "src/verificar_frase_vogal30.py"],
    }
    sources = [Path(__file__), ROOT / "src/executar_campanha_25_26.py",
               ROOT / "src/29_extrair_vogal_wavelet_ssl_avfad.py", ROOT / "src/30_avaliar_frase_vogal_wavelet_avfad.py",
               ROOT / "src/verificar_frase_vogal30.py", gpu, cpu,
               PREVIOUS / "assinatura_fila.json", DOC / "29_30_PLANO_FRASE_VOGAL.txt"]
    for name in ("13_otimizar_wpt_sym5.py", "17_robustez_distribuicao_wpt_avfad.py", "18_extrair_wpt_16khz_avfad.py",
                 "19_avaliar_wpt_16khz_avfad.py", "20_extrair_wav2vec_avfad.py", "21_avaliar_wpt_wav2vec_avfad.py",
                 "22_regularizar_wpt_wav2vec_avfad.py", "23_extrair_wpt_rms_sequencias_avfad.py",
                 "27_extrair_camadas_ssl_avfad.py", "28_avaliar_camadas_ssl_wavelet_avfad.py"):
        sources.append(ROOT / "src" / name)
    for path in sources:
        BASE.require(path.is_file(), f"Pré-requisito não finalizado: {path}")
    return {"version": "1.0", "root": str(ROOT), "wait_hours": wait_hours, "poll_seconds": poll,
            "protected_sha256": {str(p.resolve()): BASE.sha(p) for p in sources}, "commands": commands,
            "never_launches_previous_training": True, "input_unit30": "participant",
            "previous_queue_signature_sha256": BASE.sha(PREVIOUS / "assinatura_fila.json")}


class Continuation(BASE.Queue):
    def wait_previous(self):
        deadline = time.monotonic() + self.config["wait_hours"] * 3600
        while True:
            self.protected()
            state = BASE.read_json(PREVIOUS / "STATUS.json")
            if state["phase"] == "completo":
                break
            BASE.require(state["phase"] != "erro", "Fila27/28 registrou erro; preservar execução anterior")
            remaining = deadline - time.monotonic()
            BASE.require(remaining > 0, "Espera por27/28 esgotada; nenhum novo processo iniciado")
            self.status("aguardando27_28", previous_phase=state["phase"], seconds_remaining=round(remaining, 1))
            time.sleep(min(self.config["poll_seconds"], remaining))
        for number, folder in ((27, DATA27), (28, DATA28)):
            BASE.require(exports(folder) == state[f"marker{number}_sha256"], "Resultado anterior não corresponde ao estado final")
        audit = BASE.read_json(DOC / "28_VERIFICACAO_INDEPENDENTE.json")
        BASE.require(audit.get("status") == "aprovado" and audit.get("marker28_sha256") == BASE.sha(DATA28 / "ARTEFATOS_COMPLETOS.json"),
                     "Auditoria28 não aprovada ou divergente")
        self.status("27_28_concluidos_e_auditados")

    def extraction_stage(self, name, folder):
        """Uma extração já finalizada é conferida sem reler o áudio/modelo."""
        if (folder / "ARTEFATOS_COMPLETOS.json").is_file():
            self.protected()
            marker = exports(folder)
            ident = BASE.read_json(folder / "assinatura_execucao.json")
            BASE.require(bool(ident.get("code")), "Extração concluída sem fontes assinadas")
            BASE.check_file_map(ident["code"])
            self.status(name + "_preservado", marker_sha256=marker)
        else:
            self.stage(name, (folder / "ARTEFATOS_COMPLETOS.json",))
            exports(folder)

    def source_snapshot(self):
        target = DOC / "codigo_congelado_29_30"
        target.mkdir(parents=True, exist_ok=True)
        manifest = {}
        for name, digest in self.config["protected_sha256"].items():
            source = Path(name)
            if source.suffix != ".py":
                continue
            destination = target / source.name
            if destination.exists():
                BASE.require(BASE.sha(destination) == digest, "Cópia congelada diferente; preservar")
            else:
                with destination.open("xb") as stream:
                    stream.write(source.read_bytes())
            manifest[source.name] = {"original": name, "sha256": digest}
        path = target / "FONTES.json"
        if path.exists():
            BASE.require(BASE.read_json(path) == manifest, "Manifesto de fontes diferente; preservar")
        else:
            with path.open("x", encoding="utf8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
        return path

    def summary(self):
        identity = BASE.read_json(DATA30 / "assinatura_execucao.json")
        BASE.require(bool(identity.get("code")), "Assinatura30 sem fontes")
        BASE.check_file_map(identity["code"])
        with (DATA30 / "metricas_por_seed.csv").open(encoding="utf8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        groups = {}
        for row in rows:
            key = (int(row["seed"]), row["arm"])
            BASE.require(key not in groups and math.isfinite(float(row["BA"])), "Métrica duplicada/inválida")
            groups[key] = row
        BASE.require({row["arm"] for row in rows} == set(ARMS) and len(groups) == 54, "Ciclo30 incompleto")
        lines = ["EXPERIMENTOS 29–30 — FRASES E VOGAL /a/", f"Consolidado após auditoria: {BASE.now()}",
                 "411 participantes HSA: 82 normais/329 patológicos. Nenhuma exclusão ou relabeling.",
                 "Treinamento e avaliação por participante: frases são a média dos vetores das suas 5/6 gravações.",
                 "Vogal: arquivo002. Todas as tarefas da pessoa permanecem juntas em cada fold.",
                 "A unidade de treinamento difere de28; não atribuir diferenças históricas apenas à nova tarefa.",
                 "Encoder congelado; C e camada selecionados somente nos folds internos.", "",
                 "BA POR PARTICIPANTE — PRINCIPAL / MÉDIA ± DP DAS CINCO SEMENTES DE ESTABILIDADE"]
        for arm in ARMS:
            BASE.require(all((seed, arm) in groups for seed in SEEDS), f"Sementes incompletas: {arm}")
            principal = float(groups[(20261105, arm)]["BA"])
            secondary = [float(groups[(seed, arm)]["BA"]) for seed in SEEDS[1:]]
            lines.append(f"{arm}: {principal:.6f} / {statistics.mean(secondary):.6f} ± {statistics.stdev(secondary):.6f}")
        lines += ["", "CONTRASTES PAREADOS PREDEFINIDOS"]
        with (DATA30 / "comparacoes_pareadas.csv").open(encoding="utf8", newline="") as stream:
            contrasts = list(csv.DictReader(stream))
        for row in contrasts:
            lines.append(f"{row['escopo']} {row['comparacao']}: deltaBA={float(row['delta_BA']):+.6f}; "
                         f"IC95=[{float(row['IC95_inferior']):+.6f};{float(row['IC95_superior']):+.6f}]")
        lines += ["", "LEITURA E LIMITES", "C-A avalia o ganho da vogal no modelo Wavelet; G-F avalia seu incremento ao encoder multitarefa.",
                  "G-H controla a dimensão, sem igualar representações ou toda a capacidade dos pipelines.",
                  "Principal e estabilidade são reportadas separadamente. DP entre sementes não é IC.",
                  "Bootstrap condicional por pessoa não refaz treino nem corrige as múltiplas explorações históricas.",
                  "O número de pessoas independentes continua411. Não usar acurácia bruta como sinônimo de BA.",
                  "Mais áudio e arquiteturas não asseguram80–90%. Preservar todos os resultados negativos.",
                  "SVD não foi usada, encoder não foi adaptado e artigo não foi editado nesta campanha.",
                  "Resultados detalhados e laudo30 devem acompanhar qualquer conclusão do TCC.", ""]
        target = DOC / "RESUMO_TECNICO_29_30.txt"
        payload = "\n".join(lines)
        receipt = self.folder / "DOCUMENTACAO_FINAL.json"
        if target.exists():
            unchanged = receipt.exists() and BASE.read_json(receipt).get("summary_sha256") == BASE.sha(target)
            BASE.require(target.read_text(encoding="utf8") == payload or unchanged, "Resumo existente diverge; preservar")
        target.write_text(payload, encoding="utf8")
        return target

    def run(self):
        self.wait_previous()
        snapshot = self.source_snapshot()
        self.stage("verificar29")
        self.extraction_stage("piloto29", PILOT29)
        self.extraction_stage("extrair29", DATA29)
        marker29 = exports(DATA29)
        self.stage("verificar30")
        self.stage("avaliar30", (DATA30 / "ARTEFATOS_COMPLETOS.json",))
        marker30 = exports(DATA30)
        audit_txt = DOC / "30_VERIFICACAO_INDEPENDENTE.txt"
        audit_json = audit_txt.with_suffix(".json")
        self.stage("auditar30", (audit_txt, audit_json))
        audit = BASE.read_json(audit_json)
        BASE.require(audit.get("status") == "aprovado" and audit.get("n_folds") == 30
                     and audit.get("n_arms") == 9 and audit.get("n_outer") == 270, "Laudo30 não confirma o ciclo completo")
        BASE.require(audit.get("marker30_sha256") == marker30
                     and audit.get("auditor_sha256") == BASE.sha(ROOT / "src/verificar_frase_vogal30.py"), "Laudo30 de outra execução/fonte")
        self.protected()
        summary = self.summary()
        final = {"finished_at": BASE.now(), "queue_signature_sha256": self.signature,
                 "marker29_sha256": marker29, "marker30_sha256": marker30, "summary_sha256": BASE.sha(summary),
                 "audit_sha256": BASE.sha(audit_json), "source_snapshot_manifest_sha256": BASE.sha(snapshot)}
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
        previous = BASE.read_json(state_file) if state_file.exists() else {}
        BASE.require(not BASE.pid_alive(previous.get("child_pid")), "Subprocesso anterior ativo; não duplicar")
        signature = OUT / "assinatura_fila.json"
        if signature.exists():
            BASE.require(BASE.read_json(signature) == config, "Assinatura mudou; preservar pasta")
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
            queue.status("erro", child_pid=prior.get("child_pid"), prior_phase=prior.get("phase"),
                         error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
