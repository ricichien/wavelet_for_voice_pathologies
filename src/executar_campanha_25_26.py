# -*- coding: utf-8 -*-
"""Fila autorizada: aguarda 25, executa 26 e audita/documenta, sem lançar 25.

--verificar somente lê e exibe a configuração. --executar mantém uma trava
exclusiva do SO em arquivo persistente. Retomadas nunca ajustam um 26 completo.
Não apaga checkpoints, não instala pacotes e não encerra processos de treino.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/experiments/campanha_25_26_execucao"
DOC = ROOT / "results/experiments/documentacao_wav2vec"
EXP25 = ROOT / "results/experiments/resultados_25_wavelet_cnn_tcn"
EXP26 = ROOT / "results/experiments/resultados_26_hibrido_wavelet_temporal_wav2vec"
SEEDS = [20261105, 101, 102, 103, 104, 105]
ARMS25 = ["A_CNN_RAW", "B_CNN_RMS", "C_TCN_RAW", "D_TCN_RMS", "E_MLP_RESUMOS_RMS"]
ARMS26 = ["A_WAV_MLP32_ONLY", "B_WAV_MLP64_BUDGET", "C_CNN_RMS_WAV", "D_TCN_RMS_WAV"]
SOURCE_NAMES = (
    "13_otimizar_wpt_sym5.py", "17_robustez_distribuicao_wpt_avfad.py",
    "19_avaliar_wpt_16khz_avfad.py", "21_avaliar_wpt_wav2vec_avfad.py",
    "23_extrair_wpt_rms_sequencias_avfad.py", "24_avaliar_normalizacao_wpt_avfad.py",
    "25_treinar_wavelet_cnn_tcn_avfad.py", "26_hibrido_wavelet_temporal_wav2vec.py",
    "verificar_extracao23.py", "verificar_normalizacao24.py", "verificar_wpt_wav2vec22.py",
    "verificar_redes25_26.py", "gerar_documentacao_23_26.py", "executar_campanha_25_26.py",
)


def now():
    return datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


@contextmanager
def exclusive_lock(path):
    """O arquivo fica preservado; a trava de um byte dura até a saída/queda."""
    path = Path(path)
    try:
        stream = path.open("x+b")
    except FileExistsError:
        stream = path.open("r+b")
    with stream:
        if path.stat().st_size == 0:
            stream.write(b"L"); stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Outra fila já mantém a trava; nenhum processo foi iniciado") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def pid_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5  # acesso negado: conservar, sem duplicar
        try:
            code = wintypes.DWORD()
            require(kernel.GetExitCodeProcess(handle, ctypes.byref(code)), "Não foi possível consultar PID anterior")
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def check_file_map(mapping):
    for name, digest in mapping.items():
        require(Path(name).is_file() and sha(name) == digest, f"Fonte/insumo mudou: {name}")


def verify_identity(folder):
    identity = read_json(folder / "assinatura_execucao.json")
    require("code" in identity, f"Assinatura sem fontes: {folder}")
    for category in ("code", "inputs", "input_files"):
        check_file_map(identity.get(category, {}))
    protocol = folder / "PROTOCOLO.txt"
    digest = hashlib.sha256(protocol.read_text(encoding="utf8").encode("utf8")).hexdigest()
    require(digest == identity["protocol_sha256"], f"Protocolo alterado: {folder}")
    return identity


def verify_complete(folder, expected_arms):
    marker = folder / "ARTEFATOS_COMPLETOS.json"
    require(marker.is_file(), f"Experimento incompleto: {folder}")
    exports = read_json(marker)
    require(isinstance(exports, dict) and exports, f"Marcador vazio/inválido: {marker}")
    required = {"assinatura_execucao.json", "PROTOCOLO.txt", "particoes_participantes.csv",
                "predicoes_oof_participante.csv", "metricas_por_seed.csv", "comparacoes_pareadas.csv"}
    require(required.issubset(exports), f"Exports finais ausentes: {folder}")
    for name, digest in exports.items():
        require(Path(name).name == name and name not in (".", ".."), "Caminho inseguro no marcador")
        require(sha(folder / name) == digest, f"Export alterado: {folder / name}")
    identity = verify_identity(folder)
    config = identity["configuration"]
    require(config["seeds"] == SEEDS and config["arms"] == expected_arms, "Ciclo não é o protocolo completo autorizado")
    return sha(marker)


def configuration(wait_hours, poll_seconds):
    gpu_python = ROOT / ".venv_wav2vec/Scripts/python.exe"
    cpu_python = ROOT / ".venv/Scripts/python.exe"
    sources = [ROOT / "src" / name for name in SOURCE_NAMES]
    protected = sources + [gpu_python, cpu_python, EXP25 / "assinatura_execucao.json",
                           EXP25 / "PROTOCOLO.txt", EXP26 / "PROTOCOLO.txt",
                           ROOT / "results/experiments/documentacao_tg/03_INTEGRIDADE_ANTERIORES.json"]
    for path in protected:
        require(path.is_file(), f"Pré-requisito ainda indisponível: {path}")
    identity25 = verify_identity(EXP25)
    require(identity25["configuration"]["seeds"] == SEEDS and identity25["configuration"]["arms"] == ARMS25,
            "A execução25 ativa não corresponde ao ciclo completo esperado")
    commands = {
        "verificar26": [str(gpu_python), "-B", "src/26_hibrido_wavelet_temporal_wav2vec.py", "--verificar"],
        "executar26": [str(gpu_python), "-B", "src/26_hibrido_wavelet_temporal_wav2vec.py", "--executar"],
        "auditar25_26": [str(gpu_python), "-B", "src/verificar_redes25_26.py", "--experimento", "ambos", "--device", "cuda"],
        "documentar": [str(cpu_python), "-B", "src/gerar_documentacao_23_26.py"],
    }
    return {"version": "1.0", "root": str(ROOT), "wait_hours_25": wait_hours, "poll_seconds": poll_seconds,
            "protected_sha256": {str(path): sha(path) for path in protected}, "commands": commands,
            "never_launches25": True, "already_complete26_skips_training": True}


class Queue:
    def __init__(self, folder, config):
        self.folder, self.config = Path(folder), config
        self.signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf8")).hexdigest()
        self.attempt = now().replace(":", "-") + f"_pid{os.getpid()}"
        self.started = now()

    def status(self, phase, **details):
        value = {"phase": phase, "queue_pid": os.getpid(), "attempt": self.attempt,
                 "attempt_started_at": self.started, "updated_at": now(),
                 "queue_signature_sha256": self.signature, "child_pid": None, **details}
        atomic_json(self.folder / "STATUS.json", value)
        with (self.folder / "EVENTOS.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        print(f"[{value['updated_at']}] {phase}: {json.dumps(details, ensure_ascii=False)}", flush=True)

    def protected(self):
        check_file_map(self.config["protected_sha256"])

    def stage(self, name, outputs=()):
        self.protected()
        marker = self.folder / f"{name}.CONCLUIDO.json"
        command = self.config["commands"][name]
        if marker.is_file():
            receipt = read_json(marker)
            require(receipt["queue_signature_sha256"] == self.signature and receipt["command"] == command,
                    f"Etapa anterior de outra configuração: {name}")
            check_file_map(receipt["outputs_sha256"])
            self.status(name + "_confirmado", message="Nenhum subprocesso iniciado; etapa anterior íntegra")
            return
        log = self.folder / f"{self.attempt}_{name}.log.txt"
        self.status(name, command=command, log_file=str(log))
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        with log.open("x", encoding="utf8") as stream:
            stream.write("Comando estruturado: " + json.dumps(command, ensure_ascii=False) + "\n"); stream.flush()
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            while child.poll() is None:
                self.status(name, child_pid=child.pid, command=command, log_file=str(log))
                time.sleep(self.config["poll_seconds"])
            returncode = child.wait()
        self.status(name + "_encerrado", returncode=returncode, log_file=str(log))
        require(returncode == 0, f"Etapa {name} falhou com código {returncode}; consultar {log}")
        self.protected()
        mapping = {}
        for path in outputs:
            require(path.is_file(), f"Etapa {name} terminou sem saída esperada: {path}")
            mapping[str(path)] = sha(path)
        atomic_json(marker, {"queue_signature_sha256": self.signature, "command": command,
                             "finished_at": now(), "returncode": returncode, "outputs_sha256": mapping,
                             "log_file": str(log), "log_sha256": sha(log)})

    def wait25(self):
        deadline = time.monotonic() + self.config["wait_hours_25"] * 3600
        while not (EXP25 / "ARTEFATOS_COMPLETOS.json").is_file():
            self.protected()
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Limite de espera por25 atingido; execução e arquivos25 preservados")
            self.status("aguardando25", seconds_remaining=round(remaining, 1), poll_seconds=self.config["poll_seconds"])
            time.sleep(min(self.config["poll_seconds"], remaining))
        self.status("verificando25_completo")
        self.protected()
        marker_sha = verify_complete(EXP25, ARMS25)
        self.status("25_confirmado", marker_sha256=marker_sha)

    def final_integrity(self):
        manifest_path = ROOT / "results/experiments/documentacao_tg/03_INTEGRIDADE_ANTERIORES.json"
        manifest = read_json(manifest_path)
        failures = []
        for name, expected in manifest["sha256_files"].items():
            path = ROOT / name
            if not path.is_file():
                failures.append({"path": name, "reason": "ausente"}); continue
            actual = {"size_bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns, "sha256": sha(path)}
            if any(actual[key] != expected[key] for key in actual):
                failures.append({"path": name, "expected": expected, "actual": actual})
        for expected in manifest["original_zip_metadata_only"]:
            path = ROOT / expected["path"]
            actual = {"size_bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns} if path.is_file() else {}
            if any(actual.get(key) != expected[key] for key in ("size_bytes", "modified_ns")):
                failures.append({"path": expected["path"], "expected": expected, "actual": actual})
        value = {"checked_at": now(), "protected_files_sha256_size_mtime": len(manifest["sha256_files"]),
                 "zip_size_mtime_only": len(manifest["original_zip_metadata_only"]), "failures": failures,
                 "manifest_sha256": sha(manifest_path)}
        target = DOC / "23_26_INTEGRIDADE_FINAL.json"
        if target.is_file():
            previous = read_json(target)
            for key in value:
                if key != "checked_at":
                    require(previous.get(key) == value[key], "Relatório final de integridade existente diverge; preservar")
        else:
            with target.open("x", encoding="utf8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n")
        require(not failures, f"Falhas de integridade anteriores: {len(failures)}; consultar {target}")
        return target

    def run(self):
        self.wait25()
        self.stage("verificar26")
        if (EXP26 / "ARTEFATOS_COMPLETOS.json").is_file():
            marker_sha = verify_complete(EXP26, ARMS26)
            self.status("26_completo_preservado", marker_sha256=marker_sha, message="Nenhum ajuste de26 iniciado")
        else:
            self.stage("executar26", (EXP26 / "ARTEFATOS_COMPLETOS.json",))
            verify_complete(EXP26, ARMS26)
        audit_txt = DOC / "25_26_VERIFICACAO_REDES.txt"
        audit_json = audit_txt.with_suffix(".json")
        self.stage("auditar25_26", (audit_txt, audit_json))
        audited = read_json(audit_json)
        require({item["experiment"] for item in audited["experiments"]} == {25, 26}, "Auditor não confirmou ambos experimentos")
        require(audited["auditor_sha256"] == sha(ROOT / "src/verificar_redes25_26.py"), "Laudo pertence a outro auditor")
        docs = DOC / "MANIFESTO_DOCUMENTACAO_23_26.json"
        self.stage("documentar", (docs,))
        document = read_json(docs)
        require(document["generator_sha256"] == sha(ROOT / "src/gerar_documentacao_23_26.py"), "Gerador do manifesto diverge")
        for number, folder in ((25, EXP25), (26, EXP26)):
            require(document["experiments"][str(number)]["marker_sha256"] == sha(folder / "ARTEFATOS_COMPLETOS.json"), "Documentação não inclui resultado completo")
        for name, digest in document["generated_files"].items():
            require(sha(DOC / name) == digest, f"Documento final alterado: {name}")
        final = self.final_integrity()
        self.protected()
        self.status("completo", marker25_sha256=sha(EXP25 / "ARTEFATOS_COMPLETOS.json"),
                    marker26_sha256=sha(EXP26 / "ARTEFATOS_COMPLETOS.json"),
                    audit_report=str(audit_txt), documentation_manifest=str(docs), integrity_report=str(final))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verificar", action="store_true")
    action.add_argument("--executar", action="store_true")
    parser.add_argument("--espera-horas", type=float, default=4.)
    parser.add_argument("--poll-segundos", type=float, default=30.)
    args = parser.parse_args()
    require(0 < args.espera_horas <= 24 and 1 <= args.poll_segundos <= 60, "Espera/poll fora dos limites")
    config = configuration(args.espera_horas, args.poll_segundos)
    if args.verificar:
        if (EXP25 / "ARTEFATOS_COMPLETOS.json").is_file():
            verify_complete(EXP25, ARMS25)
        if (EXP26 / "ARTEFATOS_COMPLETOS.json").is_file():
            verify_complete(EXP26, ARMS26)
        print(json.dumps({"verified_at": now(), "readonly": True, "configuration": config}, ensure_ascii=False, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(OUT / "FILA.lock"):
        previous_path = OUT / "STATUS.json"
        previous = read_json(previous_path) if previous_path.is_file() else {}
        require(not pid_alive(previous.get("child_pid")), "Subprocesso anterior ainda ativo; não iniciar uma duplicata")
        signature = OUT / "assinatura_fila.json"
        if signature.is_file():
            require(read_json(signature) == config, "Fontes/configuração da fila mudaram; pasta existente preservada")
        else:
            with signature.open("x", encoding="utf8") as stream:
                json.dump(config, stream, ensure_ascii=False, indent=2); stream.write("\n")
        queue = Queue(OUT, config)
        queue.status("inicio", previous_phase=previous.get("phase"))
        try:
            queue.run()
        except BaseException as exc:
            prior = read_json(previous_path)
            queue.status("erro", child_pid=prior.get("child_pid"), error_type=type(exc).__name__,
                         error=str(exc), traceback=traceback.format_exc(), prior_phase=prior.get("phase"))
            raise


if __name__ == "__main__":
    main()
