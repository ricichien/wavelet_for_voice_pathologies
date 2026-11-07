# -*- coding: utf-8 -*-
"""Testes da fila com arquivos sintéticos; não inicia experimentos de voz."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("queue25_tests", HERE / "executar_campanha_25_26.py")
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class CampaignQueueTests(unittest.TestCase):
    def test_lock_is_exclusive_and_persistent(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "persistent.lock"
            with QUEUE.exclusive_lock(path):
                with self.assertRaises(RuntimeError):
                    with QUEUE.exclusive_lock(path):
                        self.fail("Uma segunda trava não deveria ser adquirida")
            self.assertTrue(path.is_file())
            with QUEUE.exclusive_lock(path):
                self.assertTrue(path.is_file())

    def test_stage_receipt_prevents_repeating_completed_child(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            output = folder / "synthetic_output.txt"
            # Escrever uma vez; uma repetição do subprocesso falharia.
            code = f"from pathlib import Path; Path({str(output)!r}).open('x').write('synthetic only')"
            config = {"protected_sha256": {}, "poll_seconds": .01,
                      "commands": {"test": [sys.executable, "-B", "-c", code]}}
            queue = QUEUE.Queue(folder, config)
            queue.stage("test", (output,))
            first = output.stat().st_mtime_ns
            queue.stage("test", (output,))
            self.assertEqual(output.stat().st_mtime_ns, first)
            output.write_text("tampered", encoding="utf8")
            with self.assertRaises(RuntimeError):
                queue.stage("test", (output,))

    def test_complete_marker_binds_exports_and_source(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "synthetic_source.py"
            source.write_text("# no model", encoding="utf8")
            protocol = "Synthetic protocol\n"
            (folder / "PROTOCOLO.txt").write_text(protocol, encoding="utf8")
            identity = {"code": {str(source): QUEUE.sha(source)}, "inputs": {},
                        "configuration": {"seeds": QUEUE.SEEDS, "arms": ["synthetic"]},
                        "protocol_sha256": hashlib.sha256(protocol.encode()).hexdigest()}
            (folder / "assinatura_execucao.json").write_text(json.dumps(identity), encoding="utf8")
            names = ["assinatura_execucao.json", "PROTOCOLO.txt", "particoes_participantes.csv",
                     "predicoes_oof_participante.csv", "metricas_por_seed.csv", "comparacoes_pareadas.csv"]
            for name in names[2:]:
                (folder / name).write_text("synthetic only\n", encoding="utf8")
            marker = folder / "ARTEFATOS_COMPLETOS.json"
            marker.write_text(json.dumps({name: QUEUE.sha(folder / name) for name in names}), encoding="utf8")
            self.assertEqual(QUEUE.verify_complete(folder, ["synthetic"]), QUEUE.sha(marker))
            source.write_text("# changed", encoding="utf8")
            with self.assertRaises(RuntimeError):
                QUEUE.verify_complete(folder, ["synthetic"])

    def test_wait_is_bounded_and_never_launches25(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            queue = QUEUE.Queue(folder, {"protected_sha256": {}, "poll_seconds": 30., "wait_hours_25": 0.})
            with patch.object(QUEUE, "EXP25", folder), patch.object(queue, "status"), patch.object(QUEUE.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(RuntimeError, "Limite de espera"):
                    queue.wait25()
                popen.assert_not_called()

    def test_current_pid_is_recognized(self):
        self.assertTrue(QUEUE.pid_alive(os.getpid()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
