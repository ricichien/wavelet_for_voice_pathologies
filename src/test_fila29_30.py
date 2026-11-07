# -*- coding: utf-8 -*-
"""Testes da orquestração com artefatos artificiais, sem áudio/processos."""
import csv
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("queue29_tests", HERE / "executar_campanha_29_30.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


class QueueTests(unittest.TestCase):
    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf8")

    def queue(self, folder, protected=None):
        return E.Continuation(folder, {"protected_sha256": protected or {}, "commands": {}, "poll_seconds": 1, "wait_hours": .001})

    def test_existing_extraction_is_verified_without_starting_child(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = HERE / "executar_campanha_29_30.py"
            self.write(folder / "assinatura_execucao.json", {"code": {str(source): E.BASE.sha(source)}})
            self.write(folder / "ARTEFATOS_COMPLETOS.json", {"assinatura_execucao.json": E.BASE.sha(folder / "assinatura_execucao.json")})
            queue = self.queue(folder)
            with patch.object(queue, "stage") as stage:
                queue.extraction_stage("extrair29", folder)
                stage.assert_not_called()
                (folder / "assinatura_execucao.json").write_text("{}", encoding="utf8")
                with self.assertRaises(RuntimeError):
                    queue.extraction_stage("extrair29", folder)

    def test_previous_failure_does_not_start_new_extraction(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder / "STATUS.json", {"phase": "erro"})
            queue = self.queue(folder)
            with patch.object(E, "PREVIOUS", folder), patch.object(queue, "stage") as stage:
                with self.assertRaisesRegex(RuntimeError, "registrou erro"):
                    queue.wait_previous()
                stage.assert_not_called()

    def test_unapproved_audit_cannot_create_summary(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder / "30_VERIFICACAO_INDEPENDENTE.json", {"status": "pendente"})
            queue = self.queue(folder)
            with patch.object(E, "DOC", folder), patch.object(queue, "wait_previous"), \
                 patch.object(queue, "source_snapshot"), patch.object(queue, "stage"), \
                 patch.object(queue, "extraction_stage"), patch.object(E, "exports", return_value="signed"), \
                 patch.object(queue, "summary") as summary:
                with self.assertRaisesRegex(RuntimeError, "ciclo completo"):
                    queue.run()
                summary.assert_not_called()

    def test_frozen_source_is_not_overwritten_when_changed(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "synthetic.py"
            source.write_text("x = 1\n", encoding="utf8")
            queue = self.queue(folder, {str(source): E.BASE.sha(source)})
            with patch.object(E, "DOC", folder):
                queue.source_snapshot()
                frozen = folder / "codigo_congelado_29_30/synthetic.py"
                frozen.write_text("changed\n", encoding="utf8")
                with self.assertRaisesRegex(RuntimeError, "congelada diferente"):
                    queue.source_snapshot()
                self.assertEqual(frozen.read_text(encoding="utf8"), "changed\n")
                self.assertEqual(source.read_text(encoding="utf8"), "x = 1\n")

    def test_summary_requires_exact_arms_and_all_six_seeds(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = HERE / "executar_campanha_29_30.py"
            self.write(folder / "assinatura_execucao.json", {"code": {str(source): E.BASE.sha(source)}})
            rows = [{"seed": seed, "arm": arm, "BA": .6} for seed in E.SEEDS for arm in E.ARMS]
            with (folder / "metricas_por_seed.csv").open("w", encoding="utf8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["seed", "arm", "BA"])
                writer.writeheader(); writer.writerows(rows)
            (folder / "comparacoes_pareadas.csv").write_text("escopo,comparacao,delta_BA,IC95_inferior,IC95_superior\n", encoding="utf8")
            queue = self.queue(folder)
            with patch.object(E, "DOC", folder), patch.object(E, "DATA30", folder):
                path = queue.summary()
                self.assertIn("Treinamento e avaliação por participante", path.read_text(encoding="utf8"))
                with (folder / "metricas_por_seed.csv").open("a", encoding="utf8", newline="") as stream:
                    csv.DictWriter(stream, fieldnames=["seed", "arm", "BA"]).writerow(rows[0])
                with self.assertRaisesRegex(RuntimeError, "duplicada"):
                    queue.summary()


if __name__ == "__main__":
    unittest.main(verbosity=2)
