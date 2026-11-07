# -*- coding: utf-8 -*-
"""Checks sintéticos da continuação; sem áudio, modelos ou processos filhos."""
import csv
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("queue27_tests", HERE / "executar_campanha_27_28.py")
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


class ContinuationTests(unittest.TestCase):
    def queue(self, folder):
        config = {"protected_sha256": {}, "commands": {}, "poll_seconds": 1, "wait_hours": .001}
        return E.Continuation(folder, config)

    def write(self, path, value):
        path.write_text(json.dumps(value), encoding="utf8")

    def test_previous_error_refuses_to_start_new_work(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder / "STATUS.json", {"phase": "erro"})
            queue = self.queue(folder)
            with patch.object(E, "PREVIOUS", folder), patch.object(queue, "stage") as stage:
                with self.assertRaisesRegex(RuntimeError, "registrou erro"):
                    queue.wait_previous()
                stage.assert_not_called()

    def test_completed_previous_requires_both_results_and_integrity(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder / "STATUS.json", {"phase": "completo"})
            self.write(folder / "25_26_VERIFICACAO_REDES.json", {"experiments": [{"experiment": 25}, {"experiment": 26}]})
            self.write(folder / "23_26_INTEGRIDADE_FINAL.json", {"failures": []})
            queue = self.queue(folder)
            with patch.object(E, "PREVIOUS", folder), patch.object(E, "DOC", folder), \
                 patch.object(E.BASE, "verify_complete") as verified:
                queue.wait_previous()
                self.assertEqual(verified.call_count, 2)
                self.assertEqual(json.loads((folder / "STATUS.json").read_text())["phase"], "25_26_concluidos_e_auditados")
                self.write(folder / "STATUS.json", {"phase": "completo"})
                self.write(folder / "23_26_INTEGRIDADE_FINAL.json", {"failures": ["arquivo"]})
                with self.assertRaisesRegex(RuntimeError, "Integridade"):
                    queue.wait_previous()

    def test_pending_or_stale_audit_cannot_be_reported_as_complete(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder / "28_VERIFICACAO_INDEPENDENTE.json", {"status": "pendente"})
            queue = self.queue(folder)
            with patch.object(E, "DOC", folder), patch.object(queue, "wait_previous"), \
                 patch.object(queue, "stage"), patch.object(E, "exports", return_value="signed"), \
                 patch.object(queue, "summary") as summary:
                with self.assertRaisesRegex(RuntimeError, "ciclo completo"):
                    queue.run()
                summary.assert_not_called()

    def test_summary_requires_all_nine_models_and_six_seeds(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write(folder / "assinatura_execucao.json", {"code": {str(HERE / "executar_campanha_27_28.py"): E.BASE.sha(HERE / "executar_campanha_27_28.py")}})
            rows = [{"seed": seed, "arm": f"MODEL{i}", "BA": .5 + i * .01} for seed in E.SEEDS for i in range(9)]
            with (folder / "metricas_por_seed.csv").open("w", newline="", encoding="utf8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["seed", "arm", "BA"])
                writer.writeheader(); writer.writerows(rows)
            (folder / "comparacoes_pareadas.csv").write_text("escopo,comparacao,delta_BA,IC95_inferior,IC95_superior\n", encoding="utf8")
            queue = self.queue(folder)
            with patch.object(E, "DATA28", folder), patch.object(E, "DOC", folder):
                result = queue.summary()
                self.assertIn("MODEL8", result.read_text(encoding="utf8"))
                with (folder / "metricas_por_seed.csv").open("a", newline="", encoding="utf8") as stream:
                    csv.DictWriter(stream, fieldnames=["seed", "arm", "BA"]).writerow(rows[0])
                with self.assertRaisesRegex(RuntimeError, "duplicada"):
                    queue.summary()


if __name__ == "__main__":
    unittest.main(verbosity=2)
