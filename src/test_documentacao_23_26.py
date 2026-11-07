# -*- coding: utf-8 -*-
"""Testes de integridade da documentacao; nenhum ajuste de modelos."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

source = Path(__file__).with_name("gerar_documentacao_23_26.py")
spec = importlib.util.spec_from_file_location("doc23_26_tested", source)
doc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doc)


class DocumentationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def complete_fixture(self):
        code = self.folder / "code.py"
        code.write_text("# frozen synthetic source\n", encoding="utf8")
        identity = self.folder / "assinatura_execucao.json"
        identity.write_text(json.dumps({"code": {str(code): doc.sha(code)}}), encoding="utf8")
        artifact = self.folder / "metricas_por_seed.csv"
        artifact.write_text("synthetic complete results\n", encoding="utf8")
        marker = self.folder / "ARTEFATOS_COMPLETOS.json"
        marker.write_text(json.dumps({identity.name: doc.sha(identity), artifact.name: doc.sha(artifact)}), encoding="utf8")
        return code, artifact

    def test_partial_results_are_not_read(self):
        (self.folder/"metricas_por_seed.csv").write_bytes(b"not even a valid csv\xff")
        self.assertIsNone(doc.complete_exports(self.folder))

    def test_export_tampering_is_rejected(self):
        _, artifact = self.complete_fixture()
        artifact.write_text("tampered\n", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Artefato completo alterado"):
            doc.complete_exports(self.folder)

    def test_signed_source_tampering_is_rejected(self):
        code, _ = self.complete_fixture()
        code.write_text("# changed\n", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Fonte diverge"):
            doc.complete_exports(self.folder)

    def test_stability_does_not_include_primary_seed(self):
        frame = pd.DataFrame([{"seed": seed, "arm": "SYNTHETIC", "BA": value,
                               "sensibilidade": .5, "especificidade": .5,
                               "acuracia": .5, "AUC_media_folds_ponderada_pares": .5}
                              for seed,value in ((20261105,.99),(101,.5),(102,.6),(103,.7),(104,.6),(105,.6))])
        result = doc.summary_rows(25, frame, ["SYNTHETIC"], "synthetic source")[0]
        self.assertEqual(result["BA_principal_20261105"], .99)
        self.assertAlmostEqual(result["BA_estabilidade_media"], .6)
        self.assertEqual(result["n_sementes_estabilidade"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
