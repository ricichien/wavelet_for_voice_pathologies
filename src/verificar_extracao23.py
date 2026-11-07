# -*- coding: utf-8 -*-
"""Auditoria23 posterior: somente leitura; sem WAV, extracao ou treinamento.

Sequencias NPZ sao lidas em fluxo por gravacao para limitar uso de RAM.
O unico arquivo criado e o relatorio, apos todas as verificacoes passarem.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import time
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp23_audit", HERE / "23_extrair_wpt_rms_sequencias_avfad.py")
EXP23 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXP23)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class NpyStream:
    def __init__(self, archive, member):
        self.stream = archive.open(member, "r")
        version = np.lib.format.read_magic(self.stream)
        if version == (1, 0):
            self.shape, order, self.dtype = np.lib.format.read_array_header_1_0(self.stream)
        elif version == (2, 0):
            self.shape, order, self.dtype = np.lib.format.read_array_header_2_0(self.stream)
        else:
            raise ValueError(f"Formato NPY nao previsto: {version}")
        require(not order and self.dtype == np.dtype("float32"), "Sequencia precisa ser C-contigua float32")
        require(len(self.shape) == 2 and self.shape[1] == 480, "Dimensao temporal invalida")
        self.rows = 0

    def read(self, count):
        nbytes = count * self.shape[1] * self.dtype.itemsize
        chunks, missing = [], nbytes
        while missing:
            part = self.stream.read(missing)
            require(bool(part), "NPY temporal truncado")
            chunks.append(part); missing -= len(part)
        self.rows += count
        return np.frombuffer(b"".join(chunks), dtype=self.dtype).reshape(count, self.shape[1])

    def finish(self):
        require(self.rows == self.shape[0], "NPY nao cobriu todas as janelas")
        require(self.stream.read(1) == b"", "NPY contem dados excedentes")
        self.stream.close()


def compare_mapping(expected, observed):
    for key, value in expected.items():
        actual = observed[key]
        if value is None:
            require(pd.isna(actual), f"Campo deveria ser ausente: {key}")
        elif isinstance(value, (float, int)):
            require(np.isclose(value, actual, rtol=1e-14, atol=1e-14), f"Campo numerico divergente: {key}")
        else:
            require(str(value) == str(actual), f"Campo textual divergente: {key}")


def compare_metadata18(meta18, meta23):
    """VAD igual; tolera apenas arredondamento da razao calculada em outra ordem.

    Exp18: (Nativo/48000)/(Ntotal/48000). Exp23: Nativo/Ntotal.
    Duas operacoes float64 adicionais podem alterar a ultima casa binaria.
    IDs, contagens, flags e duracoes continuam exigindo igualdade exata.
    """
    exact = ("sample_id", "speech_regions", "window_count", "active_duration_s",
             "recording_duration_s", "vad_fallback")
    for key in exact:
        np.testing.assert_array_equal(meta18[key], meta23[key], err_msg=f"Metadata18/23: {key}")
    old = meta18["speech_ratio"].to_numpy(dtype=np.float64)
    new = meta23["speech_ratio"].to_numpy(dtype=np.float64)
    require(np.isfinite(old).all() and np.isfinite(new).all(), "speech_ratio nao finito")
    # A razao pertence a [0,1]; tolerancia absoluta de dois eps de float64.
    atol = float(2 * np.finfo(np.float64).eps)
    np.testing.assert_allclose(old, new, rtol=0, atol=atol,
                               err_msg="speech_ratio18/23 excede arredondamento float64")
    difference = np.abs(old - new)
    return {"speech_ratio_different": int(np.count_nonzero(difference)),
            "speech_ratio_max_abs_error": float(difference.max(initial=0)),
            "speech_ratio_atol": atol, "speech_ratio_rtol": 0,
            "exact_fields": list(exact)}


def audit(out, report):
    started = time.perf_counter()
    marker_file = out / "ARTEFATOS_COMPLETOS.json"
    require(marker_file.is_file(), "Extracao23 ainda nao terminou; nao auditar artefatos parciais")
    require(not report.exists(), "Relatorio existente preservado; escolher outro caminho explicitamente")
    marker = json.loads(marker_file.read_text(encoding="utf8"))
    require(set(marker) == set(EXP23.EXPORTS), "Marcador23 incompleto")
    for name, digest in marker.items():
        require(EXP23.sha(out / name) == digest, f"Exportacao alterada: {name}")
    signature_file = out / "assinatura_execucao.json"
    identity = json.loads(signature_file.read_text(encoding="utf8"))
    require(identity["mode"] == "completo", "Esta auditoria exige coorte completa")
    for source, digest in identity["code"].items():
        require(EXP23.sha(source) == digest, f"Fonte alterada: {source}")
    for source, info in identity["inputs"].items():
        require(Path(source).stat().st_size == info["size_bytes"] and EXP23.sha(source) == info["sha256"], "Insumo alterado")
    for source, info in identity["raw_zip_metadata_only"].items():
        stat = Path(source).stat()
        require(stat.st_size == info["size_bytes"] and stat.st_mtime_ns == info["mtime_ns"], "ZIP mudou tamanho/mtime")
    configuration = identity["configuration"]
    require(configuration["sample_rate"] == 48000 and configuration["target_rms"] == .1
            and configuration["normalization"] == "global_active_rms"
            and configuration["vad_before_normalization"] is True, "Configuracao23 inesperada")

    original, reference, names, base_names, _, _ = EXP23.EXP18.load_inputs(ROOT)
    meta = pd.read_csv(out / "metadata_wpt_rms.csv", float_precision="round_trip",
                       dtype={"sample_id": str, "participant_id": str, "task": str})
    quality = pd.read_csv(out / "QUALIDADE_AUDIO.csv", float_precision="round_trip",
                          dtype={"sample_id": str, "participant_id": str, "task": str})
    controls = pd.read_csv(out / "CONTROLE_REFERENCIA_48KHZ.csv", float_precision="round_trip")
    for frame in (meta, quality):
        require(len(frame) == 2465 and not frame.sample_id.duplicated().any(), "Manifesto incompleto/duplicado")
        frame["task"] = frame.task.str.zfill(3)
        for key in ("sample_id", "participant_id", "task"):
            np.testing.assert_array_equal(frame[key], original[key])
    np.testing.assert_array_equal(meta.y, original.y)
    np.testing.assert_array_equal(meta.matrix_row, np.arange(2465))
    require(identity["sample_ids"] == meta.sample_id.tolist(), "Ordem da assinatura divergente")
    require(len(controls) == 2465 and controls.equivalent.eq(True).all(), "Controles48k incompletos")
    np.testing.assert_array_equal(controls.sample_id, meta.sample_id)
    people = meta.drop_duplicates("participant_id")
    require(len(people) == 411 and people.y.value_counts().to_dict() == {1:329, 0:82}, "Coorte incorreta")

    with np.load(out / "features_wpt_rms.npz", allow_pickle=False) as data:
        matrix = data["wavelet"]
        require(matrix.dtype == np.float32 and matrix.shape == (2465, 1920) and np.isfinite(matrix).all(), "MatrizRMS invalida")
        np.testing.assert_array_equal(data["feature_names"], names)
        for key in ("sample_id", "participant_id", "task"):
            np.testing.assert_array_equal(data[key].astype(str), meta[key])
    with np.load(out / "sequencias_wpt.npz", allow_pickle=False) as data:
        offsets = data["offsets"]
        require(offsets.dtype == np.int64 and offsets.shape == (2466,) and offsets[0] == 0
                and np.all(np.diff(offsets) > 0), "Offsets invalidos")
        np.testing.assert_array_equal(data["feature_names"], base_names)
        for key in ("sample_id", "participant_id", "task"):
            np.testing.assert_array_equal(data[key].astype(str), meta[key])
    np.testing.assert_array_equal(np.diff(offsets), meta.window_count)
    connection = sqlite3.connect(f"file:{(out/'checkpoint_wpt_rms_sequencias.sqlite').as_posix()}?mode=ro", uri=True)
    totals = {"n_recordings": 0, "vector_blob_hashes": 0, "max_raw_reference_error": 0.,
              "max_rms_target_error": 0., "windows": int(offsets[-1]), "pilot_max_abs_error": 0.}
    pilot = ROOT / "results/experiments/resultados_23_wpt_rms_sequencias_piloto_3"
    pilot_checks = 0
    try:
        require(connection.execute("PRAGMA quick_check").fetchall() == [("ok",)], "SQLite invalido")
        require(connection.execute("SELECT signature FROM config").fetchall() == [(EXP23.sha(signature_file),)], "AssinaturaSQLite divergente")
        require(connection.execute("SELECT count(*) FROM features").fetchone() == (2465,), "SQLite incompleto")
        require(connection.execute("SELECT count(*) FROM errors").fetchone() == (0,), "SQLite contem erros")
        with zipfile.ZipFile(out / "sequencias_wpt.npz", "r") as archive:
            raw_stream, rms_stream = NpyStream(archive, "data_raw.npy"), NpyStream(archive, "data_rms.npy")
            require(raw_stream.shape == rms_stream.shape == (int(offsets[-1]), 480), "ShapesNPZ invalidos")
            for index, row in enumerate(meta.itertuples()):
                record = connection.execute("SELECT * FROM features WHERE sample_id=?", (row.sample_id,)).fetchone()
                sid, pid, task, raw, rms, vector, diagnostics, measured_quality, control = EXP23.read_row(record)
                require(sid == row.sample_id and pid == row.participant_id and task == row.task, "IdentidadeSQLite incorreta")
                count = int(offsets[index+1] - offsets[index])
                np.testing.assert_array_equal(raw_stream.read(count), raw)
                np.testing.assert_array_equal(rms_stream.read(count), rms)
                np.testing.assert_array_equal(EXP23.summarize(rms), vector)
                np.testing.assert_array_equal(vector, matrix[index])
                raw_vector = EXP23.summarize(raw)
                error = float(np.max(np.abs(raw_vector-reference[index])))
                require(np.allclose(raw_vector, reference[index], rtol=2e-5, atol=2e-5), "Raw nao reproduz referencia")
                require(error == control["max_abs_error"] == controls.iloc[index].max_abs_error
                        and control["equivalent"] is True, "Controle48k incorreto")
                compare_mapping(diagnostics, meta.iloc[index])
                compare_mapping(measured_quality, quality.iloc[index])
                rms_error = abs(diagnostics["active_rms_after"] - .1)
                require(rms_error < 1e-7 and abs(diagnostics["active_rms_before"] * diagnostics["rms_global_gain"]-.1) < 1e-12,
                        "Ganho/RMS global inconsistente")
                require(diagnostics["window_count"] == count and len(raw) == len(rms) == count, "Quantidade de janelas mudou")
                totals["n_recordings"] += 1; totals["vector_blob_hashes"] += 3
                totals["max_raw_reference_error"] = max(totals["max_raw_reference_error"], error)
                totals["max_rms_target_error"] = max(totals["max_rms_target_error"], rms_error)
            raw_stream.finish(); rms_stream.finish()
        if pilot.is_dir() and (pilot / "ARTEFATOS_COMPLETOS.json").is_file():
            with np.load(pilot / "sequencias_wpt.npz", allow_pickle=False) as data:
                poffsets, pids = data["offsets"], data["sample_id"].astype(str)
                for key, field in (("data_raw", 3), ("data_rms", 4)):
                    pvalues = data[key]
                    for i, sid in enumerate(pids):
                        record = connection.execute("SELECT * FROM features WHERE sample_id=?", (sid,)).fetchone()
                        actual = EXP23.read_row(record)[field]
                        expected = pvalues[poffsets[i]:poffsets[i+1]]
                        require(actual.shape == expected.shape, "Piloto/execucao mudaram shape")
                        maximum = float(np.max(np.abs(actual-expected)))
                        require(maximum == 0, "Piloto/execucao mudaram sequencia")
                        totals["pilot_max_abs_error"] = max(totals["pilot_max_abs_error"], maximum)
                        pilot_checks += 1
    finally:
        connection.close()
    meta18file = ROOT / "results/experiments/resultados_18_wpt_16khz_extracao/metadata_wpt_sym5_16khz.csv"
    comparison18 = None
    if meta18file.is_file():
        meta18 = pd.read_csv(meta18file, float_precision="round_trip")
        comparison18 = compare_metadata18(meta18, meta)
    counts = {"stereo_recordings": int((quality.channels_original > 1).sum()),
              "over_60s": int(quality.flag_duration_over_60s.sum()),
              "possible_channel_cancellation": int(quality.flag_possible_channel_cancellation.sum()),
              "raw_full_scale_sample_recordings": int(quality.flag_any_full_scale_sample.sum())}
    lines = ["AUDITORIA DE EXTRACAO23 — APROVADA", f"Verificado em:{EXP23.EXP18.timestamp()}",
             f"Fonte auditor:{Path(__file__).resolve()}", f"SHA256 auditor:{EXP23.sha(__file__)}", "",
             "Nenhum WAV foi relido, nenhum sinal reextraido e nenhum classificador foi treinado.",
             "Sequencias NPZ verificadas em fluxo por gravacao, sem materializar a coorte inteira na RAM.",
             "SQLite aberto somente em mode=ro; todos os blobs descomprimidos e hashes conferidos.",
             "Comparacao raw com NPZ48k original; resumoRMS com blob/NPZ; IDs/ordem/rotulos completos.",
             "O RMS efetivo e auditado pelos diagnosticos e identidade do ganho; esta verificacao nao recalcula RMS do WAV.", ""]
    lines += [f"{key}: {value}" for key,value in totals.items()]
    lines += [f"pilot_sequence_checks: {pilot_checks}", "", "QUALIDADE — FLAGS, SEM EXCLUSOES"]
    lines += [f"{key}: {value}" for key,value in counts.items()]
    lines += ["", "COMPARACAO DE METADADOS18/23"]
    if comparison18 is None:
        lines += ["Metadata18 nao encontrada; comparacao nao realizada."]
    else:
        lines += [f"{key}: {value}" for key,value in comparison18.items()]
        lines += ["Somente speech_ratio admite rtol=0 e atol=2*eps(float64).",
                  "Exp18 calcula (Nativo/48000)/(Ntotal/48000); Exp23 calcula Nativo/Ntotal.",
                  "A diferenca decorre da ordem de operacoes float64; nao indica mudanca de VAD.",
                  "IDs, contagens, flags e ambas as duracoes foram comparados exatamente.",
                  "Nenhum dado, export, extrator ou resultado de classificacao foi modificado."]
    lines += ["", "Todos os411participantes/2465gravacoes mantidos. Flags nao demonstram defeito ou etiologia.",
              "Valores de clipping referem amplitude absoluta >=1-1e-6 do WAV decodificado.",
              "VAD/duracoes/contagemjanelas comparados com18 quando sua metadata completa esta disponivel.",
              "Hashes dos sete exports, fontes e insumos confirmados; ZIPs conferidos por tamanho/mtime.",
              "Os conteudos completos dos ZIPs nao foram re-hasheados nesta auditoria.",
              f"Segundos:{time.perf_counter()-started:.3f}", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf8") as stream:
        stream.write("\n".join(lines))
    print("AUDITORIA23 APROVADA", json.dumps(totals, sort_keys=True), flush=True)
    print("Qualidade:", json.dumps(counts, sort_keys=True), flush=True)
    print("Relatorio:", report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", type=Path, default=ROOT / "results/experiments/resultados_23_wpt_rms_sequencias")
    parser.add_argument("--relatorio", type=Path, default=ROOT / "results/experiments/documentacao_wav2vec/23_VERIFICACAO_EXTRACAO.txt")
    args = parser.parse_args()
    out, report = args.experimento.resolve(), args.relatorio.resolve()
    require(out.is_relative_to(ROOT / "results/experiments") and report.is_relative_to(ROOT / "results/experiments"), "Caminho fora do workspace de experimentos")
    audit(out, report)


if __name__ == "__main__":
    main()
