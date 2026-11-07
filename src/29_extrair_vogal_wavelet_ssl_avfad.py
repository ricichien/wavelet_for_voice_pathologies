# -*- coding: utf-8 -*-
"""Exp29: vogal /a/002, WPT RAW/RMS e XLS-R congelado nos mesmos411 HSA.

--verificar só lê metadados/hashes/diretório ZIP; não importa Torch ou lê WAV.
--piloto3/--executar exigem25–28 completos; execução real é decisão do root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import time
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value
EXP27 = module("exp27_for29", "27_extrair_camadas_ssl_avfad.py")
EXP23 = module("exp23_for29", "23_extrair_wpt_rms_sequencias_avfad.py")
EXP20, EXP18 = EXP27.EXP20, EXP27.EXP18
VERSION = "29.1"
LAYERS = (24, 5, 12, 16)
PILOT_IDS = ("AAC002", "LFC002", "ZSQ002")
ARRAY_WIDTHS = {"wavelet_raw": 1920, "wavelet_rms": 1920, **{f"embeddings_layer{x:02d}": 2048 for x in LAYERS}}
TOTAL_DIM = sum(ARRAY_WIDTHS.values())
OUT_REL = Path("results/experiments/resultados_29_vogal_wavelet_ssl_extracao")
PILOT_REL = Path("results/experiments/resultados_29_vogal_wavelet_ssl_piloto_3")
DOC_REL = Path("results/experiments/documentacao_wav2vec")
EXPORTS = ("features_vogal_wavelet_ssl.npz", "metadata_vogal_wavelet_ssl.csv", "QUALIDADE_AUDIO.csv",
           "CONTROLE_PILOTO_SSL24.csv", "assinatura_execucao.json", "PROTOCOLO.txt", "RESUMO_EXTRACAO.txt")
require, sha = EXP27.require, EXP20.sha


def names():
    return {"wavelet_feature_names": EXP18.feature_names()[0],
            **{f"feature_names_layer{x:02d}": EXP20.feature_names() if x == 24 else EXP27.feature_names(x) for x in LAYERS}}


def validate_people(people, headers, validation, real=True):
    fields = ["participant_id", "y", "label", "age", "sex", "visit_place"]
    require(set(fields).issubset(people) and not people[fields].isna().any().any(), "Pessoas29 incompletas")
    require(not people.participant_id.duplicated().any() and not headers.participant_id.duplicated().any(), "Participante29 duplicado")
    require(set(headers.participant_id) == set(people.participant_id), "Headers29 não cobrem as mesmas pessoas")
    require(not validation.participant_id.duplicated().any(), "Validação002 original duplicada")
    h, v = headers.set_index("participant_id"), validation.set_index("participant_id")
    for row in people.itertuples(index=False):
        require(row.participant_id in v.index, "Pessoa ausente da validação002 original")
        a, b = h.loc[row.participant_id], v.loc[row.participant_id]
        for field in headers.columns.drop("participant_id"):
            require(str(a[field]) == str(b[field]), f"Header legado e validação002 discordam em {field}/{row.participant_id}")
        for field in ("label", "age", "sex", "visit_place"):
            require(str(getattr(row, field)).strip() == str(a[field]).strip(), f"Metadados29 divergem em {field}")
        require(int(row.y) == int(a.numeric_code != 0) and bool(a.audio_found), "Rótulo/disponibilidade002 divergente")
        require(int(a.sample_rate_hz) == 48000 and int(a.channels) == 1 and int(a.bits_per_sample) == 16 and int(a.frames) > 0,
                "Contrato legado de WAV002 diferente do auditado")
        path = PurePosixPath(str(a.audio_path))
        require(not path.is_absolute() and ".." not in path.parts and path.name.upper() == row.participant_id + "002.WAV",
                "Caminho002 incompatível com a pessoa")
    if real:
        require((len(people), people.y.value_counts().to_dict()) == (411, {1: 329, 0: 82}), "Coorte29 deve ser411 HSA/82/329")
        require(people.visit_place.str.strip().eq("HSA").all(), "Local29 diferente deHSA")


def load_inputs(root):
    data23 = root / "results/experiments/resultados_23_wpt_rms_sequencias"
    marker23 = json.loads((data23 / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    meta_path = data23 / "metadata_wpt_rms.csv"
    require(marker23.get(meta_path.name) == sha(meta_path), "Metadata23 alterada/incompleta")
    meta = pd.read_csv(meta_path, dtype={"participant_id": str, "task": str}, float_precision="round_trip")
    columns = ["participant_id", "y", "label", "age", "sex", "visit_year", "visit_place", "matched_age_sex", "matched_age_sex_year"]
    people = meta[columns].drop_duplicates().sort_values("participant_id").reset_index(drop=True)
    headers_path = root / DOC_REL / "27_HEADERS_VOGAL_002_LEGADO.csv"
    validation_path = root / "data/processed/validacao_tarefa_002.csv"
    inventory_path = root / DOC_REL / "27_INVENTARIO_TAREFAS_HSA.csv"
    headers = pd.read_csv(headers_path, dtype={"participant_id": str}, float_precision="round_trip")
    validation = pd.read_csv(validation_path, dtype={"participant_id": str}, float_precision="round_trip")
    validate_people(people, headers, validation)
    inventory = pd.read_csv(inventory_path, dtype={"participant_id": str, "task": str, "CRC32": str})
    inventory = inventory[inventory.task.str.zfill(3).eq("002")].set_index("participant_id")
    require(not inventory.index.duplicated().any() and set(inventory.index) == set(people.participant_id), "Inventário002 incompleto/duplicado")
    archives = sorted(root.glob("AVFAD_01_00_00_*.zip"))
    require(len(archives) == 5, "Esperava os cinco ZIPs originais")
    central = {}
    for archive in archives:
        with zipfile.ZipFile(archive, "r") as handle:
            for entry in handle.infolist():
                if entry.filename.lower().endswith("002.wav"):
                    pid = PurePosixPath(entry.filename).name[:-7]
                    if pid in set(people.participant_id):
                        central.setdefault(pid, []).append((archive.name, entry))
    h = headers.set_index("participant_id")
    records = []
    for row in people.itertuples(index=False):
        old = h.loc[row.participant_id]; inv = inventory.loc[row.participant_id]
        entries = central.get(row.participant_id, [])
        require(len(entries) == 1, f"Vogal002 ausente/duplicada no diretório ZIP: {row.participant_id}")
        archive, entry = entries[0]
        require(archive == old.audio_archive == inv.archive and entry.filename == old.audio_path == inv.audio_path,
                "Caminho002 diverge do inventário/header")
        require(entry.file_size == int(inv.uncompressed_bytes) > 0 and entry.compress_size == int(inv.compressed_bytes) and
                f"{entry.CRC:08x}" == str(inv.CRC32).lower().zfill(8), "ZIP central002 mudou desde o inventário")
        record = row._asdict()
        record.update(sample_id=row.participant_id + "002", task="002", zip_name=archive, audio_path_in_zip=entry.filename,
                      expected_sample_rate_hz=int(old.sample_rate_hz), expected_channels=int(old.channels), expected_frames=int(old.frames),
                      expected_bits_per_sample=int(old.bits_per_sample), legacy_duration_s=float(old.duration_s),
                      numeric_code=int(old.numeric_code), clinical_label=str(old.clinical_label), zip_crc32=f"{entry.CRC:08x}",
                      zip_uncompressed_bytes=entry.file_size)
        records.append(record)
    selected = pd.DataFrame(records)
    require(selected.sample_id.iloc[np.linspace(0, len(selected)-1, 3, dtype=int)].tolist() == list(PILOT_IDS), "Piloto29 determinístico mudou")
    paths = [meta_path, data23 / "ARTEFATOS_COMPLETOS.json", headers_path, validation_path, inventory_path,
             root / "data/metadata/AVFAD_01_00_00_1_README/README_01_00_00.txt"]
    return selected, paths, archives


def previous_complete(root):
    folder = root / "results/experiments/campanha_27_28_execucao"
    state = json.loads((folder / "STATUS.json").read_text(encoding="utf8"))
    require(state.get("phase") == "completo", "Fila27–28 ainda não completa; nenhuma inferência29 iniciada")
    for number, name in ((27, "resultados_27_camadas_ssl_extracao"), (28, "resultados_28_camadas_ssl_wavelet")):
        out = root / "results/experiments" / name
        require(sha(out / "ARTEFATOS_COMPLETOS.json") == state.get(f"marker{number}_sha256"), "Marcador anterior diverge da fila")
        verify_exports(out)
    audit = root / DOC_REL / "28_VERIFICACAO_INDEPENDENTE.json"
    value = json.loads(audit.read_text(encoding="utf8"))
    require(sha(audit) == state.get("audit_sha256") and value.get("status") == "aprovado" and value.get("n_outer") == 270,
            "Auditoria28 não aprovada/vinculada")


def verify_exports(out):
    marker = json.loads((out / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    require(bool(marker), "Marcador vazio")
    for name, digest in marker.items():
        require(Path(name).name == name and name not in (".", "..") and sha(out / name) == digest, "Export alterado/caminho inválido")
    return marker


def configuration(device, use_safe):
    result = EXP27.configuration(EXP27.MODEL_REVISION, use_safe, device)
    result.update(task="002", task_definition="sustained_a_all_recorded_repetitions_VAD_not_Praat_segment",
                  hidden_layers=list(LAYERS), feature_dimension=TOTAL_DIM, feature_dimensions=ARRAY_WIDTHS,
                  wavelet_raw_rms=True, wavelet="sym5", wavelet_levels=[5, 6], wavelet_source_rate=48000,
                  wavelet_windows=[.4, .2], wavelet_summaries=["mean", "std", "q10", "q90"],
                  wavelet_RMS_target=.1, control_reference="new_audio_pilot_forward27_vs20_no_sentence_equivalence",
                  control_pilot_ids=list(PILOT_IDS), full_requires_signed_pilot=True)
    return result


def identity(paths, archives, selected, mode, model_dir, files, use_safe, device, pilot_dir=None):
    ident = EXP20.identity(paths, archives, selected, mode, model_dir, EXP27.MODEL_REVISION, files, use_safe, device)
    ident.update(version=VERSION, configuration=configuration(device, use_safe))
    for path in (Path(__file__), Path(EXP27.__file__), Path(EXP23.__file__)):
        ident["code"][str(path.resolve())] = sha(path)
    if pilot_dir is not None:
        ident["pilot_reference"] = {"directory": str(pilot_dir), "marker_sha256": sha(pilot_dir / "ARTEFATOS_COMPLETOS.json")}
        for name in EXPORTS:
            path = pilot_dir / name
            ident["inputs"][str(path.resolve())] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
    return ident


def protocol(ident):
    return f"""EXPERIMENTO29 — VOGAL /a/002, WPT RAW/RMS E XLS-R CONGELADO
Modo: {ident['mode']}. Coorte EXATA411 HSA,82 normais/329 patológicos, um WAV002 por pessoa.
Nova tarefa com todas as emissões retidas pelo VAD; não usar o segmento Praat anotado.
Manifesto: metadata23 + headers002 legados + validação002 original + diretório ZIP.
IDs/rótulos/demografia/caminhos/CRC central conferidos, sem exclusão/relabeling.
Rótulo clínico/código/metadados nunca entram na extração; y apenas acompanha o manifesto.
WAV float32: downmix float64→float32, reamostragem se necessária, remoçãoDC como18/23.
Cabeçalho decodificado deve reproduzir frames/48kHz/mono/PCM16 legados. Flags não são filtros.
WPT RAW e RMS: sym5 48k,L5/L6,96 nós×5 descritores×4 resumos=1920 por gravação;
janelas400ms/avanço200ms, mesmas funções23 e padding/reflexão final legado.
RMS global0.1 de TODA voz ativa, VAD antes do ganho, sem ganho por janela ou clipping.
Não existe vetor WPT legado desta vogal como referência; não alegar equivalência com frases.
SSL: {EXP20.MODEL_ID}, revisão{EXP27.MODEL_REVISION}, camadas24/5/12/16 fixas.
Reader/VAD20,48→16k por região, concatenar regiões/chunks quase iguais<=8s;
preprocessor normaliza chunks, sem overlap/padding/truncamento; encoder congelado/eval,
inference_mode float32/eager/TF32off/determinístico, todos os frames e Welfordfloat64.
Camada24 após LayerNormfinal;5/12/16 antes dela; nenhuma LayerNorm extra.
Cada camada2048=1024médias+1024DPs; API27.infer_vectors e20.prepare_signal sem alterações.
Piloto AAC002/LFC002/ZSQ002: comparar24 da API27 com uma inferência independente
20.infer_vector nos MESMOS novos chunks; nunca contra o cache de frases20.
Controle rtol=atol=2e-5, mesmos frames; registrar erro e exatidão. Repetir nos3IDs do full.
Full exige piloto assinado e compara todos os seus vetores WPT/SSL, tolerância2e-5.
Verificar identidade RMS/ganho/janelas em todos; discrepâncias de invariância são diagnósticos.
NPZ:wavelet_raw/rms411×1920; embeddings_layer24/05/12/16 cada411×2048; nomes e IDs.
Nenhuma classificação, BA, ANOVA, normalização entre pessoas, escolha de camada ou SVD.
SQLite próprio, hashes de vetores/diagnósticos/qualidade; exports somente após todos completos.
--verificar não importaTorch/lêWAV/escreve; execução exige filas anteriores completas/auditadas.
Fontes/dados/resultados anteriores permanecem intocados. AVFAD segue desenvolvimento.
"""


def read_audio(root, row, opened):
    if row.zip_name not in opened:
        opened[row.zip_name] = zipfile.ZipFile(root / row.zip_name, "r")
    raw = opened[row.zip_name].read(row.audio_path_in_zip)
    info = EXP23.sf.info(io.BytesIO(raw))
    require((info.samplerate, info.channels, info.frames, info.subtype) ==
            (row.expected_sample_rate_hz, row.expected_channels, row.expected_frames, "PCM_16"),
            "Cabeçalho decodificado002 diverge do legado; não excluir silenciosamente")
    audio, sr = EXP23.sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    signal, quality = EXP23.process_audio48(audio, sr)
    quality.update(decoded_format=info.format, decoded_subtype=info.subtype, decoded_bits_per_sample=16)
    return signal, quality


def extract_wpt(signal):
    raw, rms, rms_vector, diag, _unused_legacy_control = EXP23.extract_signal48(signal)
    raw_vector = EXP23.summarize(raw)
    gain = diag["rms_global_gain"]
    invariant = np.arange(EXP23.WIDTH) % 5 != 1
    diag.update(raw_rms_invariant_block_max_abs_error=float(np.max(np.abs(raw[:, invariant]-rms[:, invariant]))),
                raw_rms_logenergy_shift_max_abs_error=float(np.max(np.abs(rms[:, 1::5]-raw[:, 1::5]-2*np.log10(gain)))),
                normalized_peak_exceeds_one=int(diag["normalized_peak"] > 1))
    require(abs(diag["active_rms_before"] * gain - .1) < 1e-12 and
            np.isclose(diag["active_rms_after"], .1, atol=1e-8, rtol=1e-6), "Identidade RMS29 inválida")
    return {"wavelet_raw": raw_vector, "wavelet_rms": rms_vector}, diag


def check_control(actual, expected, frames, expected_frames):
    require(actual.shape == expected.shape == (2048,) and np.isfinite(actual).all() and np.isfinite(expected).all(), "ControleSSL24 inválido")
    require(frames == expected_frames and np.allclose(actual, expected, atol=2e-5, rtol=2e-5), "Inferência27/20 da vogal divergiu")
    return {"ssl24_control_done": True, "ssl24_control_exact": bool(np.array_equal(actual, expected)),
            "ssl24_control_max_abs_error": float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))),
            "ssl24_control_frames": int(expected_frames)}


def load_pilot(folder):
    require(set(verify_exports(folder)) == set(EXPORTS), "Piloto29 incompleto")
    ident = json.loads((folder / "assinatura_execucao.json").read_text(encoding="utf8"))
    require(ident.get("mode") == "piloto_3" and ident.get("sample_ids") == list(PILOT_IDS), "Manifesto do piloto29 incorreto")
    for path, digest in ident["code"].items():
        require(sha(path) == digest, "Fonte do piloto29 mudou")
    result = {}
    with np.load(folder / EXPORTS[0], allow_pickle=False) as z:
        require(z["sample_id"].astype(str).tolist() == list(PILOT_IDS), "IDs do piloto29 incorretos")
        for index, sid in enumerate(PILOT_IDS):
            result[sid] = {key: z[key][index].copy() for key in ARRAY_WIDTHS}
    meta = pd.read_csv(folder / EXPORTS[1])
    require(meta.ssl24_control_done.eq(True).all() and meta.ssl24_control_exact.notna().all(), "Controle do piloto29 ausente")
    return result, ident


def compare_pilot(arrays, expected):
    maximum, exact = 0., True
    for key, width in ARRAY_WIDTHS.items():
        a, b = arrays[key], expected[key]
        require(a.shape == b.shape == (width,) and np.allclose(a, b, atol=2e-5, rtol=2e-5), "Full29 diverge do piloto assinado")
        maximum = max(maximum, float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))))
        exact = exact and bool(np.array_equal(a, b))
    return {"pilot_reference_checked": True, "pilot_reference_exact": exact, "pilot_reference_max_abs_error": maximum}


def pack(arrays):
    for key, width in ARRAY_WIDTHS.items():
        require(arrays[key].shape == (width,) and arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all(), f"Vetor29 inválido: {key}")
    return np.concatenate([arrays[key] for key in ARRAY_WIDTHS]).tobytes()


def unpack(blob):
    x = np.frombuffer(blob, dtype=np.float32)
    require(x.shape == (TOTAL_DIM,) and np.isfinite(x).all(), "Blob29 inválido")
    offset, values = 0, {}
    for key, width in ARRAY_WIDTHS.items():
        values[key] = x[offset:offset+width].copy(); offset += width
    return values


def validate_cache(con, selected):
    wanted = selected.set_index("sample_id"); found = {}
    for sid, pid, task, blob, digest, text, text_digest in con.execute("SELECT * FROM features"):
        require(sid in wanted.index and str(wanted.loc[sid, "participant_id"]) == pid and task == "002", "IDsSQLite29 divergentes")
        require(hashlib.sha256(blob).hexdigest() == digest and hashlib.sha256(text.encode("utf8")).hexdigest() == text_digest,
                "Hash29 de vetor/diagnósticos divergente")
        payload = json.loads(text); diag, quality = payload["diagnostics"], payload["quality"]
        require(diag["window_count"] > 0 and diag["hidden_frames"] > 0 and diag["chunk_count"] > 0,
                "Contagens29 inválidas")
        require(np.isclose(diag["active_rms_after"], .1, atol=1e-8, rtol=1e-6) and
                abs(diag["active_rms_before"]*diag["rms_global_gain"]-.1) < 1e-12, "RMS29 inválido")
        require(diag["ssl24_control_done"] == (sid in PILOT_IDS), "Controle29 não corresponde ao piloto predefinido")
        if sid in PILOT_IDS:
            require(diag["ssl24_control_frames"] == diag["hidden_frames"] and np.isfinite(diag["ssl24_control_max_abs_error"]), "ControleSSL29 incompleto")
        require(quality["source_samples"] == int(wanted.loc[sid, "expected_frames"]) and quality["original_sample_rate_hz"] == 48000 and
                quality["channels_original"] == 1 and quality["decoded_bits_per_sample"] == 16, "Qualidade/header29 inconsistente")
        found[sid] = (unpack(blob), payload)
    require(all(sid in wanted.index for sid, in con.execute("SELECT sample_id FROM errors")), "Erro29 fora do manifesto")
    return found


def store(con, row, arrays, payload):
    blob, text = pack(arrays), EXP20.json_text(payload)
    with con:
        con.execute("INSERT INTO features VALUES (?,?,?,?,?,?,?)", (row.sample_id, row.participant_id, row.task, blob,
                    hashlib.sha256(blob).hexdigest(), text, hashlib.sha256(text.encode("utf8")).hexdigest()))
        con.execute("DELETE FROM errors WHERE sample_id=?", (row.sample_id,))


def export(out, selected, found, ident, elapsed, runtime):
    require(set(found) == set(selected.sample_id), "Nenhuma matriz29 parcial será exportada")
    meta = selected.copy().reset_index(drop=True)
    for category in ("diagnostics", "quality"):
        values = pd.DataFrame([found[sid][1][category] for sid in meta.sample_id])
        for key in values:
            if key in meta:
                require(np.array_equal(meta[key], values[key]), f"Coluna29 conflitante: {key}")
            meta[key] = values[key]
    meta["matrix_row"] = np.arange(len(meta))
    payload = {key: meta[key].to_numpy(dtype=str) for key in ("sample_id", "participant_id", "task")}
    payload.update(names()); payload["hidden_layer_indices"] = np.asarray(LAYERS, np.int16)
    payload.update({key: np.stack([found[sid][0][key] for sid in meta.sample_id]) for key in ARRAY_WIDTHS})
    temp = out / (EXPORTS[0]+".part")
    with temp.open("wb") as stream: np.savez_compressed(stream, **payload)
    temp.replace(out / EXPORTS[0]); EXP18.atomic_csv(out / EXPORTS[1], meta)
    quality = pd.DataFrame([{**{key: str(meta.iloc[i][key]) for key in ("sample_id", "participant_id", "task")},
                             **found[sid][1]["quality"]} for i, sid in enumerate(meta.sample_id)])
    EXP18.atomic_csv(out / EXPORTS[2], quality)
    control_columns = ["sample_id", "participant_id", "task", "hidden_frames", "ssl24_control_done", "ssl24_control_exact",
                       "ssl24_control_max_abs_error", "ssl24_control_frames", "pilot_reference_checked", "pilot_reference_exact", "pilot_reference_max_abs_error"]
    EXP18.atomic_csv(out / EXPORTS[3], meta[meta.sample_id.isin(PILOT_IDS)][control_columns])
    lines = ["EXPERIMENTO29 — VOGAL /a/002, WPT RAW/RMS E SSL CONGELADO", f"Concluído: {EXP18.timestamp()}",
             f"Modo: {ident['mode']}; pessoas/gravações: {len(meta)}; camadas: {LAYERS}",
             f"WPT: {payload['wavelet_raw'].shape} RAW/RMS; SSL: {payload['embeddings_layer24'].shape} por camada.",
             f"Janelas WPT: {int(meta.window_count.sum())}; framesSSL: {int(meta.hidden_frames.sum())}; segundos: {elapsed:.3f}",
             f"Controles24 independentes: {int(meta.ssl24_control_done.sum())}; maior erro: {meta.ssl24_control_max_abs_error.max()}",
             f"RMS máximo desvio0.1: {float(np.max(np.abs(meta.active_rms_after-.1)))}",
             f"Flags: fullscale={int(meta.flag_any_full_scale_sample.sum())}; cancelamento={int(meta.flag_possible_channel_cancellation.sum())}; duração>60s={int(meta.flag_duration_over_60s.sum())}",
             f"Runtime: {json.dumps(runtime, ensure_ascii=False, sort_keys=True)}",
             "Nenhuma exclusão/classificação/BA/SVD. Vogal é uma nova tarefa; não equivale às frases20/23.", ""]
    EXP20.atomic_text(out / EXPORTS[-1], "\n".join(lines))
    EXP20.atomic_text(out / "ARTEFATOS_COMPLETOS.json", EXP20.json_text({name: sha(out/name) for name in EXPORTS}))


def run(root, out, selected, ident, model_dir, use_safe, device, pilot=None):
    previous_complete(root)
    require(ident["mode"] == "piloto_3" or pilot is not None, "Full29 exige piloto assinado")
    text = EXP20.json_text(ident); signature = hashlib.sha256(text.encode("utf8")).hexdigest()
    sf = out / "assinatura_execucao.json"
    if sf.exists():
        require(sf.read_text(encoding="utf8") == text, "Assinatura29 mudou; preservar pasta")
    else:
        require(not out.exists() or not any(out.iterdir()), "Pasta29 não vazia sem assinatura")
        out.mkdir(parents=True, exist_ok=True); EXP20.atomic_text(sf, text)
        EXP20.atomic_text(out / "PROTOCOLO.txt", protocol(ident))
    require((out / "PROTOCOLO.txt").read_text(encoding="utf8") == protocol(ident), "Protocolo29 alterado")
    con = EXP27.init_cache(out / "checkpoint_vogal_wavelet_ssl.sqlite", signature)
    opened = {}; started = time.perf_counter()
    try:
        found = validate_cache(con, selected)
        if (out / "ARTEFATOS_COMPLETOS.json").exists():
            require(set(verify_exports(out)) == set(EXPORTS) and set(found) == set(selected.sample_id) and
                    con.execute("SELECT COUNT(*) FROM errors").fetchone()[0] == 0, "Conclusão29 inválida")
            EXP20.log(out, "Retomada29: cache/exports íntegros; nenhum WAV/encoder lido."); return
        torch, processor, model, runtime = EXP20.load_encoder(model_dir, use_safe, device)
        EXP20.log(out, f"Início/retomada29 {ident['mode']}: {len(found)}/{len(selected)}; GPU={runtime['gpu']}")
        for row in selected.itertuples(index=False):
            if row.sample_id in found: continue
            tick = time.perf_counter()
            try:
                signal, quality = read_audio(root, row, opened)
                arrays, diag = extract_wpt(signal)
                chunks, ssl_diag = EXP20.prepare_signal(signal)
                require(ssl_diag["speech_regions"] == diag["speech_regions"] and ssl_diag["vad_fallback"] == diag["vad_fallback"] and
                        abs(ssl_diag["active_duration_s"]-diag["active_duration_s"]) < 1e-12, "VAD20/23 divergiu para a vogal")
                intermediate, final, frames = EXP27.infer_vectors(chunks, torch, processor, model, device)
                arrays["embeddings_layer24"] = final
                for index, layer in enumerate(EXP27.LAYERS): arrays[f"embeddings_layer{layer:02d}"] = intermediate[index]
                diag.update(hidden_frames=frames, chunk_count=ssl_diag["chunk_count"], ssl_sample_rate_hz=16000,
                            ssl_target_active_samples=ssl_diag["target_active_samples"],
                            ssl24_control_done=False, ssl24_control_exact=None, ssl24_control_max_abs_error=None, ssl24_control_frames=None,
                            pilot_reference_checked=False, pilot_reference_exact=None, pilot_reference_max_abs_error=None)
                if row.sample_id in PILOT_IDS:
                    expected, expected_frames = EXP20.infer_vector(chunks, torch, processor, model, device)
                    diag.update(check_control(final, expected, frames, expected_frames))
                    if pilot is not None: diag.update(compare_pilot(arrays, pilot[row.sample_id]))
                diag["extraction_seconds"] = time.perf_counter()-tick
                payload = {"diagnostics": diag, "quality": quality}
                store(con, row, arrays, payload); found[row.sample_id] = (arrays, payload)
            except Exception as exc:
                with con: con.execute("INSERT OR REPLACE INTO errors VALUES (?,?)", (row.sample_id, repr(exc)))
                EXP20.log(out, f"ERRO29 {row.sample_id}: {exc!r}; nenhum export parcial"); raise
            if ident["mode"] != "completo" or len(found)%25 == 0 or len(found) == len(selected):
                EXP20.log(out, f"{len(found)}/{len(selected)}; {row.sample_id}; segundos={diag['extraction_seconds']:.3f}")
        require(con.execute("SELECT COUNT(*) FROM errors").fetchone()[0] == 0, "Cache29 ainda contém erros")
        export(out, selected, validate_cache(con, selected), ident, time.perf_counter()-started, runtime)
        EXP20.log(out, "Extração29 concluída sem exclusões; WPT/SSL com IDs e controles.")
    finally:
        con.close()
        for archive in opened.values(): archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT); parser.add_argument("--saida", type=Path)
    parser.add_argument("--modelo", type=Path); parser.add_argument("--referencia-piloto", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    action = parser.add_mutually_exclusive_group(); action.add_argument("--verificar", action="store_true")
    action.add_argument("--autoteste", action="store_true"); action.add_argument("--piloto", type=int, nargs="?", const=3)
    action.add_argument("--executar", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        import unittest
        tests = module("test_vogal_wavelet_ssl29", "test_vogal_wavelet_ssl29.py")
        if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful(): raise SystemExit(1)
        return
    require(args.piloto is None or args.piloto == 3, "Protocolo29 exige piloto de três gravações")
    root = args.dados.resolve(); selected, paths, archives = load_inputs(root)
    is_pilot = args.piloto is not None
    if is_pilot: selected = selected[selected.sample_id.isin(PILOT_IDS)].copy()
    mode = "piloto_3" if is_pilot else "completo"
    model_dir = (args.modelo or root / "models/wav2vec2-xls-r-300m" / EXP27.MODEL_REVISION).resolve()
    files, use_safe = EXP20.model_files(model_dir, EXP27.MODEL_REVISION)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf8"))
    require(config.get("do_stable_layer_norm") is True, "Arquitetura29 deve usarStableLayerNorm")
    pilot_dir = (args.referencia_piloto or root / PILOT_REL).resolve()
    pilot, pilot_ident = (None, None)
    if args.executar: pilot, pilot_ident = load_pilot(pilot_dir)
    ident = identity(paths, archives, selected, mode, model_dir, files, use_safe, args.device, pilot_dir if args.executar else None)
    print(f"ENTRADAS29 OK: mesmos411 HSA; selecionadas={len(selected)}; vogal002; WPT RAW/RMS+SSL{LAYERS}.", flush=True)
    if not (args.executar or is_pilot):
        print("Somente verificação: nenhum WAV/Torch/GPU/extração/escrita.", flush=True); return
    reference20 = json.loads((root / EXP27.REFERENCE20 / "assinatura_execucao.json").read_text(encoding="utf8"))
    for package in ("numpy", "pandas", "scipy", "librosa", "soundfile", "torch", "transformers"):
        require(ident["environment"].get(package) == reference20["environment"].get(package) and ident["environment"].get(package) is not None,
                f"Versão{package} difere do20; usar .venv_wav2vec preservada")
    if pilot_ident is not None:
        require(pilot_ident["configuration"] == ident["configuration"] and pilot_ident["model_files"] == ident["model_files"] and
                pilot_ident["environment"] == ident["environment"], "Piloto/full29 usam configurações/modelos/ambientes diferentes")
    out = (args.saida or root / (PILOT_REL if is_pilot else OUT_REL)).resolve()
    require(out.is_relative_to(root / "results/experiments") and out != root / "results/experiments", "Saída29 deve ficar emresults/experiments")
    require(not is_pilot or out != root / OUT_REL, "Piloto29 não pode usar pasta completa")
    run(root, out, selected, ident, model_dir, use_safe, args.device, pilot)


if __name__ == "__main__": main()
