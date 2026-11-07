# -*- coding: utf-8 -*-
"""Exp19: avaliacao fatorial fixa WPT48k/16k x escala/quantis por pessoa."""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
import time
import warnings

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp17_for_exp19", HERE / "17_robustez_distribuicao_wpt_avfad.py")
EXP17 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP17)
EXP13 = EXP17.EXP13
VERSION = "19.1"
SEEDS = (20261105, 101, 102, 103, 104, 105)
ARMS = ("A_48K_STANDARD", "B_16K_STANDARD", "C_48K_QUANTILE", "D_16K_QUANTILE")
DATA16_REL = Path("results/experiments/resultados_18_wpt_16khz_extracao")
REF17_REL = Path("results/experiments/resultados_17_robustez_distribuicao_wpt")
OUT_REL = Path("results/experiments/resultados_19_avaliacao_wpt_16khz")
ARTIFACTS = {"predicoes.csv", "predicoes_treino_participante.csv", "metricas.csv", "atributos.csv",
             "predicoes_teste_gravacao.csv", "predicoes_treino_gravacao.csv", "parametros_ajuste.npz"}
CONTRASTS = {
    "B-A_primaria": {ARMS[1]: 1, ARMS[0]: -1},
    "D-C_secundaria": {ARMS[3]: 1, ARMS[2]: -1},
    "D-B_secundaria": {ARMS[3]: 1, ARMS[1]: -1},
    "C-A_secundaria": {ARMS[2]: 1, ARMS[0]: -1},
    "interacao_descritiva": {ARMS[3]: 1, ARMS[2]: -1, ARMS[1]: -1, ARMS[0]: 1},
}


def validate_alignment(x48, m48, n48, x16, m16, n16, embedded, real=True):
    """Recusa desalinhamento; nao reordena ou reconcilia rotulos silenciosamente."""
    fields = ("sample_id", "participant_id", "task", "y")
    for meta in (m48, m16):
        if not set(fields).issubset(meta.columns) or meta[list(fields)].isna().any().any():
            raise ValueError("IDs/rotulos ausentes na metadata")
        if meta.sample_id.duplicated().any() or meta.duplicated(["participant_id", "task"]).any():
            raise ValueError("IDs/tarefas duplicados")
        if meta.groupby("participant_id").y.nunique().ne(1).any():
            raise ValueError("Rotulo divergente por pessoa")
        if set(meta.y.unique()) != {0, 1}:
            raise ValueError("Esperava classes binario 0/1")
    for field in fields:
        if not np.array_equal(m48[field].astype(str), m16[field].astype(str)):
            raise ValueError(f"Metadata48k/16k difere na ordem/conteudo de {field}")
    if x48.shape != x16.shape or x48.shape != (len(m48), len(n48)) or len(n48) != 1920:
        raise ValueError("Matrizes48k/16k devem ter mesmas linhas e1920 atributos")
    if not np.array_equal(n48, n16) or len(set(n48)) != len(n48):
        raise ValueError("Nomes/ordem dos atributos48k/16k divergentes")
    if not np.isfinite(x48).all() or not np.isfinite(x16).all():
        raise ValueError("Matriz auditada contem valores nao finitos")
    for key in ("sample_id", "participant_id", "task"):
        if key not in embedded or not np.array_equal(np.asarray(embedded[key]).astype(str), m16[key].astype(str)):
            raise ValueError(f"NPZ16k e CSV16k diferem em {key}")
    if "matrix_row" in m16 and not np.array_equal(m16.matrix_row, np.arange(len(m16))):
        raise ValueError("matrix_row nao corresponde a ordem exportada")
    if real:
        people = m48[["participant_id", "y"]].drop_duplicates()
        if (len(m48), len(people), people.y.value_counts().to_dict()) != (2465, 411, {1: 329, 0: 82}):
            raise ValueError("Coorte deve ser HSA2465/411/82/329")
        if m48.groupby("participant_id").size().value_counts().to_dict() != {6: 410, 5: 1}:
            raise ValueError("Contagem de frases inesperada")
        if "sample_rate_hz" not in m16 or not m16.sample_rate_hz.eq(16000).all():
            raise ValueError("Metadata16k nao declara taxa alvo16000")


def load_inputs(root, data16):
    x48, m48, n48 = EXP17.load_inputs(root)
    m16file = data16 / "metadata_wpt_sym5_16khz.csv"
    npz16 = data16 / "features_wpt_sym5_16khz.npz"
    complete = data16 / "ARTEFATOS_COMPLETOS.json"
    for path in (m16file, npz16, complete, data16 / "assinatura_execucao.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Extracao18 completa necessaria; ausente: {path}")
    completed = json.loads(complete.read_text(encoding="utf-8"))
    required = {npz16.name, m16file.name, "RESUMO_EXTRACAO.txt", "CONTROLE_REFERENCIA_48KHZ.csv",
                "CONTROLE_VALIDADO.json"}
    if set(completed) != required:
        raise ValueError("Manifesto de exportacao18 incompleto/inesperado")
    for name, digest in completed.items():
        if EXP17.sha(data16 / name) != digest:
            raise ValueError(f"Artefato18 alterado: {name}")
    validate_control_binding(data16)
    m16 = pd.read_csv(m16file, dtype={"sample_id": str, "participant_id": str, "task": str})
    m16["task"] = m16.task.str.zfill(3)
    with np.load(npz16, allow_pickle=False) as data:
        needed = {"wavelet", "feature_names", "sample_id", "participant_id", "task"}
        if not needed.issubset(data.files):
            raise ValueError("NPZ16k sem matriz, nomes ou IDs embutidos")
        x16 = np.asarray(data["wavelet"], dtype=np.float32)
        n16 = data["feature_names"].astype(str)
        embedded = {key: data[key].astype(str) for key in ("sample_id", "participant_id", "task")}
    validate_alignment(x48, m48, n48, x16, m16, n16, embedded)
    control = pd.read_csv(data16 / "CONTROLE_REFERENCIA_48KHZ.csv")
    expected_controls = set(m48.iloc[np.unique(np.linspace(0, len(m48) - 1, 3, dtype=int))].sample_id)
    if len(control) != 3 or set(control.sample_id) != expected_controls or not control.equivalent.eq(True).all():
        raise ValueError("Controles de equivalencia48k do extrator18 nao confirmados")
    return {"48k": x48, "16k": x16}, m48, n48


def validate_control_binding(data16):
    """O controle48k deve pertencer a assinatura exata da extracao18 atual."""
    binding = json.loads((data16 / "CONTROLE_VALIDADO.json").read_text(encoding="utf-8"))
    # O extrator18 assina o texto UTF-8 com LF antes de atomic_text gravar
    # CRLF no Windows. read_text reproduz exatamente esse texto canonico;
    # o hash dos bytes em disco seria diferente apenas pelos fins de linha.
    signature_text = (data16 / "assinatura_execucao.json").read_text(encoding="utf-8")
    expected = {"csv_sha256": EXP17.sha(data16 / "CONTROLE_REFERENCIA_48KHZ.csv"),
                "signature": __import__("hashlib").sha256(signature_text.encode("utf-8")).hexdigest()}
    if binding != expected:
        raise ValueError("Controle48k nao corresponde ao CSV e assinatura18 registrados")


def references(root, ref17dir, meta, seeds):
    p13 = root / "results/experiments/resultados_13_otimizacao_wpt_sym5/predicoes_oof_participante.csv"
    p17 = ref17dir / "predicoes_oof_participante.csv"
    reference13 = pd.read_csv(p13)
    reference13 = reference13[reference13.configuracao == "WPT_fixa"].copy()
    reference17 = pd.read_csv(p17)
    expected_ids = set(meta.participant_id)
    for seed in seeds:
        for arm in ("A_STANDARD_LINEAR", "B_QUANTILE_LINEAR"):
            sub = reference17[(reference17.seed == seed) & (reference17.arm == arm)]
            if len(sub) != len(expected_ids) or set(sub.participant_id) != expected_ids or sub.participant_id.duplicated().any():
                raise ValueError(f"Controle17 incompleto: seed={seed} arm={arm}")
    return reference13, reference17


def identity(root, out, data16, ref17dir, seeds):
    inputs = [root / "data/processed/features_wavelet_temporal_avfad_reduzido.npz",
              root / "data/processed/metadata_wavelet_temporal_avfad_reduzido.csv",
              data16 / "features_wpt_sym5_16khz.npz", data16 / "metadata_wpt_sym5_16khz.csv",
              data16 / "ARTEFATOS_COMPLETOS.json", data16 / "assinatura_execucao.json",
              data16 / "CONTROLE_REFERENCIA_48KHZ.csv", data16 / "CONTROLE_VALIDADO.json",
              ref17dir / "predicoes_oof_participante.csv", ref17dir / "assinatura_execucao.json",
              root / "results/experiments/resultados_13_otimizacao_wpt_sym5/predicoes_oof_participante.csv"]
    code = [Path(__file__).resolve(), HERE / "17_robustez_distribuicao_wpt_avfad.py",
            HERE / "13_otimizar_wpt_sym5.py", HERE / "18_extrair_wpt_16khz_avfad.py",
            HERE / "extrair_wavelet_temporal_avfad_reduzido.py", HERE / "avfad_paths.py"]
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"):
        environment[name] = importlib.metadata.version(name)
    return {"version": VERSION, "configuration": {"arms": list(ARMS), "seeds": list(seeds), "folds": 5,
            "input_features": 1920, "k": 128, "C": 1.0, "kernel": "linear", "class_weight": "balanced",
            "threshold": 0.0, "quantile_before_ANOVA": True, "quantile_n": 256, "quantile_output": "normal",
            "quantile_subsample": None, "quantile_seed": 42, "bootstrap_iterations": 2000,
            "primary": "B-A", "secondary": ["D-C", "D-B", "C-A"], "interaction": "descriptive"},
            "environment": environment, "input_files": {str(p.resolve()): EXP17.sha(p) for p in inputs},
            "code": {str(p): EXP17.sha(p) for p in code}, "protocol_sha256": EXP17.sha(out / "PROTOCOLO.txt")}


def model_for(arm, ntrain):
    # Usa a mesma construcao do17 para tornar os controles comparaveis tambem
    # numericamente; a identidade inclui17 e13 como dependencias executadas.
    return EXP17.model_for("B_QUANTILE_LINEAR" if arm in ARMS[2:] else "A_STANDARD_LINEAR", ntrain)


def check_reference(frame, r13, r17, arm, seed, fold):
    if arm not in (ARMS[0], ARMS[2]):
        return
    targets = []
    if r17 is not None:
        source = "A_STANDARD_LINEAR" if arm == ARMS[0] else "B_QUANTILE_LINEAR"
        targets.append(r17[(r17.seed == seed) & (r17.fold == fold) & (r17.arm == source)])
    if r13 is not None and arm == ARMS[0] and seed == 20261105:
        targets.append(r13[r13.outer_fold == fold])
    current = frame.set_index("participant_id").sort_index()
    for target in targets:
        expected = target.set_index("participant_id").sort_index()
        pd.testing.assert_index_equal(current.index, expected.index)
        for column in ("y", "predicao", "n_gravacoes"):
            np.testing.assert_array_equal(current[column], expected[column])
        np.testing.assert_allclose(current.score, expected.score, atol=1e-9, rtol=1e-9)


def recording_predictions(meta, scores, arm, seed, fold):
    frame = meta[["sample_id", "participant_id", "task", "y"]].copy().reset_index(drop=True)
    frame["score"] = scores
    frame["predicao"] = (frame.score >= 0).astype(int)
    frame["arm"], frame["seed"], frame["fold"] = arm, seed, fold
    return frame


def evaluate_fold(matrices, meta, names, train, test, seed, fold, log, r13=None, r17=None):
    train_meta = meta.iloc[train].reset_index(drop=True)
    test_meta = meta.iloc[test].reset_index(drop=True)
    if set(train_meta.participant_id) & set(test_meta.participant_id):
        raise ValueError("Participante em treino e teste")
    outputs, train_outputs, records_test, records_train, metrics, features, arrays = [], [], [], [], [], [], {}
    for arm in ARMS:
        rate = "48k" if arm in (ARMS[0], ARMS[2]) else "16k"
        x = matrices[rate]
        model = model_for(arm, len(train))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model.fit(x[train], train_meta.y.to_numpy())
        score_train, score_test = model.decision_function(x[train]), model.decision_function(x[test])
        pred = EXP17.collapse(test_meta, score_test)
        check_reference(pred, r13, r17, arm, seed, fold)
        if arm in (ARMS[0], ARMS[2]) and (r13 is not None or r17 is not None):
            log(f"CONTROLE seed={seed} fold={fold} {arm}: referencias confirmadas")
        fit_pred = EXP17.collapse(train_meta, score_train)
        for frame in (pred, fit_pred):
            frame["arm"], frame["seed"], frame["fold"], frame["limiar"] = arm, seed, fold, 0.0
        rt = recording_predictions(test_meta, score_test, arm, seed, fold)
        rr = recording_predictions(train_meta, score_train, arm, seed, fold)
        for phase, unit, frame in (("teste", "participante", pred), ("treino", "participante", fit_pred),
                                   ("teste", "gravacao", rt), ("treino", "gravacao", rr)):
            metrics.append({"arm": arm, "seed": seed, "fold": fold, "phase": phase, "unit": unit,
                            **EXP17.measures(frame)})
        reduced_names = names[model.named_steps["remove_constants"].get_support()]
        selected = model.named_steps["select"].get_support()
        features.extend({"arm": arm, "seed": seed, "fold": fold, "feature": str(name)}
                        for name in reduced_names[selected])
        for key, value in EXP17.parameters(model).items():
            arrays[f"{arm}__{key}"] = value
        if "quantile" in model.named_steps:
            qt = model.named_steps["quantile"]
            arrays[f"{arm}__quantiles"] = qt.quantiles_.copy()
            arrays[f"{arm}__quantile_references"] = qt.references_.copy()
        outputs.append(pred); train_outputs.append(fit_pred); records_test.append(rt); records_train.append(rr)
        log(f"seed={seed} fold={fold} {arm}: BA_teste_pessoa={EXP17.measures(pred)['BA']:.6f}; "
            f"BA_treino_pessoa={EXP17.measures(fit_pred)['BA']:.6f}")
    return {"predicoes.csv": pd.concat(outputs, ignore_index=True),
            "predicoes_treino_participante.csv": pd.concat(train_outputs, ignore_index=True),
            "predicoes_teste_gravacao.csv": pd.concat(records_test, ignore_index=True),
            "predicoes_treino_gravacao.csv": pd.concat(records_train, ignore_index=True),
            "metricas.csv": pd.DataFrame(metrics), "atributos.csv": pd.DataFrame(features)}, arrays


def validate(frames, arrays, meta, train, test, names, seed, fold, r13=None, r17=None):
    expected_y = meta.drop_duplicates("participant_id").set_index("participant_id").y
    for filename, indices, unit in (("predicoes.csv", test, "participante"),
                                    ("predicoes_treino_participante.csv", train, "participante"),
                                    ("predicoes_teste_gravacao.csv", test, "gravacao"),
                                    ("predicoes_treino_gravacao.csv", train, "gravacao")):
        frame = frames[filename]
        if set(frame.arm) != set(ARMS):
            raise ValueError("Conjunto de bracos inesperado")
        expected_meta = meta.iloc[indices]
        key = "participant_id" if unit == "participante" else "sample_id"
        expected_ids = set(expected_meta[key])
        for arm, sub in frame.groupby("arm"):
            if len(sub) != len(expected_ids) or sub[key].duplicated().any() or set(sub[key]) != expected_ids:
                raise ValueError(f"IDs/quantidade inesperados em {filename}:{arm}")
            if not sub.seed.eq(seed).all() or not sub.fold.eq(fold).all():
                raise ValueError("Seed/fold incorreto")
            if not np.array_equal(sub.y, sub.participant_id.map(expected_y)):
                raise ValueError("Rotulo salvo difere do manifesto")
            if not np.isfinite(sub.score).all() or not np.array_equal(sub.predicao, (sub.score >= 0).astype(int)):
                raise ValueError("Score ou decisao inconsistente")
            if unit == "participante":
                counts = expected_meta.groupby("participant_id").size()
                if not np.array_equal(sub.n_gravacoes, sub.participant_id.map(counts)):
                    raise ValueError("Numero de frases por participante inconsistente")
            else:
                original = expected_meta.set_index("sample_id").loc[sub.sample_id]
                if not np.array_equal(sub.participant_id, original.participant_id) or not np.array_equal(sub.task.astype(str).str.zfill(3), original.task.astype(str).str.zfill(3)):
                    raise ValueError("Identidade/tarefa incorreta por gravacao")
            if filename == "predicoes.csv":
                check_reference(sub, r13, r17, arm, seed, fold)
    for arm, sub in frames["atributos.csv"].groupby("arm"):
        if len(sub) != 128 or sub.feature.duplicated().any() or not sub.feature.isin(names).all():
            raise ValueError("Selecao de atributos invalida")
        if not sub.seed.eq(seed).all() or not sub.fold.eq(fold).all():
            raise ValueError("Particao incorreta em atributos")
        constant = arrays[f"{arm}__constant_mask"].astype(bool)
        selected = arrays[f"{arm}__selector_mask"].astype(bool)
        if len(constant) != len(names) or int(selected.sum()) != 128:
            raise ValueError("Mascaras de selecao inconsistentes")
        if not np.array_equal(sub.feature.to_numpy(), names[constant][selected]):
            raise ValueError("Atributos divergem dos parametros de ajuste")
    if set(frames["atributos.csv"].arm) != set(ARMS) or len(frames["metricas.csv"]) != 16:
        raise ValueError("Checkpoint incompleto")
    sources = {("teste", "participante"): "predicoes.csv", ("treino", "participante"): "predicoes_treino_participante.csv",
               ("teste", "gravacao"): "predicoes_teste_gravacao.csv", ("treino", "gravacao"): "predicoes_treino_gravacao.csv"}
    for _, row in frames["metricas.csv"].iterrows():
        source = frames[sources[(row.phase, row.unit)]]
        measured = EXP17.measures(source[source.arm == row.arm])
        for key, value in measured.items():
            if not np.isclose(value, row[key], rtol=0, atol=1e-12):
                raise ValueError(f"Metrica {key} nao corresponde as predicoes")


def save_checkpoint(destination, frames, arrays, signature):
    destination.mkdir(parents=True, exist_ok=True)
    for filename, frame in frames.items():
        EXP17.csv_write(destination / filename, frame)
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    EXP17.atomic_bytes(destination / "parametros_ajuste.npz", stream.getvalue())
    EXP17.json_write(destination / "CONCLUIDO.json", {"finished": EXP17.now(), "run_signature": signature,
                     "sha256": {name: EXP17.sha(destination / name) for name in ARTIFACTS}})


def load_checkpoint(destination, signature):
    marker = json.loads((destination / "CONCLUIDO.json").read_text(encoding="utf-8"))
    if marker["run_signature"] != signature or set(marker["sha256"]) != ARTIFACTS:
        raise ValueError("Checkpoint de outra configuracao ou incompleto")
    for name, digest in marker["sha256"].items():
        if EXP17.sha(destination / name) != digest:
            raise ValueError(f"Checkpoint alterado: {destination / name}")
    frames = {name: pd.read_csv(destination / name, float_precision="round_trip") for name in ARTIFACTS if name.endswith(".csv")}
    with np.load(destination / "parametros_ajuste.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    return frames, arrays


def consolidate(out, meta, seeds, log):
    roots = [out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}" for seed in seeds for fold in range(1, 6)]
    raw = pd.concat([pd.read_csv(p / "predicoes.csv", float_precision="round_trip") for p in roots], ignore_index=True)
    folds = pd.concat([pd.read_csv(p / "metricas.csv", float_precision="round_trip") for p in roots], ignore_index=True)
    features = pd.concat([pd.read_csv(p / "atributos.csv") for p in roots], ignore_index=True)
    summaries = []
    for (seed, arm), group in raw.groupby(["seed", "arm"]):
        if len(group) != meta.participant_id.nunique() or group.participant_id.duplicated().any():
            raise ValueError("OOF duplicado/incompleto na consolidacao")
        measured = EXP17.measures(group)
        measured["AUC_OOF_exploratoria"] = measured.pop("AUC")
        f = folds[(folds.seed == seed) & (folds.arm == arm) & (folds.phase == "teste") & (folds.unit == "participante")]
        measured["AUC_media_folds_ponderada_pares"] = float(np.average(f.AUC, weights=(f.TN + f.FP) * (f.TP + f.FN)))
        summaries.append({"seed": seed, "arm": arm, **measured})
    summary = pd.DataFrame(summaries)
    comparisons = []
    scopes = [("principal_seed20261105", [20261105])] if 20261105 in seeds else []
    secondary = [s for s in seeds if s != 20261105]
    if secondary:
        scopes.append(("estabilidade_secundaria", secondary))
    for label, selected in scopes:
        for name, coefficients in CONTRASTS.items():
            comparisons.append({"escopo": label, "comparacao": name, "seeds": str(selected),
                                **EXP17.intervals(raw, selected, coefficients, iterations=2000)})
    for name, data in (("predicoes_oof_participante.csv", raw), ("metricas_por_fold.csv", folds),
                       ("metricas_por_seed.csv", summary), ("comparacoes_pareadas.csv", pd.DataFrame(comparisons)),
                       ("atributos_selecionados.csv", features)):
        EXP17.csv_write(out / name, data)
    lines = ["EXPERIMENTO19 - RESULTADOS WPT48K/16K x STANDARD/QUANTILE", f"Consolidado: {EXP17.now()}", "",
             "Todos os folds das sementes solicitadas estao concluidos.",
             "Ciclo de seis sementes completo." if tuple(seeds) == SEEDS else "Configuracao parcial/diferente das seis sementes do protocolo.",
             "Controles A19=A17 em todas as sementes e A13 na primaria; C19=B17 em todas as sementes conferidos por fold.",
             "BA principal por participante; media das margens das frases e limiar0.",
             "AUC principal de apoio: media por fold ponderada por pares normal-patologico.",
             "AUC de margens concatenadas entre folds e apenas exploratoria.", ""]
    for row in summary.itertuples():
        lines.append(f"seed={row.seed} {row.arm}: BA={row.BA:.6f}; sens={row.sensibilidade:.6f}; esp={row.especificidade:.6f}; "
                     f"acc={row.acuracia:.6f}; AUC_fold={row.AUC_media_folds_ponderada_pares:.6f}; TN/FP/FN/TP={row.TN}/{row.FP}/{row.FN}/{row.TP}")
    lines += ["", "CONTRASTES PAREADOS; IC95 CONDICIONAL,2000BOOTSTRAPS AGRUPADOS POR PESSOA"]
    for row in comparisons:
        lines.append(f"{row['escopo']} {row['comparacao']}: deltaBA={row['delta_BA']:+.6f}; "
                     f"IC95=[{row['IC95_inferior']:+.6f},{row['IC95_superior']:+.6f}]")
    if secondary:
        lines += ["", "ESTABILIDADE NAS SEMENTES SECUNDARIAS; DP NAO E IC"]
        for arm, group in summary[summary.seed.isin(secondary)].groupby("arm"):
            lines.append(f"{arm}: BA media={group.BA.mean():.6f}; min={group.BA.min():.6f}; max={group.BA.max():.6f}; DP={group.BA.std():.6f}")
    lines += ["", "DIAGNOSTICO DE TREINO, APENAS DESCRITIVO"]
    for (arm, unit), group in folds[folds.phase == "treino"].groupby(["arm", "unit"]):
        lines.append(f"{arm} {unit}: BA media entre fits={group.BA.mean():.6f}; acc media={group.acuracia.mean():.6f}")
    lines += ["", "LIMITACOES", "Mesmo conjunto AVFAD ja explorado: desenvolvimento, nao teste externo independente.",
              "Mudar48k para16k altera largura de banda e particao de frequencia simultaneamente.",
              "Os intervalos mantem as repeticoes da pessoa no mesmo cluster e nao refazem extracao/selecao/treino.",
              "Nenhum braco/seed foi escolhido por desempenho externo. Resultados negativos sao preservados.",
              "Metricas de treino sao diagnosticos de sobreajuste, nunca resultados de generalizacao.",
              "NPZ48k legado depende de seu CSV parceiro; NPZ16k possui IDs embutidos. Rotulos16k sao verificados no CSV contra48k.",
              "SVD nao participa; nenhum dado original ou resultado17/18 foi sobrescrito.",
              "Ver PROTOCOLO.txt, assinatura_execucao.json, particoes_participantes.csv e checkpoints.",
              "Este documento nao e conclusao final do TG.\n"]
    EXP17.text_write(out / "RELATORIO_RESULTADOS.txt", "\n".join(lines))
    log("Consolidacao19 concluida; todos os controles e escopos relatados.")


def fold_table(meta, seeds):
    rows = []
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            for role, indices in (("treino", train), ("teste", test)):
                for row in meta.iloc[indices].drop_duplicates("participant_id").itertuples():
                    rows.append({"seed": seed, "fold": fold, "role": role, "participant_id": row.participant_id, "y": row.y})
    return pd.DataFrame(rows)


def selftest():
    rng = np.random.default_rng(19161105)
    meta = pd.DataFrame([{"participant_id": f"S{p:03d}", "sample_id": f"S{p:03d}_{task}", "task": task,
                          "y": int(p % 4 != 0), "matrix_row": p * 3 + t}
                         for p in range(36) for t, task in enumerate(("004", "005", "006"))])
    names = np.asarray([f"WPT__sym5__L5__TEST{i:04d}" for i in range(1920)])
    x48 = rng.normal(size=(len(meta), 1920)).astype("float32")
    x16 = (x48 + rng.normal(0, .2, size=x48.shape)).astype("float32")
    embedded = {key: meta[key].to_numpy(dtype=str) for key in ("sample_id", "participant_id", "task")}
    validate_alignment(x48, meta, names, x16, meta, names, embedded, real=False)
    for kind in ("order", "label", "embedded"):
        other = meta.copy(); ids = {key: val.copy() for key, val in embedded.items()}
        if kind == "order":
            other = other.iloc[::-1].reset_index(drop=True)
        elif kind == "label":
            other.loc[other.participant_id == "S000", "y"] = 1
        else:
            ids["participant_id"][0] = "OUTRO"
        try:
            validate_alignment(x48, meta, names, x16, other, names, ids, real=False)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Corrupcao {kind} foi aceita")
    train, test = next(EXP13.row_splits(meta, 5, 20261105))
    matrices = {"48k": x48, "16k": x16}
    frames, parameters = evaluate_fold(matrices, meta, names, train, test, 20261105, 1, lambda _: None)
    altered = x16.copy(); altered[test] = rng.normal(1e6, 1e5, size=altered[test].shape)
    altered_frames, altered_parameters = evaluate_fold({"48k": x48, "16k": altered}, meta, names, train, test, 20261105, 1, lambda _: None)
    pd.testing.assert_frame_equal(frames["atributos.csv"], altered_frames["atributos.csv"])
    for key in parameters:
        np.testing.assert_array_equal(parameters[key], altered_parameters[key])
    for arm in (ARMS[0], ARMS[2]):
        pd.testing.assert_frame_equal(frames["predicoes.csv"].query("arm == @arm"), altered_frames["predicoes.csv"].query("arm == @arm"))
    validate(frames, parameters, meta, train, test, names, 20261105, 1)
    with TemporaryDirectory(prefix="avfad19_selftest_") as temporary:
        destination = Path(temporary)
        # Controle transplantado de outra extracao continua invalido mesmo
        # que o CSV48k tenha conteudo identico.
        control = destination / "controle18"
        control.mkdir()
        EXP17.json_write(control / "assinatura_execucao.json", {"synthetic": True})
        EXP17.text_write(control / "CONTROLE_REFERENCIA_48KHZ.csv", "sample_id,equivalent\nS000_004,True\n")
        signature_text = (control / "assinatura_execucao.json").read_text(encoding="utf-8")
        binding = {"csv_sha256": EXP17.sha(control / "CONTROLE_REFERENCIA_48KHZ.csv"),
                   "signature": __import__("hashlib").sha256(signature_text.encode("utf-8")).hexdigest()}
        EXP17.json_write(control / "CONTROLE_VALIDADO.json", binding)
        validate_control_binding(control)
        # Mesmo texto, representacoes LF e CRLF: o contrato do extrator18
        # precisa ser aceito em ambas as plataformas.
        for newline in ("\n", "\r\n"):
            (control / "assinatura_execucao.json").write_bytes(signature_text.replace("\n", newline).encode("utf-8"))
            validate_control_binding(control)
        EXP17.json_write(control / "CONTROLE_VALIDADO.json", {**binding, "signature": "outra_extracao"})
        try:
            validate_control_binding(control)
        except ValueError:
            pass
        else:
            raise AssertionError("Controle de outra extracao foi aceito")
        destination = destination / "checkpoint"
        save_checkpoint(destination, frames, parameters, "synthetic")
        loaded, loaded_parameters = load_checkpoint(destination, "synthetic")
        validate(loaded, loaded_parameters, meta, train, test, names, 20261105, 1)
        before = {path.name: EXP17.sha(path) for path in destination.iterdir()}
        load_checkpoint(destination, "synthetic")
        assert before == {path.name: EXP17.sha(path) for path in destination.iterdir()}
        with (destination / "atributos.csv").open("a", encoding="utf-8") as stream:
            stream.write("corrompido\n")
        try:
            load_checkpoint(destination, "synthetic")
        except ValueError:
            pass
        else:
            raise AssertionError("Hash corrompido foi aceito")
    print("AUTOTESTE19 OK: alinhamento/rotulos, isolamento teste16k, parametros, metricas e checkpoints; sem AVFAD.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados", type=Path, default=ROOT)
    parser.add_argument("--saida", type=Path, default=ROOT / OUT_REL)
    parser.add_argument("--dados16", type=Path)
    parser.add_argument("--referencia17", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verificar", action="store_true")
    actions.add_argument("--executar", action="store_true")
    actions.add_argument("--autoteste", action="store_true")
    args = parser.parse_args()
    if args.autoteste:
        selftest()
        return
    root, out = args.dados.resolve(), args.saida.resolve()
    if not out.is_relative_to(root / "results/experiments"):
        parser.error("--saida deve estar dentro de results/experiments")
    seeds = tuple(args.seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 or s > 2**32 - 1 for s in seeds):
        parser.error("Seeds devem ser unicas e validas para NumPy/sklearn")
    if not (out / "PROTOCOLO.txt").is_file():
        raise FileNotFoundError("Registrar PROTOCOLO.txt antes de qualquer ajuste")
    data16 = args.dados16.resolve() if args.dados16 else root / DATA16_REL
    ref17dir = args.referencia17.resolve() if args.referencia17 else root / REF17_REL
    matrices, meta, names = load_inputs(root, data16)
    r13, r17 = references(root, ref17dir, meta, seeds)
    ident = identity(root, out, data16, ref17dir, seeds)
    folds = fold_table(meta, seeds)
    print(f"ENTRADAS19 OK: duas matrizes{matrices['48k'].shape};411pessoas;1920colunas;seeds={seeds}.", flush=True)
    if not args.executar:
        print("Somente verificacao, sem treino e sem escrita.", flush=True)
        return
    signature_file = out / "assinatura_execucao.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf-8")) != ident:
            raise RuntimeError("Configuracao/dados/codigo/ambiente mudaram; nao reutilizar pasta")
    else:
        if any(p.name != "PROTOCOLO.txt" for p in out.iterdir()):
            raise RuntimeError("Pasta sem assinatura contem outros arquivos; preservar e revisar")
        EXP17.json_write(signature_file, ident)
    run_signature = __import__("hashlib").sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()
    EXP17.freeze_text(out / "particoes_participantes.csv", folds.to_csv(index=False, lineterminator="\n"))
    log = EXP17.Logger(out / "LOG_EXECUCAO.txt")
    log(f"Inicio/retomada19;seeds={seeds};sem busca,sem novas exclusoes.")
    for seed in seeds:
        for fold, (train, test) in enumerate(EXP13.row_splits(meta, 5, seed), 1):
            destination = out / "checkpoints" / f"seed_{seed}_fold_{fold:02d}"
            if (destination / "CONCLUIDO.json").exists():
                frames, arrays = load_checkpoint(destination, run_signature)
                validate(frames, arrays, meta, train, test, names, seed, fold, r13, r17)
                log(f"Checkpoint19 seed={seed} fold={fold} confirmado; nenhum modelo reajustado")
                continue
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Checkpoint parcial preservado: {destination}")
            started = time.monotonic()
            frames, arrays = evaluate_fold(matrices, meta, names, train, test, seed, fold, log, r13, r17)
            validate(frames, arrays, meta, train, test, names, seed, fold, r13, r17)
            save_checkpoint(destination, frames, arrays, run_signature)
            log(f"Concluido19 seed={seed} fold={fold};segundos={time.monotonic()-started:.2f}")
    consolidate(out, meta, seeds, log)


if __name__ == "__main__":
    main()
