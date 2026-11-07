# -*- coding: utf-8 -*-
"""Exp32: adaptação LoRA do prefixo XLS-R, com ablação Wavelet por pessoa.

Cache novo com crop central de até 5 s de voz ativa, comum aos dois ramos.
Não altera fontes ou resultados anteriores. Treino real exige --executar.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp25_for_exp32", HERE / "25_treinar_wavelet_cnn_tcn_avfad.py")
EXP25 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXP25)
EXP21, EXP19, EXP17, EXP13 = EXP25.EXP21, EXP25.EXP19, EXP25.EXP17, EXP25.EXP13
VERSION = "32.1"
SEEDS = EXP25.SEEDS
ARMS = ("A_FROZEN_PREFIX5", "B_LORA_PREFIX5", "C_LORA_PREFIX5_WPT", "D_LORA_PREFIX5_WIDE")
OUT_REL = Path("results/experiments/resultados_32_prefixo_lora_wavelet")
CACHE_REL = Path("results/experiments/resultados_32_cache_crops")
MODEL_REL = Path("models/wav2vec2-xls-r-300m/1a640f32ac3e39899438a2931f9924c02f080a54")
BENCH_REL = Path("results/experiments/resultados_31_benchmark_prefixo_lora")
DEFAULT_OPTIONS = {"max_epochs": 12, "min_epochs": 3, "patience": 3, "min_delta": 1e-4,
                   "head_lr": 1e-3, "adapter_lr": 1e-4, "weight_decay": 1e-3,
                   "dropout": .25, "batch_people": 4, "prediction_batch_people": 4, "gradient_clip": 5.}
CSV_FILES = {"predicoes.csv", "predicoes_treino_participante.csv", "metricas.csv", "historico.csv", "particoes_internas.csv"}
ARTIFACTS = CSV_FILES | {"parametros_preprocessamento.npz", "modelo_refit.pt", "modelo_selecao.pt", "selecao_epocas.json"}
CONTRASTS = {"B-A_adaptacao_primaria": {ARMS[1]: 1, ARMS[0]: -1},
             "C-B_contribuicao_Wavelet_primaria": {ARMS[2]: 1, ARMS[1]: -1},
             "C-D_orcamento_secundaria": {ARMS[2]: 1, ARMS[3]: -1},
             "C-A_adaptacao_Wavelet_secundaria": {ARMS[2]: 1, ARMS[0]: -1},
             "D-B_capacidade_secundaria": {ARMS[3]: 1, ARMS[1]: -1}}


def module_at(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prefix_api():
    return module_at("prefix31_for_exp32", "31_prefixo_xlsr_lora.py")


def require(condition, message):
    if not condition: raise ValueError(message)


def protocol(options=DEFAULT_OPTIONS):
    return f"""EXPERIMENTO 32 — PREFIXO XLS-R ADAPTADO E WAVELET, POR PESSOA
Hipóteses: LoRA pode adaptar informação acústica; Wavelet pode acrescentá-la.
Mesmos 411 participantes HSA, 82 normais/329 patológicos. São 2.876 WAVs:
2.465 frases e 411 vogais /a/ 002, sem novas exclusões. Todas as gravações
da pessoa permanecem juntas no treino/validação/teste.
Entrada FAIR: detectar VAD no RAW completo a 48 kHz, concatenar regiões e
reter o trecho CENTRAL de até 5 s de VOZ ATIVA, sem usar rótulos. Guardar
offset, duração retida/descartada e hashes. Os dois ramos usam esse MESMO
crop: WPT RAW sym5 L5/L6, janelas 400/200 ms, 1.920 resumos; SSL reamostra
esse crop para 16 kHz, normaliza por gravação e usa padding/máscara até 80.000.
Essa política difere do 30; não se exige reprodução das métricas anteriores.
Cache novo, sem estatísticas aprendidas entre pessoas; caches antigos intactos.
Prefixo de cinco blocos XLS-R da revisão 1a640f32ac3e39899438a2931f9924c02f080a54,
com LayerNorm final Identity; deve reproduzir hidden_states[5] do modelo completo.
A_FROZEN_PREFIX5: prefixo congelado e cabeça treinável.
B_LORA_PREFIX5: LoRA q_proj/v_proj nos blocos 4/5 (índices 3/4), r=4,
alpha=4, dropout=0,05; pesos originais congelados.
C_LORA_PREFIX5_WPT: mesmo prefixo adaptado e ramos Wavelet das duas fontes.
D_LORA_PREFIX5_WIDE: SSL isolado com MLP hidden64, mais parâmetros que C.
Encoder sempre eval: sem SpecAugment, LayerDrop ou dropout original em treino.
Apenas dropout LoRA e cabeças treinam. Não usar inference_mode em tensor que
será entrada de cabeça treinável. A usa no_grad no encoder; B/C/D têm gradiente.
SSL por gravação: média/DP mascaradas, 2.048 atributos -> LayerNorm ->
MLP 2048->32->32 (hidden64 em D), GELU/dropout e LayerNorm final32.
Projeção SSL compartilhada entre fontes. Fazer média dos latentes das cinco/
seis frases por pessoa; a vogal tem latent próprio. Concatenar as duas fontes
e aplicar MLP64->32->logit. C acrescenta dois latentes Wavelet32 e usa
MLP128->32->logit. Contar capacidade e confirmar D>C antes do treino clínico.
Wavelet: média individual dos resumos das frases e resumo único da vogal.
Imputação/constantes/ANOVA128/StandardScaler ajustados SOMENTE nas pessoas
de treino da partição. Não usar idade, sexo, ano ou diagnóstico como entrada.
Cinco folds externos por pessoa e seis sementes: 20261105 principal e 101–105
estabilidade. Validação interna: primeiro de cinco folds por pessoa dentro
do treino externo, seed=outerseed+100+fold, como 25/26.
Loss BCE após logit por PESSOA, pesos N/(2*Nclasse) das pessoas de treino.
Batch de {options['batch_people']} pessoas e todas as suas frases/vogal. AdamW:
head_lr={options['head_lr']}, LoRA_lr={options['adapter_lr']}, wd={options['weight_decay']},
dropout cabeça={options['dropout']}, gradclip={options['gradient_clip']}.
Float32, TF32 off, operações determinísticas, sem aumento de dados.
Seleção por perda interna ponderada: máximo {options['max_epochs']} épocas,
mínimo {options['min_epochs']}, patience {options['patience']}, min_delta {options['min_delta']}.
Depois reinicializar DO ZERO os mesmos adaptadores/cabeças, novo otimizador,
novos preprocessadores ajustados no treino externo completo; treinar pela
época escolhida. Só depois prever teste externo. Limiar zero fixo.
Salvar somente adaptadores/cabeças, preprocessadores, épocas e históricos;
base congelada referenciada por SHA e verificada antes/depois de ambos treinos.
120 ajustes externos no ciclo completo, sem seleção por BA externa.
Primários B-A e C-B; secundários C-D/C-A/D-B. Métricas somente por participante.
Bootstrap 2.000 por pessoa; sementes da pessoa no mesmo cluster; escopos
principal/estabilidade separados. ICs condicionais, sem corrigir todo histórico.
AVFAD é desenvolvimento, não teste independente. Sem SVD, promessa de
80–90% ou alegação clínica. Preservar negativos e fontes/dados anteriores.
"""


def build_inputs(audio, lengths, wpt, records, real=True):
    required = {"participant_id", "sample_id", "task", "y", "matrix_row"}
    require(required.issubset(records) and not records[list(required)].isna().any().any(), "Manifesto do cache incompleto")
    require(not records.sample_id.duplicated().any() and not records.duplicated(["participant_id", "task"]).any(), "Gravação duplicada")
    require(records.groupby("participant_id").y.nunique().eq(1).all() and set(records.y.unique()) == {0, 1}, "Rótulos por pessoa inconsistentes")
    require(np.array_equal(records.matrix_row, np.arange(len(records))), "matrix_row do cache inválido")
    require(audio.shape[0] == len(records) and lengths.shape == (len(records),) and wpt.shape == (len(records), 1920), "Shapes do cache inválidos")
    require(audio.dtype == np.float32 and wpt.dtype == np.float32 and lengths.dtype.kind in "iu" and (lengths > 0).all() and (lengths <= audio.shape[1]).all(), "Comprimentos/tipos do cache inválidos")
    require(np.isfinite(wpt).all(), "Wavelet não finita")
    people = EXP13.person_table(records); people["matrix_row"] = np.arange(len(people))
    task = records.task.astype(str).str.zfill(3).to_numpy(); ids = records.participant_id.to_numpy()
    groups_sent, groups_vow = [], []
    for pid in people.participant_id:
        sr = np.flatnonzero((ids == pid) & np.isin(task, ["004", "005", "006", "007", "008", "009"]))
        vr = np.flatnonzero((ids == pid) & (task == "002"))
        require(len(sr) > 0 and len(vr) == 1, "Pessoa sem frases ou com vogal duplicada/ausente")
        groups_sent.append(sr); groups_vow.append(int(vr[0]))
    people["n_sentences"] = [len(rows) for rows in groups_sent]
    require(sum(len(rows)+1 for rows in groups_sent) == len(records), "Tarefa fora do protocolo no cache")
    if real:
        require((len(records), len(people), people.y.value_counts().to_dict(), people.n_sentences.value_counts().to_dict()) ==
                (2876, 411, {1: 329, 0: 82}, {6: 410, 5: 1}) and audio.shape[1] == 80000, "Coorte/padding do cache mudou")
    return {"audio": audio, "lengths": lengths, "wpt": wpt, "records": records,
            "people": people, "groups_sent": groups_sent, "groups_vow": np.asarray(groups_vow, np.int64),
            "wpt_sent": np.stack([wpt[rows].mean(axis=0, dtype=np.float64).astype(np.float32) for rows in groups_sent]),
            "wpt_vow": wpt[np.asarray(groups_vow, np.int64)].copy()}


def person_batches(inputs, rows, batch_size, rng=None):
    order = np.asarray(rows, np.int64)
    if rng is not None: order = rng.permutation(order)
    for start in range(0, len(order), batch_size):
        selected = order[start:start+batch_size]; groups, counts = [], []
        for row in selected:
            sentences = inputs["groups_sent"][int(row)]
            groups.append(np.r_[sentences, inputs["groups_vow"][int(row)]])
            counts.append(len(sentences))
        yield selected, np.concatenate(groups), counts


def fit_wavelet(inputs, people, train, enabled):
    if not enabled: return {}
    result = {}
    for modality in ("wpt_sent", "wpt_vow"):
        pre = EXP13.pipeline_for("SVC_linear", 128, 1.)[:-1]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pre.fit(inputs[modality][train], people.iloc[train].y.to_numpy())
        steps = pre.named_steps
        result[modality] = {"imputer": steps["imputer"].statistics_.copy(), "constant_mask": steps["remove_constants"].get_support(),
                            "selector_scores": steps["select"].scores_.copy(), "selector_mask": steps["select"].get_support(),
                            "scaler_mean": steps["scale"].mean_.copy(), "scaler_scale": steps["scale"].scale_.copy(),
                            "train_rows": np.asarray(train, np.int64)}
    return result


def wavelet_transform(values, rows, parameters):
    x = np.asarray(values[rows], dtype=np.float32).copy()
    x = np.where(np.isnan(x), parameters["imputer"], x).astype(np.float32)
    x = x[:, parameters["constant_mask"].astype(bool)][:, parameters["selector_mask"].astype(bool)].copy()
    x -= parameters["scaler_mean"]; x /= parameters["scaler_scale"]
    require(np.isfinite(x).all(), "Normalização Wavelet não finita")
    return np.ascontiguousarray(x)


class LatentMLP(nn.Module):
    def __init__(self, width, hidden=32, dropout=.25, input_norm=False):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(width) if input_norm else nn.Identity(), nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 32), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(32))

    def forward(self, values): return self.body(values)


class PrefixPersonNetwork(nn.Module):
    def __init__(self, prefix, api, arm, options=DEFAULT_OPTIONS):
        super().__init__(); self.prefix, self.api, self.arm = prefix, api, arm
        self.adapt = arm != ARMS[0]; self.with_wavelet = arm == ARMS[2]
        hidden = 64 if arm == ARMS[3] else 32
        self.heads = nn.ModuleDict({"ssl": LatentMLP(2048, hidden, options["dropout"], input_norm=True)})
        if self.with_wavelet:
            self.heads["wpt_sent"] = LatentMLP(128, 32, options["dropout"])
            self.heads["wpt_vow"] = LatentMLP(128, 32, options["dropout"])
        self.heads["fusion"] = nn.Sequential(nn.Linear(128 if self.with_wavelet else 64, 32), nn.GELU(), nn.Dropout(options["dropout"]), nn.Linear(32, 1))
        require(all(not param.requires_grad for name, param in self.prefix.named_parameters() if ".lora_" not in name), "Base do prefixo deve estar congelada")

    def train(self, mode=True):
        super().train(mode); self.api.set_training(self.prefix, bool(mode and self.adapt)); return self

    def forward(self, input_values, attention_mask, sentence_counts, wavelet_sent=None, wavelet_vow=None):
        if self.adapt:
            pooled = self.api.encode_pooled(self.prefix, input_values, attention_mask)
        else:
            with torch.no_grad(): pooled = self.api.encode_pooled(self.prefix, input_values, attention_mask)
        latent = self.heads["ssl"](pooled)
        require(sum(n+1 for n in sentence_counts) == len(latent) and all(n > 0 for n in sentence_counts), "Agrupamento fonte/pessoa incorreto")
        chunks = torch.split(latent, [n+1 for n in sentence_counts])
        sent = torch.stack([chunk[:-1].mean(dim=0) for chunk in chunks]); vow = torch.stack([chunk[-1] for chunk in chunks])
        values = [sent, vow]
        if self.with_wavelet:
            values.extend([self.heads["wpt_sent"](wavelet_sent), self.heads["wpt_vow"](wavelet_vow)])
        return self.heads["fusion"](torch.cat(values, dim=1)).squeeze(-1)

    def trainable_state(self):
        result = {"heads."+key: value.detach().cpu().clone() for key, value in self.heads.state_dict().items()}
        result.update({"adapter."+key: value.detach().cpu().clone() for key, value in self.api.adapter_state(self.prefix).items()})
        return result

    def load_trainable_state(self, state):
        expected = self.trainable_state(); require(set(state) == set(expected), "Estado incompleto ou inclui pesos da base")
        for key, value in state.items(): require(value.shape == expected[key].shape and torch.isfinite(value).all(), "Tensor de estado inválido")
        self.heads.load_state_dict({key.removeprefix("heads."): value for key, value in state.items() if key.startswith("heads.")}, strict=True)
        self.api.load_adapter_state(self.prefix, {key.removeprefix("adapter."): value for key, value in state.items() if key.startswith("adapter.")})


def arm_seed(seed, fold):
    return int((int(seed)*1009 + int(fold)*67 + 32*7919) % (2**32-1))


def make_model(base_prefix, api, arm, random_seed, device, options=DEFAULT_OPTIONS):
    EXP25.seed_everything(random_seed, device)
    prefix = api.copy_prefix(base_prefix, adapt=arm != ARMS[0])
    # Instalar LoRA consome RNG; resetar antes da cabeça pareia A/B.
    EXP25.seed_everything(random_seed+1, device)
    model = PrefixPersonNetwork(prefix, api, arm, options).to(device)
    model.eval()
    return model


def parameter_counts(base_prefix, api, options=DEFAULT_OPTIONS):
    counts = {}
    for arm in ARMS:
        model = make_model(base_prefix, api, arm, 3201, "cpu", options)
        counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        del model
    require(counts[ARMS[3]] > counts[ARMS[2]], "Controle D não excede a capacidade treinável de C")
    return counts


def optimizer_for(model, options):
    head = [param for param in model.heads.parameters() if param.requires_grad]
    adapter = [param for param in model.prefix.parameters() if param.requires_grad]
    groups = [{"params": head, "lr": options["head_lr"]}]
    if adapter: groups.append({"params": adapter, "lr": options["adapter_lr"]})
    return torch.optim.AdamW(groups, weight_decay=options["weight_decay"])


def tensors_for(inputs, selected, record_rows, processor, preprocessors, device):
    # Cópias impedem que a normalização do processor escreva no cache readonly.
    raw = [np.array(inputs["audio"][int(row), :int(inputs["lengths"][int(row)])], dtype=np.float32, copy=True) for row in record_rows]
    batch = processor(raw, sampling_rate=16000, return_tensors="pt", return_attention_mask=True,
                      padding="max_length", max_length=inputs["audio"].shape[1], truncation=False)
    values, mask = batch["input_values"].to(device), batch["attention_mask"].to(device)
    require(values.shape == mask.shape and values.shape[0] == len(record_rows) and torch.isfinite(values).all(), "Processor/máscara inválidos")
    waves = [torch.from_numpy(wavelet_transform(inputs[modality], selected, preprocessors[modality])).to(device)
             for modality in ("wpt_sent", "wpt_vow")] if preprocessors else [None, None]
    return values, mask, waves


def train_epoch(model, inputs, people, rows, processor, preprocessors, optimizer, weights, device, options, random_seed, epoch):
    model.train(); loss_sum, count = 0., 0
    for selected, records, counts in person_batches(inputs, rows, options["batch_people"], np.random.default_rng(random_seed+epoch)):
        values, mask, waves = tensors_for(inputs, selected, records, processor, preprocessors, device)
        y = people.iloc[selected].y.to_numpy(dtype=np.int64); labels = torch.as_tensor(y, device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(values, mask, counts, *waves)
        loss = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * torch.as_tensor(weights[y], device=device, dtype=torch.float32)).mean()
        require(torch.isfinite(loss), "Perda não finita")
        loss.backward(); nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], options["gradient_clip"], error_if_nonfinite=True)
        optimizer.step(); loss_sum += float(loss.detach().cpu())*len(selected); count += len(selected)
    return loss_sum/count


def predict_people(model, inputs, rows, processor, preprocessors, device, batch_people=4):
    model.eval(); values_out = []
    with torch.no_grad():
        for selected, records, counts in person_batches(inputs, rows, batch_people):
            values, mask, waves = tensors_for(inputs, selected, records, processor, preprocessors, device)
            values_out.append(model(values, mask, counts, *waves).detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(values_out); require(np.isfinite(result).all(), "Logits não finitos")
    return result


def validation_loss(scores, labels, weights):
    return float(np.mean((np.logaddexp(0., scores)-np.asarray(labels)*scores)*weights[np.asarray(labels, dtype=int)]))


def prediction_frame(people, rows, scores, arm, seed, fold):
    frame = people.iloc[rows][["participant_id", "y", "n_sentences"]].copy().reset_index(drop=True)
    frame["score"], frame["predicao"] = scores, (np.asarray(scores) >= 0).astype(int)
    frame["arm"], frame["seed"], frame["fold"], frame["limiar"] = arm, seed, fold, 0.
    return frame


def train_arm(inputs, people, train, test, seed, fold, arm, base_prefix, processor, api=None, device="cuda", options=None, log=lambda _: None):
    api = prefix_api() if api is None else api; options = dict(DEFAULT_OPTIONS if options is None else options)
    train, test = np.asarray(train, np.int64), np.asarray(test, np.int64)
    require(not set(people.iloc[train].participant_id) & set(people.iloc[test].participant_id), "Vazamento externo por pessoa")
    fit, validation = EXP25.internal_split(people, train, seed, fold)
    preprocessors = fit_wavelet(inputs, people, fit, arm == ARMS[2]); weights = EXP25.class_weights(people, fit)
    random_seed = arm_seed(seed, fold); base_reference = api.frozen_weight_digest(base_prefix)
    model = make_model(base_prefix, api, arm, random_seed, device, options); initial_state = model.trainable_state()
    initial_digest = EXP25.state_digest(initial_state); before_selection = api.frozen_weight_digest(model.prefix)
    require(before_selection == base_reference, "Cópia/LoRA alterou pesos originais")
    optimizer = optimizer_for(model, options); history = []; best_loss, best_epoch, stale, best_state = float("inf"), None, 0, None
    started = time.monotonic()
    for epoch in range(1, options["max_epochs"]+1):
        loss = train_epoch(model, inputs, people, fit, processor, preprocessors, optimizer, weights, device, options, random_seed, epoch)
        scores = predict_people(model, inputs, validation, processor, preprocessors, device, options["prediction_batch_people"])
        valid_loss = validation_loss(scores, people.iloc[validation].y.to_numpy(), weights)
        if epoch >= options["min_epochs"]:
            if valid_loss < best_loss-options["min_delta"]:
                best_loss, best_epoch, stale, best_state = valid_loss, epoch, 0, model.trainable_state()
            else: stale += 1
        history.append({"phase": "selecao", "epoch": epoch, "train_loss": loss, "validation_loss": valid_loss})
        if epoch >= options["min_epochs"] and stale >= options["patience"]: break
    require(best_epoch is not None, "Nenhuma época elegível")
    after_selection = api.frozen_weight_digest(model.prefix); require(after_selection == base_reference, "Base mudou na seleção")
    del model, optimizer
    model = make_model(base_prefix, api, arm, random_seed, device, options)
    refit_initial_digest = EXP25.state_digest(model.trainable_state()); require(refit_initial_digest == initial_digest, "Refit não reiniciou o mesmo estado")
    before_refit = api.frozen_weight_digest(model.prefix); require(before_refit == base_reference, "Base mudou antes do refit")
    outer_preprocessors = fit_wavelet(inputs, people, train, arm == ARMS[2]); outer_weights = EXP25.class_weights(people, train)
    optimizer = optimizer_for(model, options)
    for epoch in range(1, best_epoch+1):
        loss = train_epoch(model, inputs, people, train, processor, outer_preprocessors, optimizer, outer_weights, device, options, random_seed, epoch)
        history.append({"phase": "refit", "epoch": epoch, "train_loss": loss, "validation_loss": np.nan})
    after_refit = api.frozen_weight_digest(model.prefix); require(after_refit == base_reference, "Base mudou no refit")
    train_scores = predict_people(model, inputs, train, processor, outer_preprocessors, device, options["prediction_batch_people"])
    # Primeiro acesso aos sinais/features de teste após terminar seleção/refit.
    test_scores = predict_people(model, inputs, test, processor, outer_preprocessors, device, options["prediction_batch_people"])
    test_frame = prediction_frame(people, test, test_scores, arm, seed, fold); train_frame = prediction_frame(people, train, train_scores, arm, seed, fold)
    arrays = {"selection__train_rows": fit, "outer__train_rows": train}
    for phase, preprocess in (("selection", preprocessors), ("outer", outer_preprocessors)):
        arrays.update({f"{phase}__{modality}__{key}": value for modality, values in preprocess.items() for key, value in values.items()})
    partitions = []
    for role, rows in (("treino_interno", fit), ("validacao_interna", validation), ("teste_externo", test)):
        partitions.extend({"seed": seed, "fold": fold, "arm": arm, "role": role, "participant_id": row.participant_id, "y": row.y}
                          for row in people.iloc[rows].itertuples())
    selection = {"arm": arm, "seed": int(seed), "fold": int(fold), "random_seed": random_seed, "selected_epoch": int(best_epoch),
                 "selection_epochs_run": int(sum(row["phase"] == "selecao" for row in history)), "best_validation_loss": float(best_loss),
                 "initial_trainable_sha256": initial_digest, "refit_initial_trainable_sha256": refit_initial_digest,
                 "base_reference_sha256": base_reference, "base_before_selection_sha256": before_selection,
                 "base_after_selection_sha256": after_selection, "base_before_refit_sha256": before_refit, "base_after_refit_sha256": after_refit,
                 "n_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                 "n_adapter": sum(p.numel() for p in model.prefix.parameters() if p.requires_grad),
                 "selection_class_weights": weights.tolist(), "outer_class_weights": outer_weights.tolist(),
                 "uses_wavelet": arm == ARMS[2], "seconds": time.monotonic()-started}
    metrics = [{"arm": arm, "seed": seed, "fold": fold, "phase": phase, "unit": "participante", **EXP17.measures(frame)}
               for phase, frame in (("teste", test_frame), ("treino", train_frame))]
    result = {"frames": {"predicoes.csv": test_frame, "predicoes_treino_participante.csv": train_frame, "metricas.csv": pd.DataFrame(metrics),
                         "historico.csv": pd.DataFrame(history), "particoes_internas.csv": pd.DataFrame(partitions)},
              "arrays": arrays, "selection": selection, "state": model.trainable_state(), "selection_state": best_state}
    log(f"32 seed={seed} fold={fold} {arm}: época={best_epoch}; BA teste={EXP17.measures(test_frame)['BA']:.6f}; treino={EXP17.measures(train_frame)['BA']:.6f}; segundos={selection['seconds']:.1f}")
    del model, optimizer
    return result

