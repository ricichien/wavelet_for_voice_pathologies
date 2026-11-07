# -*- coding: utf-8 -*-
"""Exp26: CNN/TCN Wavelet + embeddings XLS-R congelados, por participante."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("exp25_for_exp26", HERE / "25_treinar_wavelet_cnn_tcn_avfad.py")
EXP25 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXP25)
EXP21, EXP19, EXP17, EXP13 = EXP25.EXP21, EXP25.EXP19, EXP25.EXP17, EXP25.EXP13
VERSION = "26.1"
SEEDS = EXP25.SEEDS
ARMS = ("A_WAV_MLP32_ONLY", "B_WAV_MLP64_BUDGET", "C_CNN_RMS_WAV", "D_TCN_RMS_WAV")
DEFAULT_OPTIONS = dict(EXP25.DEFAULT_OPTIONS)
DATA20_REL = EXP21.DATA20_REL
DATA23_REL = EXP25.DATA23_REL
REF25_REL = EXP25.OUT_REL
OUT_REL = Path("results/experiments/resultados_26_hibrido_wavelet_temporal_wav2vec")
MODEL_REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
BASE_CONTRASTS = {"C-A_complementaridade_CNN_primaria": {ARMS[2]: 1, ARMS[0]: -1},
                  "D-A_complementaridade_TCN_secundaria": {ARMS[3]: 1, ARMS[0]: -1},
                  "C-B_controle_capacidade_secundaria": {ARMS[2]: 1, ARMS[1]: -1},
                  "D-B_controle_capacidade_secundaria": {ARMS[3]: 1, ARMS[1]: -1},
                  "D-C_arquiteturas_secundaria": {ARMS[3]: 1, ARMS[2]: -1}}
REFERENCE_CONTRASTS = {"C-CNN25_ganho_sobre_Wavelet_secundaria": {ARMS[2]: 1, "REF25_B_CNN_RMS": -1},
                       "D-TCN25_ganho_sobre_Wavelet_secundaria": {ARMS[3]: 1, "REF25_D_TCN_RMS": -1}}
save_checkpoint, load_checkpoint = EXP25.save_checkpoint, EXP25.load_checkpoint


def protocol(options=DEFAULT_OPTIONS):
    return f"""EXPERIMENTO 26 — WAVELET TEMPORAL + WAV2VEC 2.0 CONGELADO
Hipótese: a representação temporal WPT acrescenta informação aos embeddings
XLS-R; a hipótese exige comparação com o modelo profundo isolado.
Coorte HSA: 411 participantes,82 normais/329 patológicos,2.465 frases.
Sem exclusões novas, alteração de rótulos, dados SVD ou treinamento do encoder.
Wavelet RMS do23: sym5 48 kHz, L5/L6,480 atributos por janela400/200 ms,
interpolação de toda a extensão para128 passos, como25. Não trunca o áudio,
mas a interpolação comprime as sequências e pode perder transientes.
Embeddings20: facebook/wav2vec2-xls-r-300m, revisão{MODEL_REVISION},
encoder congelado, última camada1024, média/DP temporal:2.048 atributos.
A_WAV_MLP32_ONLY:2048→32→32, GELU e dropout{options['dropout']}, cabeça linear32→1.
B_WAV_MLP64_BUDGET:2048→64→32, mesma ativação/dropout/cabeça; controle com
mais parâmetros treináveis que cada híbrido, não igualdade de capacidade.
C_CNN_RMS_WAV: encoder CNN_RMS25→32 e MLP32 dos embeddings→32;
concatenar os dois vetores e aplicar uma cabeça linear64→1.
D_TCN_RMS_WAV: mesmo desenho, com encoder TCN_RMS25→32.
As cabeças escalares Wavelet25 são removidas dos módulos híbridos; usa encode.
CNN/TCN e RMS foram definidos ANTES de consultar resultados25; não são vencedores
escolhidos pelo teste. Todos os ramos acima do cache de embeddings são treináveis.
Estatísticas dos embeddings e canais Wavelet são ajustadas apenas no treino.
Mesmos cinco folds externos por participante e seis sementes13/17/21/25:
20261105 principal,101..105 estabilidade. Todas as frases da pessoa juntas.
Validação interna: primeiro fold de StratifiedKFold5 por pessoa dentro do
treino externo, seed=outerseed+100+outerfold, como25.
Treinar usando média dos logits das frases ANTES da BCE, pesos N/(2*Nclasse)
por participante de treino e batches de{options['batch_people']} pessoas.
Escolher época por BCE de validação interna, usando pesos do treino interno:
máximo{options['max_epochs']}, mínimo{options['min_epochs']}, patience{options['patience']},
min_delta{options['min_delta']}. Nenhuma escolha de época por BA externa.
Refit do zero no treino externo completo pela época escolhida, com a mesma
inicialização e novas estatísticas do treino externo completo.
AdamW lr{options['learning_rate']}, weight_decay{options['weight_decay']}; sem scheduler,
augmentation, busca de arquitetura, SMOTE ou calibração de limiar.
Clipping global de gradiente norm={options['gradient_clip']}, erro se não finito,
mesma regra dos modelos25. Esse limite não é ajustado usando o teste externo.
Float32, TF32 desligado, operações determinísticas; mesmas famílias de seeds25.
Decisão por participante: média dos logits das frases e limiar0. Métricas de
gravação/treino são apoio. AUC por fold ponderada por pares; concatenada exploratória.
C-A primário; D-A,C-B,D-B,D-C secundários. Referências Wavelet25 são OOFs
importadas sem reajuste; C-CNN25 e D-TCN25 são comparações secundárias.
IC95 pareado por pessoa,2000 bootstraps estratificados; seeds da mesma pessoa
permanecem juntas. ICs condicionais não refazem treino nem corrigem exploração
histórica da AVFAD. Estudo de desenvolvimento, não validação clínica independente.
Guardar partições, história, épocas, normalizadores, estados CPU e hashes.
Resultados antigos permanecem intactos; nenhum resultado negativo é omitido.
"""


def uses_wavelet(arm):
    return arm in ARMS[2:]


def validate_embedding_alignment(embeddings, embedding_meta, embedded, names, meta, identity, real=True):
    EXP21.validate_model_configuration(identity)
    if identity["configuration"]["model_revision"] != MODEL_REVISION:
        raise ValueError("Revisão XLS-R difere do cache congelado previsto")
    if embeddings.shape != (len(meta), 2048) or not np.isfinite(embeddings).all() or not np.array_equal(names, EXP21.embedding_names()):
        raise ValueError("Embeddings ou nomes fora do contrato20")
    for key in ("sample_id", "participant_id", "task", "y"):
        if key not in embedding_meta or embedding_meta[key].isna().any() or not np.array_equal(embedding_meta[key].astype(str), meta[key].astype(str)):
            raise ValueError(f"Metadata20/23 diverge em {key}")
    for key in ("sample_id", "participant_id", "task"):
        if key not in embedded or not np.array_equal(embedded[key].astype(str), meta[key].astype(str)):
            raise ValueError(f"NPZ20/23 diverge em {key}")
    if embedding_meta.sample_id.duplicated().any() or embedding_meta.groupby("participant_id").y.nunique().ne(1).any():
        raise ValueError("IDs ou rótulos20 inconsistentes")
    if "matrix_row" not in embedding_meta or not np.array_equal(embedding_meta.matrix_row, np.arange(len(meta))):
        raise ValueError("Ordem dos embeddings não corresponde às linhas do manifesto")
    if real:
        people = meta[["participant_id", "y"]].drop_duplicates()
        if (len(meta), len(people), people.y.value_counts().to_dict()) != (2465, 411, {1:329, 0:82}):
            raise ValueError("Coorte não corresponde a2465/411/82/329")


def load_inputs(root=ROOT, data23=None, data20=None, materialize=True, options=DEFAULT_OPTIONS):
    root = Path(root).resolve()
    inputs, meta, names = EXP25.load_inputs(root, data23, options["steps"], materialize=materialize)
    data20 = Path(data20 or root / DATA20_REL).resolve()
    exports = json.loads((data20 / "ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    required = {"features_wav2vec_xlsr300m.npz", "metadata_wav2vec.csv", "assinatura_execucao.json", "RESUMO_EXTRACAO.txt"}
    if not required.issubset(exports):
        raise ValueError("Cache20 completo é necessário")
    for name, digest in exports.items():
        if Path(name).name != name or EXP17.sha(data20/name) != digest:
            raise ValueError(f"Artefato20 alterado ou caminho inválido:{name}")
    identity20 = json.loads((data20 / "assinatura_execucao.json").read_text(encoding="utf8"))
    embedding_meta = pd.read_csv(data20 / "metadata_wav2vec.csv", dtype={"sample_id":str,"participant_id":str,"task":str}, float_precision="round_trip")
    embedding_meta["task"] = embedding_meta.task.str.zfill(3)
    with np.load(data20 / "features_wav2vec_xlsr300m.npz", allow_pickle=False) as z:
        embeddings, feature_names = np.asarray(z["embeddings"], np.float32), z["feature_names"].astype(str)
        embedded = {key:z[key].astype(str) for key in ("sample_id", "participant_id", "task")}
    validate_embedding_alignment(embeddings, embedding_meta, embedded, feature_names, meta, identity20)
    # Os outros braços do25 não são usados neste experimento; liberar referências.
    return {"rms": inputs["rms"], "wav": embeddings}, meta, names


class EmbeddingMLP(nn.Module):
    def __init__(self, hidden=32, dropout=.25):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2048, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 32), nn.GELU(), nn.Dropout(dropout))

    def forward(self, embeddings):
        return self.body(embeddings)


class HybridNetwork(nn.Module):
    def __init__(self, family=None, hidden=32, dropout=.25):
        super().__init__()
        if family is None:
            self.wavelet = None
        else:
            self.wavelet = (EXP25.CNNWavelet if family == "cnn" else EXP25.TCNWavelet)(dropout=dropout)
            self.wavelet.classifier = nn.Identity()
        self.wav = EmbeddingMLP(hidden, dropout)
        self.classifier = nn.Linear(64 if family else 32, 1)

    def encode(self, sequence, embeddings):
        wav = self.wav(embeddings)
        return wav if self.wavelet is None else torch.cat([self.wavelet.encode(sequence), wav], dim=1)

    def forward(self, sequence, embeddings):
        return self.classifier(self.encode(sequence, embeddings)).squeeze(-1)


def model_for(arm, options=DEFAULT_OPTIONS):
    if arm not in ARMS:
        raise ValueError("Braço26 desconhecido")
    return HybridNetwork({ARMS[2]:"cnn", ARMS[3]:"tcn"}.get(arm), 64 if arm == ARMS[1] else 32, options["dropout"])


def arm_seed(seed, fold, arm):
    family_arm = {ARMS[0]:"E_MLP_RESUMOS_RMS", ARMS[1]:"E_MLP_RESUMOS_RMS", ARMS[2]:"B_CNN_RMS", ARMS[3]:"D_TCN_RMS"}[arm]
    return EXP25.arm_seed(seed, fold, family_arm)


def fit_scalers(inputs, rows, arm):
    scalers = {"wav": EXP25.fit_scaler(inputs["wav"], rows)}
    if uses_wavelet(arm):
        scalers["rms"] = EXP25.fit_scaler(inputs["rms"], rows)
    return scalers


def transformed_batch(inputs, rows, scalers, device):
    embedding = torch.from_numpy(EXP25.transform(inputs["wav"], rows, scalers["wav"])).to(device)
    sequence = torch.from_numpy(EXP25.transform(inputs["rms"], rows, scalers["rms"])).to(device) if "rms" in scalers else None
    return sequence, embedding


def train_epoch(model, inputs, meta, rows, scalers, optimizer, weights, device, options, random_seed, epoch):
    model.train()
    total_loss, total_people = 0., 0
    for rr, counts, labels in EXP25.batches(meta, rows, options["batch_people"], np.random.default_rng(random_seed+epoch)):
        xseq, xwav = transformed_batch(inputs, rr, scalers, device)
        y = torch.as_tensor(labels, device=device, dtype=torch.float32)
        w = torch.as_tensor(weights[labels], device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = EXP25.person_logits(model(xseq, xwav), counts)
        loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Perda26 não finita")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(labels)
        total_people += len(labels)
    return total_loss / total_people


def predict_records(model, inputs, rows, scalers, device, batch_size=64):
    model.eval(); scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            seq, wav = transformed_batch(inputs, rows[start:start+batch_size], scalers, device)
            scores.append(model(seq, wav).detach().cpu().numpy().astype(np.float64))
    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise RuntimeError("Logits26 não finitos")
    return result


def train_arm(inputs, meta, train, test, seed, fold, arm, device="cuda", options=None, log=lambda _:None):
    options = dict(DEFAULT_OPTIONS if options is None else options)
    train, test = np.asarray(train, np.int64), np.asarray(test, np.int64)
    if set(meta.iloc[train].participant_id) & set(meta.iloc[test].participant_id):
        raise ValueError("Participante compartilhado entre treino e teste externo")
    fit, validation = EXP25.internal_split(meta, train, seed, fold)
    selection_scalers = fit_scalers(inputs, fit, arm)
    weights = EXP25.class_weights(meta, fit)
    random_seed = arm_seed(seed, fold, arm)
    EXP25.seed_everything(random_seed, device)
    model = model_for(arm, options).to(device)
    initial_digest = EXP25.state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    history, best_loss, best_epoch, stale, best_state = [], float("inf"), None, 0, None
    started = time.monotonic()
    for epoch in range(1, options["max_epochs"]+1):
        loss = train_epoch(model, inputs, meta, fit, selection_scalers, optimizer, weights, device, options, random_seed, epoch)
        val_scores = predict_records(model, inputs, validation, selection_scalers, device, options["prediction_batch"])
        val_loss = EXP25.person_loss(meta.iloc[validation].reset_index(drop=True), val_scores, weights)
        if epoch >= options["min_epochs"]:
            if val_loss < best_loss - options["min_delta"]:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            else:
                stale += 1
        history.append({"phase":"selecao", "epoch":epoch, "train_loss":loss, "validation_loss":val_loss})
        if epoch >= options["min_epochs"] and stale >= options["patience"]:
            break
    if best_epoch is None:
        raise RuntimeError("Nenhuma época elegível selecionada")
    del model, optimizer
    EXP25.seed_everything(random_seed, device)
    model = model_for(arm, options).to(device)
    if EXP25.state_digest(model.state_dict()) != initial_digest:
        raise RuntimeError("Refit26 não reinicializou os pesos exatamente")
    outer_scalers = fit_scalers(inputs, train, arm)
    outer_weights = EXP25.class_weights(meta, train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    for epoch in range(1, best_epoch+1):
        loss = train_epoch(model, inputs, meta, train, outer_scalers, optimizer, outer_weights, device, options, random_seed, epoch)
        history.append({"phase":"refit", "epoch":epoch, "train_loss":loss, "validation_loss":np.nan})
    train_score = predict_records(model, inputs, train, outer_scalers, device, options["prediction_batch"])
    test_score = predict_records(model, inputs, test, outer_scalers, device, options["prediction_batch"])
    selected = {"arm":arm, "seed":int(seed), "fold":int(fold), "random_seed":random_seed,
                "selected_epoch":int(best_epoch), "selection_epochs_run":int(sum(row["phase"]=="selecao" for row in history)),
                "best_validation_loss":float(best_loss), "selection_class_weights":weights.tolist(),
                "outer_class_weights":outer_weights.tolist(), "initial_state_sha256":initial_digest,
                "n_parameters":sum(p.numel() for p in model.parameters()), "seconds":time.monotonic()-started,
                "uses_wavelet":uses_wavelet(arm), "encoder_frozen":True}
    arrays = {f"{phase}__{modality}__{key}":value for phase, scalers in (("selection",selection_scalers),("outer",outer_scalers))
              for modality, scaler in scalers.items() for key,value in scaler.items()}
    # Aliases do ramo wav permitem reutilizar o validador/checkpoint comum25.
    arrays.update({f"{phase}__{key}":value for phase, scalers in (("selection",selection_scalers),("outer",outer_scalers)) for key,value in scalers["wav"].items()})
    partition = []
    for role, rows in (("treino_interno",fit),("validacao_interna",validation),("teste_externo",test)):
        for row in meta.iloc[rows].drop_duplicates("participant_id").itertuples():
            partition.append({"seed":seed,"fold":fold,"arm":arm,"role":role,"participant_id":row.participant_id,"y":row.y})
    frames = EXP25.prediction_frames(meta,train,test,train_score,test_score,arm,seed,fold)
    frames["historico.csv"], frames["particoes_internas.csv"] = pd.DataFrame(history), pd.DataFrame(partition)
    result = {"frames":frames,"arrays":arrays,"selection":selected,"selection_state":best_state,
              "state":{key:value.detach().cpu().clone() for key,value in model.state_dict().items()}}
    log(f"26 seed={seed} fold={fold} {arm}: época={best_epoch}; BA treino={EXP17.measures(frames['predicoes_treino_participante.csv'])['BA']:.6f}; BA teste={EXP17.measures(frames['predicoes.csv'])['BA']:.6f}; segundos={selected['seconds']:.2f}")
    return result


def validate_result(result, meta, train, test, seed, fold, arm, options=DEFAULT_OPTIONS):
    EXP25.validate_result(result,meta,train,test,seed,fold,arm,options)
    fit, _ = EXP25.internal_split(meta,np.asarray(train),seed,fold)
    for phase, expected_rows in (("selection",fit),("outer",train)):
        modalities = ("wav","rms") if uses_wavelet(arm) else ("wav",)
        for modality in modalities:
            prefix = f"{phase}__{modality}__"
            if not np.array_equal(result["arrays"][prefix+"train_rows"],expected_rows):
                raise ValueError("Normalizador26 foi ajustado em linhas incorretas")
            mean,scale=result["arrays"][prefix+"mean"],result["arrays"][prefix+"scale"]
            width=2048 if modality=="wav" else EXP25.WIDTH
            if mean.shape!=(width,) or scale.shape!=(width,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
                raise ValueError("Normalizador26 fora do contrato")
        for key in ("mean","scale","train_rows","n_frames"):
            np.testing.assert_array_equal(result["arrays"][f"{phase}__{key}"],result["arrays"][f"{phase}__wav__{key}"])
    if result["selection"]["uses_wavelet"] != uses_wavelet(arm) or result["selection"]["encoder_frozen"] is not True:
        raise ValueError("Modalidade/congelamento incompatível")
    probe=model_for(arm,options)
    probe.load_state_dict(result["state"],strict=True)
    probe.load_state_dict(result["selection_state"],strict=True)
    if any(not torch.isfinite(value).all() for value in result["state"].values()):
        raise ValueError("Pesos26 não finitos")


def reference25(folder,meta,seeds):
    exports=json.loads((folder/"ARTEFATOS_COMPLETOS.json").read_text(encoding="utf8"))
    if not {"predicoes_oof_participante.csv","assinatura_execucao.json","particoes_participantes.csv"}.issubset(exports):
        raise ValueError("Referências25 completas necessárias antes de concluir26")
    for name,digest in exports.items():
        if Path(name).name!=name or EXP17.sha(folder/name)!=digest:
            raise ValueError("Referência25 alterada")
    result=pd.read_csv(folder/"predicoes_oof_participante.csv",float_precision="round_trip")
    labels=meta.drop_duplicates("participant_id").set_index("participant_id").y
    result=result[result.arm.isin(("B_CNN_RMS","D_TCN_RMS")) & result.seed.isin(seeds)].copy()
    for seed in seeds:
        for arm in ("B_CNN_RMS","D_TCN_RMS"):
            sub=result[(result.seed==seed)&(result.arm==arm)]
            if sub.participant_id.duplicated().any() or set(sub.participant_id)!=set(labels.index) or not np.array_equal(sub.y,sub.participant_id.map(labels)):
                raise ValueError("Referência25 incompleta ou com rótulos incorretos")
            for fold,(_,test) in enumerate(EXP13.row_splits(meta,5,seed),1):
                if set(sub[sub.fold==fold].participant_id)!=set(meta.iloc[test].participant_id):
                    raise ValueError("Folds26 diferem de25")
    result["arm"]="REF25_"+result.arm
    return result


def identity(root,data23,data20,ref25,seeds,arms,options,device):
    paths=[data23/name for name in ("sequencias_wpt.npz","features_wpt_rms.npz","metadata_wpt_rms.csv","ARTEFATOS_COMPLETOS.json","assinatura_execucao.json")]
    paths += [data20/name for name in ("features_wav2vec_xlsr300m.npz","metadata_wav2vec.csv","ARTEFATOS_COMPLETOS.json","assinatura_execucao.json")]
    paths += [ref25/name for name in ("predicoes_oof_participante.csv","particoes_participantes.csv","ARTEFATOS_COMPLETOS.json","assinatura_execucao.json")]
    code=[Path(__file__).resolve(),HERE/"25_treinar_wavelet_cnn_tcn_avfad.py",HERE/"23_extrair_wpt_rms_sequencias_avfad.py",
          HERE/"21_avaliar_wpt_wav2vec_avfad.py",HERE/"19_avaliar_wpt_16khz_avfad.py",HERE/"17_robustez_distribuicao_wpt_avfad.py",HERE/"13_otimizar_wpt_sym5.py"]
    environment={"python":sys.version,"executable":sys.executable,"platform":platform.platform()}
    for package in ("numpy","pandas","scipy","scikit-learn","torch"):
        environment[package]=importlib.metadata.version(package)
    EXP25.seed_everything(26)
    counts={arm:sum(p.numel() for p in model_for(arm,options).parameters()) for arm in ARMS}
    if not counts[ARMS[1]]>max(counts[ARMS[2]],counts[ARMS[3]]):
        raise ValueError("Controle B não excede o orçamento de cada híbrido")
    return {"version":VERSION,"configuration":{"arms":list(arms),"seeds":list(seeds),"options":options,"device":device,
            "model_revision":MODEL_REVISION,"encoder_frozen":True,"parameters":counts,"primary":"C-A","threshold":0.,
            "loss":"class_weighted_BCE_after_person_mean_logits","epochs":"inner_loss_then_reinitialize_refit_outer"},
            "environment":environment,"input_files":{str(p.resolve()):EXP17.sha(p) for p in paths},
            "code":{str(p.resolve()):EXP17.sha(p) for p in code},"protocol_sha256":hashlib.sha256(protocol(options).encode("utf8")).hexdigest()}


def consolidate(out,meta,seeds,arms,reference,log):
    destinations=[out/"checkpoints"/f"seed_{seed}_fold_{fold:02d}"/arm for seed in seeds for fold in range(1,6) for arm in arms]
    raw=pd.concat([pd.read_csv(p/"predicoes.csv",float_precision="round_trip") for p in destinations],ignore_index=True)
    folds=pd.concat([pd.read_csv(p/"metricas.csv",float_precision="round_trip") for p in destinations],ignore_index=True)
    histories=pd.concat([pd.read_csv(p/"historico.csv").assign(arm=p.name,seed=int(p.parent.name.split('_')[1]),fold=int(p.parent.name.split('_')[-1])) for p in destinations],ignore_index=True)
    choices=pd.DataFrame([json.loads((p/"selecao_epocas.json").read_text(encoding="utf8")) for p in destinations])
    summary=[]
    for (seed,arm),group in raw.groupby(["seed","arm"]):
        if group.participant_id.duplicated().any() or len(group)!=meta.participant_id.nunique(): raise ValueError("OOF26 incompleto")
        measured=EXP17.measures(group);measured["AUC_OOF_exploratoria"]=measured.pop("AUC")
        f=folds[(folds.seed==seed)&(folds.arm==arm)&(folds.phase=="teste")&(folds.unit=="participante")]
        measured["AUC_media_folds_ponderada_pares"]=float(np.average(f.AUC,weights=(f.TN+f.FP)*(f.TP+f.FN)))
        summary.append({"seed":seed,"arm":arm,**measured})
    summary=pd.DataFrame(summary);joined=pd.concat([raw,reference],ignore_index=True)
    scopes=[("principal_seed20261105",[20261105])] if 20261105 in seeds else []
    secondary=[s for s in seeds if s!=20261105]
    if secondary: scopes.append(("estabilidade_secundaria",secondary))
    contrasts={**BASE_CONTRASTS,**REFERENCE_CONTRASTS};comparisons=[]
    available=set(joined.arm)
    for scope,chosen in scopes:
        for name,coefficients in contrasts.items():
            if set(coefficients).issubset(available): comparisons.append({"escopo":scope,"comparacao":name,**EXP17.intervals(joined,chosen,coefficients,iterations=2000)})
    outputs={"predicoes_oof_participante.csv":raw,"metricas_por_fold.csv":folds,"metricas_por_seed.csv":summary,
             "historico_todos.csv":histories,"selecoes_epocas.csv":choices,"comparacoes_pareadas.csv":pd.DataFrame(comparisons),"referencia25_oof.csv":reference}
    for name,frame in outputs.items(): EXP17.csv_write(out/name,frame)
    lines=["EXPERIMENTO26 — HÍBRIDOS WAVELET TEMPORAL + XLS-R",f"Consolidado:{EXP17.now()}",
           "Todos os folds solicitados concluídos; seleção de época somente interna.",
           "Ciclo completo de quatro braços/seis sementes." if tuple(arms)==ARMS and tuple(seeds)==SEEDS else "Ciclo parcial/diferente do protocolo.",""]
    for row in summary.itertuples(): lines.append(f"seed={row.seed} {row.arm}: BA={row.BA:.6f}; sens={row.sensibilidade:.6f}; esp={row.especificidade:.6f}; acc={row.acuracia:.6f}; AUCfold={row.AUC_media_folds_ponderada_pares:.6f}")
    lines += ["","CONTRASTES PAREADOS — IC95 CONDICIONAL"]
    for row in comparisons: lines.append(f"{row['escopo']} {row['comparacao']}: ΔBA={row['delta_BA']:+.6f}; IC95=[{row['IC95_inferior']:+.6f}; {row['IC95_superior']:+.6f}]")
    if secondary:
        lines += ["","ESTABILIDADE NAS CINCO SEMENTES SECUNDÁRIAS"]
        for arm,group in summary[summary.seed.isin(secondary)].groupby("arm"): lines.append(f"{arm}: BA média={group.BA.mean():.6f}; mínimo={group.BA.min():.6f}; máximo={group.BA.max():.6f}; DP={group.BA.std():.6f}")
    lines += ["","DIAGNÓSTICO DE TREINO"]
    for arm,group in folds[(folds.phase=="treino")&(folds.unit=="participante")].groupby("arm"): lines.append(f"{arm}: BA treino média={group.BA.mean():.6f}")
    lines += ["","LIMITAÇÕES","AVFAD explorada: desenvolvimento, não teste independente.",
              "B possui mais parâmetros que cada híbrido; capacidades e representações ainda diferem.",
              "RMS/CNN/TCN foram predefinidos, sem escolher vencedores do25; encoder20 permanece congelado.",
              "Interpolação128 comprime as sequências; pooling/logits não são probabilidades calibradas.",
              "ICs condicionais não refazem busca/treino e não corrigem tentativas históricas.",
              "Ganhar sobre Wavelet isolada não demonstra contribuição Wavelet: examinar C-A/D-A e C-B/D-B.",
              "Nenhum dado SVD ou resultado anterior foi alterado; nenhuma conclusão clínica automática.\n"]
    EXP17.text_write(out/"RELATORIO_RESULTADOS.txt","\n".join(lines))
    names=list(outputs)+["RELATORIO_RESULTADOS.txt","assinatura_execucao.json","particoes_participantes.csv","PROTOCOLO.txt"]
    EXP17.json_write(out/"ARTEFATOS_COMPLETOS.json",{name:EXP17.sha(out/name) for name in names})
    log("Consolidação26 concluída; todos os braços e referências preservados.")


def selftest():
    import unittest
    spec=importlib.util.spec_from_file_location("test_wavelet_hibrido26",HERE/"test_wavelet_hibrido26.py")
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful(): raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dados",type=Path,default=ROOT);parser.add_argument("--dados23",type=Path);parser.add_argument("--dados20",type=Path)
    parser.add_argument("--referencia25",type=Path);parser.add_argument("--saida",type=Path,default=ROOT/OUT_REL)
    parser.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS));parser.add_argument("--arms",choices=ARMS,nargs="+",default=list(ARMS))
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda");parser.add_argument("--max-epochs",type=int,default=DEFAULT_OPTIONS["max_epochs"])
    action=parser.add_mutually_exclusive_group();action.add_argument("--verificar",action="store_true");action.add_argument("--executar",action="store_true");action.add_argument("--autoteste",action="store_true")
    args=parser.parse_args()
    if args.autoteste:selftest();return
    root,out,seeds,arms=args.dados.resolve(),args.saida.resolve(),tuple(args.seeds),tuple(args.arms)
    if not out.is_relative_to(root/"results/experiments") or out==root/"results/experiments":parser.error("Saída deve ser subpasta de results/experiments")
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 or s>2**32-1024 for s in seeds):parser.error("Sementes únicas e válidas obrigatórias")
    if len(set(arms))!=len(arms) or args.max_epochs<DEFAULT_OPTIONS["min_epochs"]:parser.error("Braços únicos e épocas>=5")
    options={**DEFAULT_OPTIONS,"max_epochs":args.max_epochs}
    data23=(args.dados23 or root/DATA23_REL).resolve();data20=(args.dados20 or root/DATA20_REL).resolve();ref25=(args.referencia25 or root/REF25_REL).resolve()
    inputs,meta,_=load_inputs(root,data23,data20,materialize=args.executar,options=options)
    reference=reference25(ref25,meta,seeds)
    EXP25.configure_torch(args.device)
    print(f"ENTRADAS26 OK:411 pessoas/2465 gravações; braços={arms}; seeds={seeds}",flush=True)
    if not args.executar:print("Somente verificação, sem ajuste ou escrita.",flush=True);return
    ident=identity(root,data23,data20,ref25,seeds,arms,options,args.device)
    signature_file=out/"assinatura_execucao.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf8"))!=ident:raise RuntimeError("Assinatura26 mudou; preservar pasta")
    else:
        if out.exists() and any(p.name!="PROTOCOLO.txt" for p in out.iterdir()):raise RuntimeError("Pasta sem assinatura contém artefatos")
        out.mkdir(parents=True,exist_ok=True);EXP17.freeze_text(out/"PROTOCOLO.txt",protocol(options));EXP17.json_write(signature_file,ident)
    EXP17.freeze_text(out/"PROTOCOLO.txt",protocol(options));EXP17.freeze_text(out/"particoes_participantes.csv",EXP19.fold_table(meta,seeds).to_csv(index=False,lineterminator="\n"))
    signature=hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest();log=EXP17.Logger(out/"LOG_EXECUCAO.txt")
    log(f"Início/retomada26; encoder congelado; máximo{options['max_epochs']} épocas; sem escolha por teste.")
    for seed in seeds:
        for fold,(train,test) in enumerate(EXP13.row_splits(meta,5,seed),1):
            for arm in arms:
                destination=out/"checkpoints"/f"seed_{seed}_fold_{fold:02d}"/arm
                if (destination/"CONCLUIDO.json").exists():
                    result=load_checkpoint(destination,signature);validate_result(result,meta,train,test,seed,fold,arm,options)
                    log(f"Checkpoint26 confirmado:seed={seed} fold={fold} {arm}; nenhum reajuste");continue
                if destination.exists() and any(destination.iterdir()):raise RuntimeError(f"Checkpoint parcial preservado:{destination}")
                result=train_arm(inputs,meta,train,test,seed,fold,arm,args.device,options,log);validate_result(result,meta,train,test,seed,fold,arm,options)
                save_checkpoint(destination,result,signature)
    consolidate(out,meta,seeds,arms,reference,log)


if __name__=="__main__":main()
