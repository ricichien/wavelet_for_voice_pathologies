# -*- coding: utf-8 -*-
"""Auditoria posterior25/26: não treina modelos e não modifica resultados.

Exige execuções completas. Recalcula escalas, partições, métricas e ICs;
reproduz por inferência a semente principal/fold1 de todos os braços.
Executar somente depois dos treinamentos, preferencialmente em CPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = importlib.util.spec_from_file_location("audit_exp26", HERE / "26_hibrido_wavelet_temporal_wav2vec.py")
EXP26 = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(EXP26)
EXP25 = EXP26.EXP25
REPLAY_ATOL = 5e-5
REPLAY_RTOL = 5e-5


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def splits(meta, seed):
    people = meta[["participant_id", "y"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True)
    require(not people.participant_id.duplicated().any(), "Rótulos inconsistentes por participante")
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(people)), people.y):
        yield (np.flatnonzero(meta.participant_id.isin(people.iloc[train].participant_id)),
               np.flatnonzero(meta.participant_id.isin(people.iloc[test].participant_id)))


def internal(meta, train, seed, fold):
    fit, validation = next(splits(meta.iloc[train].reset_index(drop=True), int(seed) + 100 + int(fold)))
    return np.asarray(train)[fit], np.asarray(train)[validation]


def independently_scaled(values, rows):
    """Duas passagens: soma/média e desvios; distinto do Welford do treino."""
    width = values.shape[1]
    count = len(rows) * (values.shape[2] if values.ndim == 3 else 1)
    total = np.zeros(width, np.float64)
    def blocks():
        for start in range(0, len(rows), 32):
            block = np.asarray(values[rows[start:start+32]], np.float64)
            yield block.transpose(0, 2, 1).reshape(-1, width) if values.ndim == 3 else block
    for block in blocks():
        require(np.isfinite(block).all(), "Atributos não finitos no normalizador")
        total += block.sum(axis=0)
    mean = total / count
    squared = np.zeros(width, np.float64)
    for block in blocks():
        squared += np.square(block - mean).sum(axis=0)
    scale = np.sqrt(squared / count)
    scale[scale < 1e-8] = 1.
    return {"mean": mean, "scale": scale, "train_rows": np.asarray(rows, np.int64), "n_frames": np.asarray(count)}


def metrics(frame):
    y, pred, score = frame.y.to_numpy(int), frame.predicao.to_numpy(int), frame.score.to_numpy(float)
    require(set(np.unique(y)) == {0, 1} and np.isfinite(score).all(), "Classes/scores incorretos")
    np.testing.assert_array_equal(pred, (score >= 0).astype(int))
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    fn, tp = int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())
    return {"n": len(y), "BA": .5*(tn/(tn+fp)+tp/(tp+fn)), "acuracia": (tn+tp)/len(y),
            "sensibilidade": tp/(tp+fn), "especificidade": tn/(tn+fp), "AUC": roc_auc_score(y,score),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def bootstrap(raw, seeds, coefficients):
    """Unidade de reamostragem = pessoa; seeds da pessoa permanecem juntas."""
    relevant = raw[raw.seed.isin(seeds)]
    ids = sorted(relevant.participant_id.unique())
    labels = relevant.drop_duplicates("participant_id").set_index("participant_id").loc[ids].y.to_numpy()
    repeated = []
    for seed in seeds:
        change = np.zeros(len(ids), np.float64)
        for arm, coefficient in coefficients.items():
            frame = relevant[(relevant.seed == seed) & (relevant.arm == arm)].set_index("participant_id")
            require(len(frame) == len(ids) and not frame.index.duplicated().any(), "Contraste sem OOF completo")
            frame = frame.loc[ids]
            np.testing.assert_array_equal(frame.y, labels)
            change += coefficient * (frame.predicao.to_numpy() == labels)
        repeated.append(change)
    change = np.mean(repeated, axis=0)
    normal, pathological = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    rng = np.random.default_rng(20261105)
    bn = rng.choice(normal, (2000, len(normal)), replace=True)
    bp = rng.choice(pathological, (2000, len(pathological)), replace=True)
    boot = .5 * (change[bn].mean(axis=1) + change[bp].mean(axis=1))
    lo, hi = np.quantile(boot, [.025, .975])
    return {"delta_BA": .5*(change[normal].mean()+change[pathological].mean()),
            "IC95_inferior": lo, "IC95_superior": hi}


def compare_numbers(actual, expected, atol=1e-12):
    for key, value in actual.items():
        require(np.isclose(value, expected[key], atol=atol, rtol=0), f"Valor incorreto em {key}: {value} / {expected[key]}")


def check_history(history, selected, options):
    selection = history[history.phase == "selecao"]
    refit = history[history.phase == "refit"]
    require(set(history.phase) == {"selecao", "refit"}, "Fase de história desconhecida")
    np.testing.assert_array_equal(selection.epoch, np.arange(1,len(selection)+1))
    require(options["min_epochs"] <= len(selection) <= options["max_epochs"], "História de seleção fora dos limites")
    require(np.isfinite(selection[["train_loss","validation_loss"]]).all().all(), "Loss de seleção não finita")
    require(np.isfinite(refit.train_loss).all() and refit.validation_loss.isna().all(), "História refit inválida")
    best_epoch, best_loss, stale = None, float("inf"), 0
    for row in selection.itertuples():
        if row.epoch >= options["min_epochs"]:
            if row.validation_loss < best_loss - options["min_delta"]:
                best_epoch, best_loss, stale = int(row.epoch), row.validation_loss, 0
            else:
                stale += 1
        require(stale < options["patience"] or row.epoch == len(selection), "Treinou após patience esgotada")
    require(len(selection) == options["max_epochs"] or stale >= options["patience"], "Parou antes da regra de stopping")
    require(best_epoch == selected["selected_epoch"] and len(selection) == selected["selection_epochs_run"], "Épocas divergentes")
    require(abs(best_loss - selected["best_validation_loss"]) < 1e-12, "Loss selecionada diverge do histórico")
    np.testing.assert_array_equal(refit.epoch, np.arange(1,best_epoch+1))


class Auditor:
    def __init__(self, root, replay_device="cpu"):
        self.root, self.device = Path(root).resolve(), replay_device
        self.hashes, self.scaler_cache = {}, {}
        self.totals = {"checkpoint_files_hashed":0,"checkpoints":0,"normalizers_checked":0,
                       "normalizers_independently_recomputed":0,"replayed_models":0,
                       "replayed_recordings":0,"max_replay_abs_error":0.,"replayed_validation_losses":0,
                       "OOF_rows":0,"paired_intervals":0}
        self.inputs, self.meta = None, None

    def digest(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            self.hashes[path] = sha(path)
        return self.hashes[path]

    def verify_map(self, folder, mapping):
        for name, digest in mapping.items():
            require(Path(name).name == name, "Caminho inseguro no manifesto")
            require(self.digest(folder/name) == digest, f"Hash divergente: {folder/name}")

    def load_data(self):
        if self.inputs is not None:
            return
        # Usa memmaps já assinados; não gera um cache novo na auditoria.
        cache = self.root / "checkpoints/cache_25_interpolacao"
        require(all((cache/name).is_file() for name in ("assinatura_cache.json","raw_128.npy","raw_128.json","rms_128.npy","rms_128.json")),
                "Auditoria exige cache25 completo existente; não cria cache")
        inputs, meta, _ = EXP25.load_inputs(self.root, materialize=True)
        self.inputs, self.meta = inputs, meta

    def add_embeddings(self):
        if "wav" in self.inputs:
            return
        folder = self.root / EXP26.DATA20_REL
        self.verify_map(folder,read_json(folder/"ARTEFATOS_COMPLETOS.json"))
        meta = pd.read_csv(folder/"metadata_wav2vec.csv",dtype={"sample_id":str,"participant_id":str,"task":str},float_precision="round_trip")
        meta["task"] = meta.task.str.zfill(3)
        with np.load(folder/"features_wav2vec_xlsr300m.npz",allow_pickle=False) as z:
            wav = np.asarray(z["embeddings"],np.float32)
            embedded = {key:z[key].astype(str) for key in ("sample_id","participant_id","task")}
            names = z["feature_names"].astype(str)
        EXP26.validate_embedding_alignment(wav,meta,embedded,names,self.meta,read_json(folder/"assinatura_execucao.json"))
        self.inputs["wav"] = wav

    def check_normalizer(self, values, rows, arrays, prefix, modality):
        key = (modality,tuple(np.asarray(rows,int)))
        if key not in self.scaler_cache:
            self.scaler_cache[key] = independently_scaled(values,rows)
            self.totals["normalizers_independently_recomputed"] += 1
        expected = self.scaler_cache[key]
        for name in ("train_rows","n_frames"):
            np.testing.assert_array_equal(arrays[prefix+name],expected[name])
        for name in ("mean","scale"):
            np.testing.assert_allclose(arrays[prefix+name],expected[name],atol=1e-10,rtol=1e-9)
        self.totals["normalizers_checked"] += 1

    def check_predictions(self, result, train, test, seed, fold, arm):
        frames, meta = result["frames"], self.meta
        for rows,person_name,record_name in ((train,"predicoes_treino_participante.csv","predicoes_treino_gravacao.csv"),
                                            (test,"predicoes.csv","predicoes_teste_gravacao.csv")):
            people, records = frames[person_name],frames[record_name]
            expected_meta = meta.iloc[rows]
            require(not records.sample_id.duplicated().any() and set(records.sample_id)==set(expected_meta.sample_id),"Gravações previstas incorretas")
            require(not people.participant_id.duplicated().any() and set(people.participant_id)==set(expected_meta.participant_id),"Pessoas previstas incorretas")
            for frame,key in ((records,"sample_id"),(people,"participant_id")):
                labels = expected_meta.drop_duplicates(key).set_index(key).y
                np.testing.assert_array_equal(frame.y,frame[key].map(labels))
                require(frame.arm.eq(arm).all() and frame.seed.eq(seed).all() and frame.fold.eq(fold).all(),"Metadados da previsão incorretos")
                metrics(frame)
            aggregate = records.groupby("participant_id").score.agg(["mean","count"]).sort_index()
            observed = people.set_index("participant_id").sort_index()
            np.testing.assert_allclose(observed.score,aggregate["mean"],atol=1e-12,rtol=0)
            np.testing.assert_array_equal(observed.n_gravacoes,aggregate["count"])
        for row in frames["metricas.csv"].to_dict("records"):
            name = {("treino","participante"):"predicoes_treino_participante.csv",("treino","gravacao"):"predicoes_treino_gravacao.csv",
                    ("teste","participante"):"predicoes.csv",("teste","gravacao"):"predicoes_teste_gravacao.csv"}[row["phase"],row["unit"]]
            compare_numbers(metrics(frames[name]),row)

    def replay(self, module, number, result, train, test, seed, fold, arm, options):
        model = module.model_for(arm,options).to(self.device)
        model.load_state_dict(result["state"],strict=True)
        model.eval()
        arrays = result["arrays"]
        def scalers(phase):
            if number == 25:
                return {key:arrays[f"{phase}__{key}"] for key in ("mean","scale","n_frames","train_rows")}
            modalities = ("wav","rms") if EXP26.uses_wavelet(arm) else ("wav",)
            return {mod:{key:arrays[f"{phase}__{mod}__{key}"] for key in ("mean","scale","n_frames","train_rows")} for mod in modalities}
        values = self.inputs[EXP25.ARM_INPUT[arm]] if number == 25 else self.inputs
        scores = module.predict_records(model,values,test,scalers("outer"),self.device,options["prediction_batch"])
        exported = result["frames"]["predicoes_teste_gravacao.csv"].set_index("sample_id").loc[self.meta.iloc[test].sample_id].score.to_numpy()
        np.testing.assert_allclose(scores,exported,atol=REPLAY_ATOL,rtol=REPLAY_RTOL)
        np.testing.assert_array_equal(scores>=0,exported>=0)
        self.totals["max_replay_abs_error"] = max(self.totals["max_replay_abs_error"],float(np.max(np.abs(scores-exported))))
        self.totals["replayed_models"] += 1
        self.totals["replayed_recordings"] += len(scores)
        # Reproduzir também loss da época escolhida, sem atualizar pesos.
        model.load_state_dict(result["selection_state"],strict=True)
        _,validation = internal(self.meta,train,seed,fold)
        scores = module.predict_records(model,values,validation,scalers("selection"),self.device,options["prediction_batch"])
        frame = self.meta.iloc[validation][["participant_id","y"]].copy();frame["score"]=scores
        pooled = frame.groupby("participant_id").agg(y=("y","first"),score=("score","mean"))
        y = pooled.y.to_numpy(int);logits=pooled.score.to_numpy()
        loss = np.mean((np.logaddexp(0.,logits)-y*logits)*np.asarray(result["selection"]["selection_class_weights"])[y])
        require(np.isclose(loss,result["selection"]["best_validation_loss"],atol=2e-6,rtol=2e-6),"Loss da época selecionada não reproduz")
        self.totals["replayed_validation_losses"] += 1

    def audit(self, number):
        module = EXP25 if number == 25 else EXP26
        out = self.root/module.OUT_REL
        require((out/"ARTEFATOS_COMPLETOS.json").is_file(),f"Experimento {number} ainda não completo; auditoria não executada")
        exports = read_json(out/"ARTEFATOS_COMPLETOS.json")
        required = {"predicoes_oof_participante.csv","metricas_por_fold.csv","metricas_por_seed.csv","historico_todos.csv",
                    "selecoes_epocas.csv","comparacoes_pareadas.csv","RELATORIO_RESULTADOS.txt","assinatura_execucao.json","PROTOCOLO.txt","particoes_participantes.csv"}
        if number == 26: required.add("referencia25_oof.csv")
        require(set(exports)==required,"Lista final de artefatos inesperada")
        self.verify_map(out,exports)
        identity = read_json(out/"assinatura_execucao.json")
        for category in ("code","inputs" if number==25 else "input_files"):
            for name,digest in identity[category].items():
                require(self.digest(name)==digest,f"Fonte/insumo alterado:{name}")
        require(hashlib.sha256((out/"PROTOCOLO.txt").read_text(encoding="utf8").encode()).hexdigest()==identity["protocol_sha256"],"Protocolo alterado")
        configuration=identity["configuration"]
        seeds,arms=tuple(configuration["seeds"]),tuple(configuration["arms"])
        require(seeds==module.SEEDS and arms==module.ARMS,"Auditoria final exige seis seeds e todos os braços predefinidos")
        options = {key:configuration[key] for key in module.DEFAULT_OPTIONS} if number==25 else configuration["options"]
        require((out/"PROTOCOLO.txt").read_text(encoding="utf8")==module.protocol(options),"Texto do protocolo difere do código congelado")
        signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.load_data()
        if number==26:self.add_embeddings()
        meta=self.meta
        expected_count=len(seeds)*5*len(arms)
        require(len(list((out/"checkpoints").rglob("CONCLUIDO.json")))==expected_count,"Número de checkpoints diferente do protocolo")
        partitions,all_predictions,all_metrics,all_history,all_choices=[],[],[],[],[]
        for seed in seeds:
            for fold,(train,test) in enumerate(splits(meta,seed),1):
                fit,validation=internal(meta,train,seed,fold)
                role_rows={"treino_interno":fit,"validacao_interna":validation,"teste_externo":test}
                for role,rows in (("treino",train),("teste",test)):
                    for row in meta.iloc[rows].drop_duplicates("participant_id").itertuples():
                        partitions.append({"seed":seed,"fold":fold,"role":role,"participant_id":row.participant_id,"y":row.y})
                for arm in arms:
                    folder=out/"checkpoints"/f"seed_{seed}_fold_{fold:02d}"/arm
                    marker=read_json(folder/"CONCLUIDO.json")
                    require(marker["run_signature"]==signature and set(marker["sha256"])==EXP25.ARTIFACTS,"Assinatura/artefatos do checkpoint incorretos")
                    self.verify_map(folder,marker["sha256"])
                    self.totals["checkpoint_files_hashed"]+=len(marker["sha256"])
                    result=module.load_checkpoint(folder,signature)
                    module.validate_result(result,meta,train,test,seed,fold,arm,options)
                    arrays,selected,frames=result["arrays"],result["selection"],result["frames"]
                    check_history(frames["historico.csv"],selected,options)
                    for role,rows in role_rows.items():
                        actual=frames["particoes_internas.csv"];actual=actual[actual.role==role]
                        require(not actual.participant_id.duplicated().any() and set(actual.participant_id)==set(meta.iloc[rows].participant_id),"Partição interna/externa divergente")
                        labels=meta.iloc[rows].drop_duplicates("participant_id").set_index("participant_id").y
                        np.testing.assert_array_equal(actual.y,actual.participant_id.map(labels))
                    for phase,rows in (("selection",fit),("outer",train)):
                        if number==25:
                            modality=EXP25.ARM_INPUT[arm]
                            self.check_normalizer(self.inputs[modality],rows,arrays,f"{phase}__",modality)
                        else:
                            for modality in (("wav","rms") if EXP26.uses_wavelet(arm) else ("wav",)):
                                self.check_normalizer(self.inputs[modality],rows,arrays,f"{phase}__{modality}__",modality)
                        labels=meta.iloc[rows].drop_duplicates("participant_id").y
                        weights=np.asarray([len(labels)/(2*(labels==y).sum()) for y in (0,1)])
                        np.testing.assert_array_equal(selected[phase+"_class_weights"],weights)
                    self.check_predictions(result,train,test,seed,fold,arm)
                    # Todas as redes: formato e finitude dos dois estados salvos.
                    expected_seed=module.arm_seed(seed,fold,arm)
                    require(selected["random_seed"]==expected_seed,"Seed de inicialização incorreta")
                    EXP25.seed_everything(expected_seed)
                    model=module.model_for(arm,options)
                    initial_digest=hashlib.sha256()
                    for key,value in sorted(model.state_dict().items()):
                        initial_digest.update(key.encode());initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
                    require(initial_digest.hexdigest()==selected["initial_state_sha256"],"Estado inicial não corresponde à seed declarada")
                    for key in ("state","selection_state"):
                        model.load_state_dict(result[key],strict=True)
                        require(all(torch.isfinite(value).all() for value in result[key].values()),"Estado de rede não finito")
                    require(sum(p.numel() for p in model.parameters())==selected["n_parameters"],"Contagem de parâmetros incorreta")
                    if seed==20261105 and fold==1:
                        self.replay(module,number,result,train,test,seed,fold,arm,options)
                    all_predictions.append(frames["predicoes.csv"])
                    all_metrics.append(frames["metricas.csv"])
                    all_history.append(frames["historico.csv"].assign(arm=arm,seed=seed,fold=fold))
                    all_choices.append(selected)
                    self.totals["checkpoints"]+=1
                print(f"Auditoria{number}:seed={seed} fold={fold} íntegro",flush=True)
        pd.testing.assert_frame_equal(read_csv(out/"particoes_participantes.csv"),pd.DataFrame(partitions))
        choices_table=pd.DataFrame(all_choices)
        # CSV representa listas JSON como texto; os checkpoints preservam listas.
        for column in ("selection_class_weights","outer_class_weights"):
            choices_table[column]=choices_table[column].map(str)
        for name,expected in (("predicoes_oof_participante.csv",pd.concat(all_predictions,ignore_index=True)),
                              ("metricas_por_fold.csv",pd.concat(all_metrics,ignore_index=True)),
                              ("historico_todos.csv",pd.concat(all_history,ignore_index=True)),
                              ("selecoes_epocas.csv",choices_table)):
            pd.testing.assert_frame_equal(read_csv(out/name),expected)
        raw=read_csv(out/"predicoes_oof_participante.csv")
        require(not raw.duplicated(["seed","arm","participant_id"]).any(),"OOF duplicado")
        summary=read_csv(out/"metricas_por_seed.csv")
        folds=read_csv(out/"metricas_por_fold.csv")
        for row in summary.to_dict("records"):
            frame=raw[(raw.seed==row["seed"])&(raw.arm==row["arm"])]
            require(set(frame.participant_id)==set(meta.participant_id),"OOF incompleto")
            measured=metrics(frame);measured["AUC_OOF_exploratoria"]=measured.pop("AUC")
            relevant=folds[(folds.seed==row["seed"])&(folds.arm==row["arm"])&(folds.phase=="teste")&(folds.unit=="participante")]
            measured["AUC_media_folds_ponderada_pares"]=np.average(relevant.AUC,weights=(relevant.TN+relevant.FP)*(relevant.TP+relevant.FN))
            compare_numbers(measured,row)
        contrasts=module.CONTRASTS if number==25 else {**EXP26.BASE_CONTRASTS,**EXP26.REFERENCE_CONTRASTS}
        joined=raw
        if number==26:
            reference=read_csv(out/"referencia25_oof.csv")
            original=read_csv(self.root/EXP25.OUT_REL/"predicoes_oof_participante.csv")
            original=original[original.arm.isin(("B_CNN_RMS","D_TCN_RMS")) & original.seed.isin(seeds)].copy()
            original["arm"]="REF25_"+original.arm
            pd.testing.assert_frame_equal(reference.reset_index(drop=True),original.reset_index(drop=True))
            joined=pd.concat([raw,reference],ignore_index=True)
        comparisons=read_csv(out/"comparacoes_pareadas.csv")
        expected_keys={(scope,name) for scope in ("principal_seed20261105","estabilidade_secundaria") for name in contrasts}
        require(set(zip(comparisons.escopo,comparisons.comparacao))==expected_keys and len(comparisons)==len(expected_keys),"Lista de contrastes incorreta")
        for row in comparisons.to_dict("records"):
            chosen=[20261105] if row["escopo"]=="principal_seed20261105" else [s for s in seeds if s!=20261105]
            compare_numbers(bootstrap(joined,chosen,contrasts[row["comparacao"]]),row)
            self.totals["paired_intervals"]+=1
        self.totals["OOF_rows"]+=len(raw)
        return {"experiment":number,"checkpoints":expected_count,"OOF_rows":len(raw),"arms":list(arms),"seeds":list(seeds)}


def report_text(experiments, totals, seconds):
    return "\n".join([
        "AUDITORIA POSTERIOR — REDES WAVELET 25/26",
        f"Executada em: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Experimentos completos auditados: {', '.join(str(e['experiment']) for e in experiments)}.",
        f"Duração: {seconds:.2f} segundos.","",
        "VERIFICAÇÕES CONCLUÍDAS",
        f"Checkpoints: {totals['checkpoints']}; arquivos de checkpoints com SHA256: {totals['checkpoint_files_hashed']}.",
        "Todos os exports finais, fontes e insumos vinculados conferidos por SHA256.",
        "Partições externas e internas reproduzidas por participante; frases e rótulos conferidos.",
        f"Normalizadores verificados: {totals['normalizers_checked']}; estatísticas distintas recalculadas em duas passagens: {totals['normalizers_independently_recomputed']}.",
        "Médias/escalas apenas do treino correspondente; pesos de classe calculados por pessoa.",
        "Regras de época mínima, min_delta, patience e refit conferidas em todos os históricos.",
        "Todas as predições conferidas quanto a IDs, folds, limiar0 e agregação por pessoa.",
        f"OOFs por pessoa: {totals['OOF_rows']}; intervalos pareados recalculados: {totals['paired_intervals']}.",
        "Todas as métricas por fold/seed e tabelas consolidadas reproduzidas dos checkpoints.",
        "Estados selecionados/refit de todas as redes carregados estritamente e verificados finitos.","",
        "REPLAY POR INFERÊNCIA — AMOSTRAGEM PREDEFINIDA",
        "Somente a seed principal 20261105, fold externo1, de TODOS os braços de cada experimento.",
        f"Modelos reproduzidos: {totals['replayed_models']}; gravações externas: {totals['replayed_recordings']}.",
        f"Maior erro absoluto dos logits: {totals['max_replay_abs_error']:.12g}.",
        f"Tolerância declarada CPU/GPU: atol={REPLAY_ATOL}, rtol={REPLAY_RTOL}; decisões externas exigidas idênticas.",
        f"Loss da validação da época escolhida também reproduzida em {totals['replayed_validation_losses']} modelos.","",
        "AÇÕES E LIMITES",
        "Nenhum treinamento, extração de áudio, atualização de pesos ou alteração dos resultados.",
        "A escrita limita-se a este relatório e seu resumo JSON, depois de todos os checks aprovarem.",
        "Replay de logits/loss é amostral: não foi repetido para todas as 270 redes.",
        "Hashes, partições, escalas, épocas, métricas e OOFs foram verificados em TODOS os checkpoints solicitados.",
        "A auditoria verifica integridade e coerência computacional; não prova ausência de confundidores,",
        "generalização externa ou validade clínica. A AVFAD permanece uma coorte de desenvolvimento.",
        "Os ICs condicionais não refazem treino nem corrigem a exploração histórica da coorte.",""])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento",choices=("25","26","ambos"),default="ambos")
    parser.add_argument("--dados",type=Path,default=ROOT)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--relatorio",type=Path)
    parser.add_argument("--autoteste",action="store_true")
    args=parser.parse_args()
    if args.autoteste:
        import unittest
        spec=importlib.util.spec_from_file_location("audit_network_tests",HERE/"test_verificar_redes25_26.py")
        tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
        if not result.wasSuccessful():raise SystemExit(1)
        return
    numbers=(25,26) if args.experimento=="ambos" else (int(args.experimento),)
    # Todos os marcadores finais são exigidos antes de carregar caches/usar GPU.
    for number in numbers:
        module=EXP25 if number==25 else EXP26
        require((args.dados/module.OUT_REL/"ARTEFATOS_COMPLETOS.json").is_file(),f"Experimento {number} incompleto; aguardar sem interferir no treinamento")
    EXP25.configure_torch(args.device,threads=4)
    started=time.monotonic();auditor=Auditor(args.dados,args.device)
    experiments=[auditor.audit(number) for number in numbers]
    seconds=time.monotonic()-started
    report=args.relatorio or args.dados/"results/experiments/documentacao_wav2vec"/f"{'_'.join(map(str,numbers))}_VERIFICACAO_REDES.txt"
    report=report.resolve()
    require(report.is_relative_to(args.dados.resolve()/"results/experiments"),"Relatório deve ficar em results/experiments")
    report.parent.mkdir(parents=True,exist_ok=True)
    text=report_text(experiments,auditor.totals,seconds)
    report.write_text(text,encoding="utf8")
    report.with_suffix(".json").write_text(json.dumps({"experiments":experiments,"totals":auditor.totals,
        "seconds":seconds,"replay_device":args.device,"auditor_sha256":sha(Path(__file__))},ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(text,flush=True)


if __name__=="__main__":main()
