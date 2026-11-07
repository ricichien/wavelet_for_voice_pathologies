# -*- coding: utf-8 -*-
"""Auditoria30: m?dia de frases, vogal e classificadores por PESSOA.

N?o usa ?udio/GPU nem ajusta classificadores. Emite laudos s? ap?s export final.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
COHORT_COUNTS=(411,82,329)
SPEC=importlib.util.spec_from_file_location("exp30_independent_audit",HERE/"30_avaliar_frase_vogal_wavelet_avfad.py")
EXP30=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP30)

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")

def independent_splits(meta, n_splits, seed):
    people = meta[["participant_id", "y"]].drop_duplicates().sort_values("participant_id").reset_index(drop=True)
    require(not people.participant_id.duplicated().any(), "Pessoa com rótulos divergentes")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_person, test_person in splitter.split(np.zeros(len(people)), people.y.to_numpy()):
        train_ids = people.iloc[train_person].participant_id
        test_ids = people.iloc[test_person].participant_id
        require(not set(train_ids) & set(test_ids), "Pessoa compartilhada")
        yield np.flatnonzero(meta.participant_id.isin(train_ids)), np.flatnonzero(meta.participant_id.isin(test_ids))

def metrics(frame):
    y, prediction = frame.y.to_numpy(int), frame.predicao.to_numpy(int)
    require(set(np.unique(y)) == {0, 1}, "Métrica exige as duas classes")
    require(np.isfinite(frame.score).all(), "Margens não finitas")
    require(np.array_equal(prediction, (frame.score.to_numpy() >= 0).astype(int)), "Decisão diferente de margem>=0")
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    tp = int(((y == 1) & (prediction == 1)).sum())
    return {"n": len(frame), "BA": .5 * (tn/(tn+fp) + tp/(tp+fn)),
            "acuracia": (tn+tp)/len(frame), "sensibilidade": tp/(tp+fn), "especificidade": tn/(tn+fp),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp, "AUC": float(roc_auc_score(y, frame.score))}

def compare_numbers(actual, expected, tolerance=1e-12):
    for key, value in actual.items():
        require(np.isclose(value, getattr(expected, key), atol=tolerance, rtol=0), f"Métrica incorreta: {key}")

def verify_preprocessor(matrix, y, rows, arrays, prefix, k):
    """Estatísticas calculadas explicitamente nas linhas de treino declaradas."""
    np.testing.assert_array_equal(arrays[prefix+"train_rows"], rows)
    raw = matrix[rows]
    require(np.isfinite(raw).all(), "Insumos completos devem ser finitos")
    np.testing.assert_array_equal(np.median(raw, axis=0), arrays[prefix+"imputer"])
    variable = np.ptp(raw, axis=0) > 0
    np.testing.assert_array_equal(variable, arrays[prefix+"constant_mask"])
    require(k <= int(variable.sum()), "Atributos não constantes insuficientes")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = f_classif(raw[:, variable], y[rows])[0]
    np.testing.assert_allclose(scores, arrays[prefix+"selector_scores"], rtol=1e-12, atol=1e-12, equal_nan=True)
    cleaned = np.where(np.isnan(scores), np.finfo(scores.dtype).min, scores)
    selected = np.zeros(len(scores), dtype=bool)
    selected[np.argsort(cleaned, kind="mergesort")[-k:]] = True
    np.testing.assert_array_equal(selected, arrays[prefix+"selector_mask"])
    chosen = raw[:, variable][:, selected].astype(np.float64)
    mean, variance = chosen.mean(axis=0), chosen.var(axis=0)
    epsilon = np.finfo(np.float64).eps
    constant = variance <= len(rows)*epsilon*variance + (len(rows)*mean*epsilon)**2
    scale = np.sqrt(variance); scale[constant] = 1.
    np.testing.assert_allclose(mean, arrays[prefix+"scaler_mean"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scale, arrays[prefix+"scaler_scale"], rtol=1e-12, atol=1e-12)

def transform(matrix, rows, arrays, prefix):
    values = matrix[rows].copy()
    require(np.isfinite(values).all(), "Atributos não finitos durante replay")
    values = values[:, arrays[prefix+"constant_mask"].astype(bool)]
    values = values[:, arrays[prefix+"selector_mask"].astype(bool)]
    # Atribuição in-place reproduz arredondamento float32 do StandardScaler.
    values -= arrays[prefix+"scaler_mean"]
    values /= arrays[prefix+"scaler_scale"]
    return values

def paired_interval(raw, seeds, coefficients, iterations=2000):
    ids = sorted(raw.participant_id.unique())
    target = raw.drop_duplicates("participant_id").set_index("participant_id").loc[ids].y.to_numpy()
    contributions = []
    for seed in seeds:
        delta = np.zeros(len(ids), dtype=np.float64)
        for arm, coefficient in coefficients.items():
            rows = raw[(raw.seed == seed) & (raw.arm == arm)].set_index("participant_id")
            require(not rows.index.duplicated().any() and set(rows.index) == set(ids), "Contraste sem OOF completo")
            rows = rows.loc[ids]
            np.testing.assert_array_equal(rows.y, target)
            delta += coefficient*(rows.predicao.to_numpy() == target)
        contributions.append(delta)
    contribution = np.mean(contributions, axis=0)
    normal, pathological = np.flatnonzero(target == 0), np.flatnonzero(target == 1)
    rng = np.random.default_rng(20261105)
    ns = rng.choice(normal, size=(iterations, len(normal)), replace=True)
    ps = rng.choice(pathological, size=(iterations, len(pathological)), replace=True)
    values = .5*(contribution[ns].mean(axis=1)+contribution[ps].mean(axis=1))
    low, high = np.quantile(values, [.025, .975])
    return np.array([.5*(contribution[normal].mean()+contribution[pathological].mean()), low, high])


def independent_branches(rep):
    if rep in ("WPT_SENT128", "WPT_VOW128"):
        return (rep,)
    if rep == "WPT_SENT_VOW256":
        return ("WPT_SENT128", "WPT_VOW128")
    kind, layer = rep.rsplit("_L", 1)
    require(int(layer) in (24,5,12,16), "Camada desconhecida")
    sent, vow = f"SSL_SENT128_L{layer}", f"SSL_VOW128_L{layer}"
    branches = {"SSL_SENT128": (sent,), "SSL_VOW128": (vow,), "SSL_SENT_VOW256": (sent,vow),
                "HYB_SENT_VOW512": ("WPT_SENT128","WPT_VOW128",sent,vow),
                "SSL_SENT_VOW512": (f"SSL_SENT256_L{layer}",f"SSL_VOW256_L{layer}"),
                "HYB_SENT256": ("WPT_SENT128",sent)}
    require(kind in branches, "Representação desconhecida")
    return branches[kind]


def representation(values, rep):
    branches = independent_branches(rep)
    if len(branches)==1:
        return values[branches[0]]
    divisor=2. if rep.startswith(("HYB_SENT_VOW512", "SSL_SENT_VOW512")) else np.sqrt(2.)
    return np.concatenate([values[key] for key in branches],axis=1)/divisor


def independent_winner(candidates, allowed):
    eligible=candidates[candidates.representation.isin(allowed)]
    require(len(eligible)>0 and np.isfinite(eligible.inner_BA).all(), "Candidatos inválidos")
    tied=eligible[eligible.inner_BA>=eligible.inner_BA.max()-1e-12]
    order={0:-1,24:0,5:1,12:2,16:3}
    return min(tied.itertuples(),key=lambda row:(row.C,order[row.layer],row.candidate_id))


def verify_aggregation(matrices, people, names, mapping, record_matrices, record_meta, record_names,
                       vowel_arrays, vowel_meta):
    require(not people.participant_id.duplicated().any(), "Linhas de pessoas duplicadas")
    ids=sorted(record_meta.participant_id.unique())
    np.testing.assert_array_equal(people.participant_id,ids)
    require(set(vowel_meta.participant_id)==set(ids) and not vowel_meta.participant_id.duplicated().any(), "Vogal omite/duplica pessoas")
    require(not record_meta.sample_id.duplicated().any(), "Frase duplicada")
    label=record_meta.groupby("participant_id").y
    require(label.nunique().eq(1).all(), "Alvo inconsistente nas frases")
    np.testing.assert_array_equal(people.y,people.participant_id.map(label.first()))
    expected_mapping=[]
    for person_row,pid in enumerate(ids):
        rows=np.flatnonzero(record_meta.participant_id.to_numpy()==pid)
        require(people.iloc[person_row].n_sentences==len(rows), "Contagem de frases incorreta")
        for source_row in rows:
            record=record_meta.iloc[source_row]
            expected_mapping.append({"participant_id":pid,"person_row":person_row,"source_row":int(source_row),
                                     "sample_id":record.sample_id,"task":str(record.task).zfill(3),
                                     "weight":1./len(rows),"n_sentences":len(rows)})
    expected_mapping=pd.DataFrame(expected_mapping)
    checked=mapping.copy();checked["task"]=checked.task.astype(str).str.zfill(3)
    pd.testing.assert_frame_equal(checked.reset_index(drop=True),expected_mapping,check_dtype=False)
    require(not mapping.source_row.duplicated().any() and set(mapping.source_row)==set(range(len(record_meta))), "Vínculo de frases incompleto")
    np.testing.assert_allclose(mapping.groupby("participant_id").weight.sum().loc[ids],1.,atol=1e-15,rtol=0)
    for original,matrix in record_matrices.items():
        new="wpt_sent" if original=="wpt" else "ssl_sent"+original[3:]
        expected=np.stack([matrix[record_meta.participant_id.eq(pid).to_numpy()].mean(axis=0,dtype=np.float64).astype(np.float32) for pid in ids])
        np.testing.assert_array_equal(matrices[new],expected)
        np.testing.assert_array_equal(names[new],np.asarray(["SENT__"+str(x) for x in record_names[original]]))
    ordered=vowel_meta.set_index("participant_id").loc[ids]
    np.testing.assert_array_equal(ordered.y,people.y)
    np.testing.assert_array_equal(ordered.sample_id,people.vowel_sample_id)
    np.testing.assert_array_equal(ordered.task.astype(str).str.zfill(3),np.repeat("002",len(ids)))
    rows=ordered.matrix_row.to_numpy(dtype=int)
    require(len(set(rows))==len(ids) and set(rows)==set(range(len(ids))), "Ordem da vogal inválida")
    np.testing.assert_array_equal(matrices["wpt_vow"],vowel_arrays["wavelet_raw"][rows])
    np.testing.assert_array_equal(names["wpt_vow"],np.asarray(["VOW__"+str(x) for x in vowel_arrays["wavelet_feature_names"]]))
    for layer in (24,5,12,16):
        key=f"ssl_vow{layer:02d}"
        np.testing.assert_array_equal(matrices[key],vowel_arrays[f"embeddings_layer{layer:02d}"][rows])
        np.testing.assert_array_equal(names[key],np.asarray(["VOW__"+str(x) for x in vowel_arrays[f"feature_names_layer{layer:02d}"]]))
    for key in ("sample_id","participant_id","task"):
        np.testing.assert_array_equal(np.asarray(vowel_arrays[key]).astype(str),vowel_meta[key].astype(str))
    np.testing.assert_array_equal(vowel_arrays["hidden_layer_indices"],[24,5,12,16])


def blank_totals():
    return {"outer_folds":0,"inner_folds":0,"checkpoint_hashes":0,"preprocessors":0,"classifiers":0,
            "inner_OOF":0,"candidate_summaries":0,"selections":0,"test_margins":0,"train_margins":0,
            "metric_rows":0,"max_margin_error":0.}


def audit_checkpoint(matrices,meta,names,train,test,seed,fold,frames,arrays,totals,c_grid=None):
    c_grid=EXP30.C_GRID if c_grid is None else c_grid
    EXP30.validate(frames,arrays,meta,train,test,names,seed,fold,c_grid)
    require(not meta.participant_id.duplicated().any(), "O classificador deve receber uma linha por pessoa")
    y=meta.y.to_numpy(dtype=int)
    require(not set(meta.iloc[train].participant_id)&set(meta.iloc[test].participant_id), "Vazamento externo")
    inner,candidates,choices=(frames[key] for key in ("predicoes_oof_internas.csv","candidatos_internos.csv","selecoes.csv"))
    totals["inner_OOF"]+=len(inner);totals["candidate_summaries"]+=len(candidates);totals["selections"]+=len(choices)
    grouped=dict(tuple(inner.groupby("candidate_id")))
    for row in candidates.itertuples():
        measured=metrics(grouped[row.candidate_id])
        for key in ("BA","sensibilidade","especificidade","acuracia"):
            require(abs(measured[key]-getattr(row,"inner_"+key))<1e-12,"Métrica interna incorreta")
    for selected in choices.itertuples():
        winner=independent_winner(candidates,EXP30.ARM_REPS[selected.arm])
        require(selected.candidate_id==winner.candidate_id and abs(selected.inner_BA-winner.inner_BA)<1e-12,"Escolha não corresponde à BA interna/desempate")
    for inner_fold,(itr,iva) in enumerate(independent_splits(meta.iloc[train].reset_index(drop=True),3,int(seed)+100+int(fold)),1):
        tr,va=train[itr],train[iva]
        require(not set(meta.iloc[tr].participant_id)&set(meta.iloc[va].participant_id),"Pessoa compartilhada internamente")
        require(not set(tr)&set(test) and not set(va)&set(test),"Teste externo entrou na busca")
        for branch,(modality,k) in EXP30.PREPROCESSORS.items():
            verify_preprocessor(matrices[modality],y,tr,arrays,f"inner{inner_fold}__{branch}__",k)
            totals["preprocessors"]+=1
        totals["inner_folds"]+=1
    for branch,(modality,k) in EXP30.PREPROCESSORS.items():
        verify_preprocessor(matrices[modality],y,train,arrays,f"outer__{branch}__",k)
        totals["preprocessors"]+=1
    for rows,filename,count_key in ((test,"predicoes.csv","test_margins"),(train,"predicoes_treino_participante.csv","train_margins")):
        values={branch:transform(matrices[modality],rows,arrays,f"outer__{branch}__") for branch,(modality,_) in EXP30.PREPROCESSORS.items()}
        for selected in choices.itertuples():
            arm=selected.arm
            z=representation(values,selected.representation).astype(np.float64)
            reconstructed=(z@arrays[f"{arm}__coef"].T).reshape(-1)+arrays[f"{arm}__intercept"][0]
            observed=frames[filename].query("arm == @arm").set_index("participant_id").loc[meta.iloc[rows].participant_id].score.to_numpy()
            error=float(np.max(np.abs(reconstructed-observed)))
            require(error<1e-7,f"Margem por pessoa não reproduzida: {arm}; erro={error}")
            totals["max_margin_error"]=max(totals["max_margin_error"],error);totals[count_key]+=len(reconstructed)
    for selected in choices.itertuples():
        arm=selected.arm
        np.testing.assert_array_equal(arrays[f"{arm}__classes"],[0,1])
        np.testing.assert_allclose(arrays[f"{arm}__class_weight"],len(train)/(2*np.bincount(y[train],minlength=2)),atol=1e-15,rtol=0)
        support=arrays[f"{arm}__support_indices"].astype(int)
        require(np.all((support>=0)&(support<len(train))),"Pessoa de suporte fora do treino")
        np.testing.assert_array_equal(arrays[f"{arm}__support_original_rows"],train[support])
        values={branch:transform(matrices[modality],train[support],arrays,f"outer__{branch}__") for branch,(modality,_) in EXP30.PREPROCESSORS.items()}
        np.testing.assert_allclose(representation(values,selected.representation),arrays[f"{arm}__support_vectors"],atol=1e-12,rtol=0)
        np.testing.assert_allclose(arrays[f"{arm}__dual_coef"]@arrays[f"{arm}__support_vectors"],arrays[f"{arm}__coef"],atol=1e-10,rtol=1e-10)
        totals["classifiers"]+=1
    require(frames["metricas.csv"].unit.eq("participante").all(),"Métricas de gravação não pertencem ao30")
    for row in frames["metricas.csv"].itertuples():
        filename="predicoes.csv" if row.phase=="teste" else "predicoes_treino_participante.csv"
        compare_numbers(metrics(frames[filename].query("arm == @row.arm")),row)
        totals["metric_rows"]+=1
    totals["outer_folds"]+=1


def audit(out,data20,data23,data27,data29,report):
    started=time.perf_counter();marker_path=out/"ARTEFATOS_COMPLETOS.json"
    require(marker_path.is_file(),"Experimento30 incompleto; nenhum laudo será escrito")
    exports=json.loads(marker_path.read_text(encoding="utf8"))
    required={"predicoes_oof_participante.csv","metricas_por_seed.csv","comparacoes_pareadas.csv",
              "entradas_participantes.csv","agregacao_frases.csv","amostras_suporte.csv","CODIGO_CONGELADO.json"}
    require(required.issubset(exports),"Exports finais incompletos")
    for name,digest in exports.items():
        require(Path(name).name==name and sha(out/name)==digest,f"Export alterado: {name}")
    identity=json.loads((out/"assinatura_execucao.json").read_text(encoding="utf8"))
    for category in ("code","input_files"):
        for name,digest in identity[category].items():
            require(sha(name)==digest,f"Fonte/insumo assinado alterado: {name}")
    protocol_hash=hashlib.sha256((out/"PROTOCOLO.txt").read_text(encoding="utf8").encode("utf8")).hexdigest()
    require(protocol_hash==identity["protocol_sha256"],"Protocolo alterado")
    frozen=json.loads((out/"CODIGO_CONGELADO.json").read_text(encoding="utf8"))
    require(frozen=={Path(name).name:digest for name,digest in identity["code"].items()},"Manifesto das cópias de código diferente")
    for name,digest in frozen.items():
        require(sha(out/"codigo_congelado"/name)==digest,"Cópia congelada diferente da fonte executada")
    config=identity["configuration"];seeds=tuple(config["seeds"])
    require(seeds==EXP30.SEEDS and config["arms"]==list(EXP30.ARMS),"Auditoria final exige as seis sementes/nove braços")
    expected={"fit_unit":"participant","sentence_aggregation":"arithmetic_equal_recording_mean_float64_to_float32_before_CV",
              "wavelet_input":"RAW_legacy_validated_exactly23","vowel_task":"002","outer_folds":5,"inner_folds":3,
              "class_balance":"balanced_person_train_only","threshold":0.,"n_preprocessors_per_train":18,
              "n_candidates_per_inner":135,"clinical_inputs":False}
    require(all(config.get(key)==value for key,value in expected.items()),"Contrato de unidade/entrada/avaliação diferente")
    require(config["C_grid"]==list(EXP30.C_GRID) and config["layers"]==[24,5,12,16],"Grade/camadas diferentes")
    old28=ROOT/EXP30.EXP28.OUT_REL
    old28_marker_path=old28/"ARTEFATOS_COMPLETOS.json"
    old28_marker=json.loads(old28_marker_path.read_text(encoding="utf8"))
    require(old28_marker.get("particoes_participantes.csv")==sha(old28/"particoes_participantes.csv"),"Partições28 de referência alteradas")
    report_json=report.with_suffix(".json")
    if report.exists() or report_json.exists():
        require(report.exists() and report_json.exists(),"Laudo parcial preservado")
        prior=json.loads(report_json.read_text(encoding="utf8"))
        require(prior["status"]=="aprovado" and prior["auditor_sha256"]==sha(__file__) and
                prior["marker30_sha256"]==sha(marker_path) and prior["report_txt_sha256"]==sha(report),"Laudo prévio divergente/alterado, preservado")
        print("AUDITORIA30 ANTERIOR CONFIRMADA; nenhum arquivo alterado",flush=True);return
    matrices,meta,names,mapping=EXP30.load_inputs(ROOT,data20,data23,data27,data29)
    n_people,n_normal,n_pathological=COHORT_COUNTS
    require(len(meta)==n_people and not meta.participant_id.duplicated().any(),"Não há uma linha por pessoa")
    require(meta.y.value_counts().to_dict()=={1:n_pathological,0:n_normal},"Classes diferentes da coorte")
    record_matrices,record_meta,record_names=EXP30.EXP28.load_inputs(ROOT,data20,data27)
    vowel_meta=pd.read_csv(data29/"metadata_vogal_wavelet_ssl.csv",dtype={"sample_id":str,"participant_id":str,"task":str},float_precision="round_trip")
    vowel_meta["task"]=vowel_meta.task.str.zfill(3)
    with np.load(data29/"features_vogal_wavelet_ssl.npz",allow_pickle=False) as data:
        vowel_arrays={name:data[name] for name in data.files}
    verify_aggregation(matrices,meta,names,mapping,record_matrices,record_meta,record_names,vowel_arrays,vowel_meta)
    pd.testing.assert_frame_equal(read_csv(out/"entradas_participantes.csv"),meta,check_dtype=False)
    saved_mapping=read_csv(out/"agregacao_frases.csv");saved_mapping["task"]=saved_mapping.task.astype(str).str.zfill(3)
    pd.testing.assert_frame_equal(saved_mapping,mapping,check_dtype=False)
    pd.testing.assert_frame_equal(read_csv(out/"particoes_participantes.csv"),EXP30.EXP19.fold_table(meta,seeds),check_dtype=False)
    # As divisões de pessoas continuam iguais às de28, apesar da nova unidade.
    old_partitions=read_csv(old28/"particoes_participantes.csv")
    new_partitions=read_csv(out/"particoes_participantes.csv")
    columns=["seed","fold","role","participant_id","y"]
    left=new_partitions[columns].sort_values(columns).reset_index(drop=True)
    right=old_partitions[old_partitions.seed.isin(seeds)][columns].sort_values(columns).reset_index(drop=True)
    pd.testing.assert_frame_equal(left,right,check_dtype=False)
    raw=read_csv(out/"predicoes_oof_participante.csv");summary=read_csv(out/"metricas_por_seed.csv")
    require(len(raw)==n_people*len(seeds)*len(EXP30.ARMS) and len(summary)==len(seeds)*len(EXP30.ARMS),"Resultados OOF incompletos")
    require(not raw.duplicated(["seed","arm","participant_id"]).any(),"OOFs duplicados")
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();totals=blank_totals()
    aggregate={name:[] for name in ("predicoes.csv","metricas.csv","atributos.csv","candidatos_internos.csv","selecoes.csv","amostras_suporte.csv")}
    for seed in seeds:
        for fold,(train,test) in enumerate(independent_splits(meta,5,seed),1):
            destination=out/"checkpoints"/f"seed_{seed}_fold_{fold:02d}"
            frames,arrays=EXP30.load_checkpoint(destination,signature)
            totals["checkpoint_hashes"]+=len(EXP30.ARTIFACTS)
            audit_checkpoint(matrices,meta,names,train,test,seed,fold,frames,arrays,totals)
            for name in aggregate:aggregate[name].append(frames[name])
            print(f"Auditoria30: seed={seed}, fold={fold}, nove modelos por pessoa conferidos",flush=True)
    n_folds,n_arms=5*len(seeds),len(EXP30.ARMS);n_candidates=len(EXP30.REPRESENTATIONS)*len(EXP30.C_GRID)
    expected_totals={"outer_folds":n_folds,"inner_folds":3*n_folds,"checkpoint_hashes":len(EXP30.ARTIFACTS)*n_folds,
                     "preprocessors":len(EXP30.PREPROCESSORS)*4*n_folds,"classifiers":n_arms*n_folds,
                     "inner_OOF":n_candidates*len(seeds)*4*n_people,"candidate_summaries":n_candidates*n_folds,
                     "selections":n_arms*n_folds,"test_margins":n_people*len(seeds)*n_arms,
                     "train_margins":n_people*4*len(seeds)*n_arms,"metric_rows":2*n_arms*n_folds}
    for key,value in expected_totals.items():require(totals[key]==value,f"Contagem incorreta {key}: {totals[key]} != {value}")
    consolidated={"predicoes.csv":"predicoes_oof_participante.csv","metricas.csv":"metricas_por_fold.csv",
                  "atributos.csv":"atributos_selecionados.csv","candidatos_internos.csv":"candidatos_internos.csv",
                  "selecoes.csv":"selecoes_internas.csv","amostras_suporte.csv":"amostras_suporte.csv"}
    for name,filename in consolidated.items():
        pd.testing.assert_frame_equal(read_csv(out/filename),pd.concat(aggregate[name],ignore_index=True),check_dtype=False)
    for row in summary.itertuples():
        group=raw[(raw.seed==row.seed)&(raw.arm==row.arm)]
        require(len(group)==n_people and not group.participant_id.duplicated().any(),"Resumo não cobre todas as pessoas")
        measured=metrics(group);measured["AUC_OOF_exploratoria"]=measured.pop("AUC");compare_numbers(measured,row)
        aucs,weights=[],[]
        for _,fg in group.groupby("fold"):
            aucs.append(float(roc_auc_score(fg.y,fg.score)));weights.append(int(fg.y.eq(0).sum()*fg.y.eq(1).sum()))
        require(abs(np.average(aucs,weights=weights)-row.AUC_media_folds_ponderada_pares)<1e-12,"AUC por fold ponderada incorreta")
    comparisons=read_csv(out/"comparacoes_pareadas.csv")
    n_scopes=int(20261105 in seeds)+int(any(seed!=20261105 for seed in seeds))
    require(len(comparisons)==n_scopes*len(EXP30.CONTRASTS) and not comparisons.duplicated(["escopo","comparacao"]).any(),"Contrastes incompletos")
    ci_error=0.
    for row in comparisons.itertuples():
        require(row.escopo in {"principal_seed20261105","estabilidade_secundaria"} and row.comparacao in EXP30.CONTRASTS,"Escopo/contraste inválido")
        selected_seeds=[seeds[0]] if row.escopo=="principal_seed20261105" else list(seeds[1:])
        expected_ci=paired_interval(raw,selected_seeds,EXP30.CONTRASTS[row.comparacao])
        observed=np.asarray([row.delta_BA,row.IC95_inferior,row.IC95_superior])
        np.testing.assert_allclose(expected_ci,observed,atol=1e-14,rtol=0);ci_error=max(ci_error,float(np.max(np.abs(expected_ci-observed))))
    now=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")
    lines=["AUDITORIA FINAL DO EXPERIMENTO30 — APROVADA",f"Verificado em: {now}",f"Experimento: {out}",
           f"Auditor: {Path(__file__).resolve()}",f"SHA256 auditor: {sha(__file__)}","",
           "Nenhum classificador foi treinado/reajustado; nenhum WAV foi decodificado e nenhuma GPU foi usada.",
           "Dados, fontes, caches e resultados foram lidos sem modificação. Laudo TXT/JSON só após todos os controles passarem.","",
           "UNIDADE, AGREGAÇÃO E CONTRATOS","Uma linha de entrada e predição por pessoa. Nenhuma métrica por gravação.",
           "Todas as médias de5/6frases foram recalculadas por PID: acumuladorfloat64, saídafloat32, pesos iguais1/n.",
           "Vínculos de todas as2.465frases, ordem/IDs/rótulos, nomesSENT__/VOW__, vogal002 e todas as quatro camadas conferidos.",
           "Somente WPT RAW predefinido entra no30. RMS29 não participa da escolha; clínica/demografia não entra nas matrizes.",
           "As pessoas/folds correspondem aos de28, mas a unidade de treinamento mudou; não se exige igualdade de scores28.","",
           "VERIFICAÇÕES COMPUTACIONAIS",f"Hashes de todos os exports, fontes, insumos e {totals['checkpoint_hashes']} arquivos de checkpoints confirmados.",
           "Medianas, constantes, ANOVA, ranking, média e escala calculados somente nas pessoas do treino da partição.",
           "ANOVA usa f_classif como função estatística, sem fit de classificador. Fusão e divisores sqrt2/2 reconstruídos independentemente.",
           "C/camada conferidos pela BA OOF interna e desempates predefinidos. Camada é compartilhada entre fontes nas fusões.",
           "Todas as margens externas de treino e teste por pessoa foram reproduzidas dos coeficientes, com tolerância absoluta1e-7.",
           "Pesos balanced conferidos sobre pessoas de treino, não gravações. Suportes/coefs/partições/nomes selecionados conferidos.",
           "Margens dos classificadores internos não são refeitas: seus coeficientes não são exportados. OOFs salvos/estatísticas/seleção são auditados.",
           "Resumos por semente, matrizes, BA, sensibilidade, especificidade, acurácia, AUC e ICs pareados recalculados.",""]
    lines.extend(f"{key}: {value}" for key,value in totals.items())
    lines.extend([f"outer_OOF: {len(raw)}",f"summary_rows: {len(summary)}",f"paired_CIs: {len(comparisons)}",
                  f"max_CI_error: {ci_error:.3g}",f"seconds: {time.perf_counter()-started:.3f}","","LIMITES",
                  "Aprovação computacional não estabelece validade clínica, estado da arte ou generalização externa.",
                  "A mesma AVFAD foi repetidamente explorada; ICs condicionais não corrigem todo o histórico nem refazem modelos.",
                  "As sementes reutilizam as mesmas pessoas. Idade/ano/sexo/aquisição podem continuar confundidores acústicos.",
                  "A média perde ordem temporal; orçamento de atributos não iguala todas as propriedades dos modelos.",
                  "C-A e F-D mudam tanto a tarefa disponível quanto a dimensão128→256; não isolam um efeito causal da vogal.",
                  "G-F deve ser lido com G-H: o controle H iguala512 atributos, mantendo diferenças de representação e escolhas internas.",
                  "Não houve novo treinamento SVD nem adaptação do encoder nesta auditoria.",""])
    text_bytes="\n".join(lines).encode("utf8")
    outcome={"status":"aprovado","verified_at":now,"auditor_sha256":sha(__file__),"marker30_sha256":sha(marker_path),
             "n_folds":n_folds,"n_arms":n_arms,"n_outer":n_folds*n_arms,"n_people":n_people,"totals":totals,
             "reference28_marker_sha256":sha(old28_marker_path),
             "aggregation_checked":True,"fit_unit":"participant","paired_CIs":len(comparisons),"max_CI_error":ci_error,
             "all_external_train_and_test_margins_replayed":True,"internal_classifier_margins_replayed":False,
             "report_txt_sha256":hashlib.sha256(text_bytes).hexdigest(),"report_txt":str(report),"model_refits":0,"audio_decodes":0,"gpu_used":False}
    report.parent.mkdir(parents=True,exist_ok=True)
    with report.open("xb") as stream:stream.write(text_bytes)
    with report_json.open("x",encoding="utf8") as stream:json.dump(outcome,stream,ensure_ascii=False,indent=2);stream.write("\n")
    print("AUDITORIA30 APROVADA",json.dumps(totals,sort_keys=True),flush=True)
    print("Laudos:",report,report_json,flush=True)


def selftest():
    import unittest
    spec=importlib.util.spec_from_file_location("test_independent_audit30",HERE/"test_verificar_frase_vogal30.py")
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests)).wasSuccessful():raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento",type=Path,default=ROOT/EXP30.OUT_REL)
    parser.add_argument("--relatorio",type=Path,default=ROOT/"results/experiments/documentacao_wav2vec/30_VERIFICACAO_INDEPENDENTE.txt")
    defaults={20:EXP30.EXP21.DATA20_REL,23:EXP30.DATA23_REL,27:EXP30.EXP28.DATA27_REL,29:EXP30.DATA29_REL}
    for number,relative in defaults.items():parser.add_argument(f"--dados{number}",type=Path,default=ROOT/relative)
    parser.add_argument("--autoteste",action="store_true");args=parser.parse_args()
    if args.autoteste:selftest();return
    out,report=args.experimento.resolve(),args.relatorio.resolve();paths={number:getattr(args,f"dados{number}").resolve() for number in defaults}
    require(all(path.is_relative_to(ROOT/"results/experiments") for path in (out,report,*paths.values())),"Caminho fora do workspace de experimentos")
    audit(out,paths[20],paths[23],paths[27],paths[29],report)


if __name__=="__main__":main()
