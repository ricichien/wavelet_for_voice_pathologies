# -*- coding: utf-8 -*-
"""Auditoria focalizada do ComParE na AVFAD HSA.

Reutiliza os NPZ/CSV e os mesmos folds do benchmark existente. Nao altera
extratores, checkpoints ou resultados anteriores. Testa o efeito de reduzir
6373 features ComParE a apenas 128, e de usar balanceamento.

Coloque na mesma pasta de modelar_classificadores_avfad.py e execute:
    python auditar_compare_avfad.py

Saidas novas:
    auditoria_compare_avfad_resumo.csv
    auditoria_compare_avfad_folds.csv
    auditoria_compare_avfad_predicoes.csv

IMPORTANTE: apenas diagnostico exploratorio na HSA_411; nao replica os folds,
a coorte, nem o protocolo de avaliacao da publicacao de Ribas et al. (2023).
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.svm import SVC

import modelar_classificadores_avfad as base

ROOT = project_root()
OUT_RESUMO = asset(ROOT, "auditoria_compare_avfad_resumo.csv")
OUT_FOLDS = asset(ROOT, "auditoria_compare_avfad_folds.csv")
OUT_PREDS = asset(ROOT, "auditoria_compare_avfad_predicoes.csv")

# Controle de uma variavel por vez. O caso 128 balanceado e o benchmark existente.
CONFIGURACOES = [
    ("ComParE_128_balanceado", "compare", 128, True, "linear"),
    ("ComParE_512_balanceado", "compare", 512, True, "linear"),
    ("ComParE_6373_balanceado", "compare", 6373, True, "linear"),
    ("ComParE_6373_sem_pesos", "compare", 6373, False, "linear"),
    ("ComParE_6373_poly_grau1_sem_pesos", "compare", 6373, False, "poly"),
    ("MFCC_128_balanceado", "mfcc", 128, True, "linear"),
    ("MFCC_150_balanceado", "mfcc", 150, True, "linear"),
    ("WPT_sym5_128_balanceado", "wpt_sym5", 128, True, "linear"),
]


def calc_metrics(y, scores):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        n=int(y.size), accuracy=float(accuracy_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        auc=float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan"),
        f1=float(f1_score(y, pred, zero_division=0)),
        sensitivity=float(tp / (tp + fn)) if tp + fn else float("nan"),
        specificity=float(tn / (tn + fp)) if tn + fp else float("nan"),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def main():
    start = time.perf_counter()
    print("Carregando os arquivos extraidos anteriormente, sem processar WAVs...")
    X_wave, wave_names, X_mfcc, mfcc_names, X_compare, compare_names, df, _ = base.load_inputs()
    if X_compare is None:
        raise RuntimeError("Nao encontrei matriz ComParE no arquivo NPZ.")
    n_compare = X_compare.shape[1]
    if n_compare != 6373:
        raise RuntimeError(f"Esperava 6373 features ComParE, encontrei {n_compare}.")

    masks = np.array([str(n).startswith('WPT__sym5__') for n in wave_names])
    indices = np.flatnonzero(masks)
    Xs = {
        'compare': (X_compare, np.arange(n_compare), compare_names),
        'mfcc': (X_mfcc, np.arange(X_mfcc.shape[1]), mfcc_names),
        'wpt_sym5': (X_wave, indices, wave_names),
    }
    _, fold_by_row = None, None
    fold_by_row, assignments = base.create_participant_folds(df)
    print("Folds por participante confirmados:", assignments.groupby('fold').size().to_dict())
    print("ComParE bruto: NaN=", int(np.isnan(X_compare).sum()),
          "Inf=", int(np.isinf(X_compare).sum()),
          "colunas=", X_compare.shape[1])

    summary = []
    folds = []
    predictions = []
    y = df['y'].to_numpy(dtype=int)
    for config_name, repr_name, k, balanced, kernel in CONFIGURACOES:
        X, feat_idx, feat_names = Xs[repr_name]
        if not len(feat_idx):
            raise RuntimeError(f"Representacao vazia: {repr_name}")
        base.TOP_K = k
        fold_people = []
        fold_recordings = []
        selected_counts = []
        for fold in range(1, 6):
            tr = np.flatnonzero(fold_by_row != fold - 1)
            te = np.flatnonzero(fold_by_row == fold - 1)
            train_people = set(df.iloc[tr]['participant_id'])
            test_people = set(df.iloc[te]['participant_id'])
            assert not (train_people & test_people), "VAZAMENTO DE PARTICIPANTE"
            xt, xv, yt, selected_names = base.prepare_fold_data(
                X, feat_idx, feat_names, df, tr, te, config_name, fold)
            selected_counts.append(len(selected_names))
            model = SVC(kernel=kernel, degree=1, C=1.0,
                        class_weight='balanced' if balanced else None,
                        cache_size=512)
            model.fit(xt, yt)
            scores = model.decision_function(xv)
            m = df.iloc[te][['participant_id', 'sample_id', 'y']].copy()
            m['score'] = scores
            record_metrics = calc_metrics(m['y'], m['score'])
            people = m.groupby('participant_id', as_index=False).agg(
                y=('y', 'first'), score=('score', 'mean'), n_gravacoes=('score', 'size'))
            people['fold'] = fold
            people['configuracao'] = config_name
            people['predicao'] = (people.score >= 0).astype(int)
            fold_people.append(people)
            fold_recordings.append(m)
            pmetrics = calc_metrics(people['y'], people['score'])
            folds.append(dict(configuracao=config_name, fold=fold,
                              selecionadas=len(selected_names),
                              BA_participante=pmetrics['balanced_accuracy'],
                              BA_gravacao=record_metrics['balanced_accuracy'],
                              auc_participante=pmetrics['auc'],
                              auc_gravacao=record_metrics['auc']))
            print(f"  {config_name:39s} fold {fold}: BA pessoa={pmetrics['balanced_accuracy']:.3f}", flush=True)

        people_oof = pd.concat(fold_people, ignore_index=True).sort_values('participant_id')
        records_oof = pd.concat(fold_recordings, ignore_index=True).sort_values('sample_id')
        if len(people_oof) != 411 or people_oof.participant_id.nunique() != 411:
            raise RuntimeError(f"{config_name}: predições por pessoa nao incluem 411 IDs únicos")
        if len(records_oof) != 2465 or records_oof.sample_id.nunique() != 2465:
            raise RuntimeError(f"{config_name}: predições por gravação incompletas")
        m_p = calc_metrics(people_oof.y, people_oof.score)
        m_r = calc_metrics(records_oof.y, records_oof.score)
        summary.append(dict(configuracao=config_name, representacao=repr_name,
                            top_k=k, pesos_balanceados=balanced, kernel=kernel,
                            selecionadas_media=float(np.mean(selected_counts)),
                            BA_participante=m_p['balanced_accuracy'],
                            AUC_participante=m_p['auc'],
                            ACC_participante=m_p['accuracy'],
                            sens_participante=m_p['sensitivity'],
                            esp_participante=m_p['specificity'],
                            BA_gravacao=m_r['balanced_accuracy'],
                            AUC_gravacao=m_r['auc'],
                            ACC_gravacao=m_r['accuracy'],
                            sens_gravacao=m_r['sensitivity'],
                            esp_gravacao=m_r['specificity'],
                            tp=m_p['tp'], tn=m_p['tn'], fp=m_p['fp'], fn=m_p['fn']))
        predictions.append(people_oof)
        pd.DataFrame(summary).to_csv(OUT_RESUMO, index=False, encoding='utf-8-sig')
        pd.DataFrame(folds).to_csv(OUT_FOLDS, index=False, encoding='utf-8-sig')
        pd.concat(predictions, ignore_index=True).to_csv(OUT_PREDS, index=False, encoding='utf-8-sig')
        print(f"OOF {config_name}: BA pessoa={m_p['balanced_accuracy']:.4f}, "
              f"AUC={m_p['auc']:.4f}, BA gravação={m_r['balanced_accuracy']:.4f}", flush=True)

    print("\n=== RESUMO ===")
    show = ['configuracao','BA_participante','AUC_participante','BA_gravacao','ACC_gravacao']
    print(pd.DataFrame(summary)[show].to_string(index=False))
    print(f"\nAuditoria concluida em {(time.perf_counter()-start)/60:.2f} min")
    print(f"Arquivos novos: {OUT_RESUMO.name}, {OUT_FOLDS.name}, {OUT_PREDS.name}")
    print("ATENCAO: rodadas exploratorias; nao escolher um vencedor como estimativa final imparcial.")

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f"\nERRO NA AUDITORIA: {exc}", file=sys.stderr)
        raise
