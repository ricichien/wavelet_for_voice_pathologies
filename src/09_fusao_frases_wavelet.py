# -*- coding: utf-8 -*-
"""AVFAD - teste de tres arquiteturas WPT sym5 que preservam a estrutura CAPE-V.

Objetivo cientifico: testar se manter a identidade das seis frases melhora
as caracteristicas WPT existentes (sem releitura dos WAVs).

Arquiteturas PREDEFINIDAS, nao escolhidas por desempenho dos folds:
  POOLED: 6 frases independentes no treino, media dos scores por pessoa.
  FUSAO_TARDIA: um SVC por frase, media IGUAL dos 6 scores.
  BLOCOS_POR_FRASE: 32 features selecionadas EM CADA frase, depois concatena
                     os 6 blocos de um participante e treina um unico SVC.

Cada seed usa os mesmos participantes e folds para as tres arquiteturas.
Todas as transformacoes supervisionadas/normalizacoes sao ajustadas SOMENTE
nos participantes de treino. Resultados exploratorios; nao validacao externa.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from avfad_core import load_data, representations

VERSION = '09-v1.0'
TASKS = ('004', '005', '006', '007', '008', '009')
ARMS = ('POOLED', 'FUSAO_TARDIA', 'BLOCOS_POR_FRASE')


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados', default='.', help='Diretorio dos NPZ e CSV ja extraidos')
    p.add_argument('--saida', default='resultados_09_fusao_wavelet', help='Diretorio novo, dedicado a este experimento')
    p.add_argument('--repeticoes', default=5, type=int, help='Numero de divisoes repetidas (1 a 20), padrao 5')
    p.add_argument('--primeira-seed', default=101, type=int, help='Seed inicial (padrao 101)')
    a = p.parse_args()
    if not 1 <= a.repeticoes <= 20:
        p.error('--repeticoes entre 1 e 20')
    if a.primeira_seed < 0:
        p.error('--primeira-seed deve ser >= 0')
    return a


def metricas(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = (score >= 0).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'ba': float(balanced_accuracy_score(y, pred)),
        'auc': float(roc_auc_score(y, score)),
        'sensibilidade': float(tp / (tp + fn)),
        'especificidade': float(tn / (tn + fp)),
        'acuracia': float(np.mean(pred == y)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def selecionar_so_treino(xt, xv, y, nomes, k):
    """Imputacao, Variancia, ANOVA e escala - FIT somente no treino."""
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    xt = imp.fit_transform(xt)
    xv = imp.transform(xv)
    var = VarianceThreshold(0.0)
    xt = var.fit_transform(xt)
    xv = var.transform(xv)
    nomes = np.asarray(nomes)[var.get_support()]
    if not len(nomes):
        raise RuntimeError('Variancia zero em todas as colunas.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sel = SelectKBest(f_classif, k=min(k, xt.shape[1]))
        xt = sel.fit_transform(xt, y)
        xv = sel.transform(xv)
    nomes = nomes[sel.get_support()]
    scl = StandardScaler()
    xt = scl.fit_transform(xt).astype(np.float32)
    xv = scl.transform(xv).astype(np.float32)
    return xt, xv, nomes


def svc_fit(x, y):
    model = SVC(kernel='linear', C=1.0, class_weight='balanced', cache_size=512)
    model.fit(x, y)
    return model


def load_matrix(root):
    meta, mats, paths = load_data(root, require_baselines=False)
    names = mats['WAVELET'][1]
    ix = representations(names, 'teste')['WPT_sym5']
    x = mats['WAVELET'][0][:, ix]
    names = names[ix]
    # Exige exatamente as 6 tarefas para comparar arquiteturas na mesma coorte.
    cnt = meta.groupby('participant_id').task.nunique()
    ids = sorted(cnt[cnt == 6].index)
    df = meta[meta.participant_id.isin(ids)].copy()
    if len(ids) != 410 or len(df) != 2460 or not (df.groupby('participant_id').task.apply(lambda z: set(z) == set(TASKS))).all():
        raise RuntimeError('Esperados 410 participantes com seis tarefas 004-009; pare e revise entradas.')
    if df.duplicated(['participant_id', 'task']).any():
        raise RuntimeError('Duas gravacoes da mesma tarefa para um participante.')
    pid_idx = pd.Index(ids, name='participant_id')
    people = meta[['participant_id', 'y']].drop_duplicates().set_index('participant_id').loc[pid_idx].reset_index()
    # Montagem deterministica, preservando correspondencia com a matriz X original.
    lookup = pd.Series(np.arange(len(meta)), index=pd.MultiIndex.from_frame(meta[['participant_id', 'task']]))
    order = np.array([int(lookup.loc[(pid, task)]) for pid in ids for task in TASKS], dtype=int)
    xx = np.asarray(x[order], dtype=np.float32).reshape(len(ids), len(TASKS), -1)
    if len(names) != 1920 or xx.shape != (410, 6, 1920) or not np.isfinite(xx).all():
        raise RuntimeError(f'WPT sym5 incorreta. Tensor={xx.shape}, nomes={len(names)}')
    return people, xx, np.asarray(names, dtype=str), paths[:2]


def fit_fold(train, test, y_train, names, fold, seed):
    """Dados 3D = pessoa x frase x feature. Retorna scores, selecoes e metricas."""
    ntr = len(y_train)
    nte = len(test)
    records = []
    preds = {}
    # 1) Pooling de todas as frases, baseline equivalente ao experimento 07.
    pt = train.reshape(ntr * 6, -1)
    pv = test.reshape(nte * 6, -1)
    yt = np.repeat(y_train, 6)
    zt, zv, fn = selecionar_so_treino(pt, pv, yt, names, 128)
    model = svc_fit(zt, yt)
    preds['POOLED'] = model.decision_function(zv).reshape(nte, 6).mean(axis=1)
    for n in fn:
        records.append((seed, fold, 'POOLED', 'TODAS', n))
    # 2) Um modelo por frase e media igual dos scores SVM (fusão tardia).
    # 3) Selecionar um bloco por frase (32 cada, 192 colunas), depois um SVC.
    task_scores = []
    blocks_train = []
    blocks_test = []
    for idx, task in enumerate(TASKS):
        xt = train[:, idx, :]
        xv = test[:, idx, :]
        zt, zv, selected = selecionar_so_treino(xt, xv, y_train, names, 128)
        task_scores.append(svc_fit(zt, y_train).decision_function(zv))
        for n in selected:
            records.append((seed, fold, 'FUSAO_TARDIA', task, n))
        bt, bv, selected_block = selecionar_so_treino(xt, xv, y_train, names, 32)
        blocks_train.append(bt)
        blocks_test.append(bv)
        for n in selected_block:
            records.append((seed, fold, 'BLOCOS_POR_FRASE', task, n))
    preds['FUSAO_TARDIA'] = np.stack(task_scores, axis=1).mean(axis=1)
    # Blocos ja padronizados com parametros do treino, preservam identidade da frase.
    block_tr = np.concatenate(blocks_train, axis=1)
    block_te = np.concatenate(blocks_test, axis=1)
    preds['BLOCOS_POR_FRASE'] = svc_fit(block_tr, y_train).decision_function(block_te)
    if any(len(sc) != nte or not np.isfinite(sc).all() for sc in preds.values()):
        raise RuntimeError('Scores invalidos no fold.')
    return preds, records


def fingerprint(root, paths, a):
    files = []
    for path in paths:
        p = Path(path)
        st = p.stat()
        files.append([str(p.resolve()), int(st.st_size), int(st.st_mtime_ns)])
    return hashlib.sha256(json.dumps({
        'version': VERSION, 'files': files, 'seeds': [a.primeira_seed, a.repeticoes],
        'models': ARMS, 'threshold': 0.0, 'cohort': 410,
    }, sort_keys=True).encode('utf8')).hexdigest()


def salvar_csv(df, target):
    target = Path(target)
    tmp = target.with_suffix('.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8-sig')
    tmp.replace(target)


def agrupar_descritores(feature):
    # Feature names, exemplo WPT__sym5__L6__aaad...__entropy__q90.
    import re
    desc = next((v for v in ('rel_energy','log_energy','entropy','kurtosis','zcr')
                 if f'__{v}__' in feature), 'outro')
    level = re.search(r'__L([56])__', feature)
    temporal = feature.split('__')[-1]
    return pd.Series({'nivel':f'L{level.group(1)}' if level else 'outro',
                      'descritor':desc, 'resumo_temporal':temporal})


def main():
    a = args_parser()
    root = Path(a.dados).resolve()
    dest = Path(a.saida)
    if not dest.is_absolute():
        dest = asset(root, dest)
    dest.mkdir(parents=True, exist_ok=True)
    ck = dest / 'checkpoints'
    ck.mkdir(exist_ok=True)
    people, tensor, names, paths = load_matrix(root)
    sig = fingerprint(root, paths, a)
    cp = dest / 'configuracao.json'
    if cp.exists():
        old = json.loads(cp.read_text(encoding='utf8'))
        if old.get('assinatura') != sig:
            raise RuntimeError('Saida existente com entradas/opcoes diferentes. Use --saida outra_pasta.')
    else:
        cp.write_text(json.dumps({'assinatura':sig, 'versao':VERSION,
                                  'repeticoes':a.repeticoes, 'primeira_seed':a.primeira_seed,
                                  'coorte':410, 'arms':ARMS, 'arquivo_entradas':[str(q) for q in paths]},
                                 ensure_ascii=False,indent=2),encoding='utf8')
    ids = people.participant_id.to_numpy()
    y = people.y.to_numpy(dtype=int)
    print(f'Coorte: {len(y)} pessoas | normais={sum(y==0)} patologicos={sum(y==1)} | tensor WPT={tensor.shape}',flush=True)
    print('Arquiteturas fixas:',', '.join(ARMS),'| SVC linear C=1 balanceado | corte=zero',flush=True)
    print('Todas as transformacoes ajustadas somente com treino. Sem leitura de WAV.',flush=True)
    for seed in range(a.primeira_seed, a.primeira_seed+a.repeticoes):
        pred_path=ck / f'predicoes_{seed}.csv'
        feat_path=ck / f'atributos_{seed}.csv'
        fold_path=ck / f'folds_{seed}.csv'
        if pred_path.exists() and feat_path.exists() and fold_path.exists():
            print('Seed',seed,'checkpoint concluido',flush=True)
            continue
        rows=[]; features=[]; folds=[]
        cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for fold, (tr, te) in enumerate(cv.split(ids,y),1):
            yt = y[tr]
            preds, selected = fit_fold(tensor[tr], tensor[te],yt,names,fold,seed)
            features.extend(selected)
            for arm, sc in preds.items():
                for i,s in zip(te,sc):
                    rows.append({'seed':seed,'fold':fold,'arm':arm,'participant_id':ids[i],
                                 'y':int(y[i]),'score':float(s)})
                fmet = metricas(y[te], sc)
                folds.append({'seed':seed,'fold':fold,'arm':arm,'n':len(te),**fmet})
            print(f'  seed {seed} fold {fold}/5: '
                  + ', '.join(f'{k} BA={metricas(y[te],p)["ba"]:.3f}' for k,p in preds.items()),flush=True)
        frame = pd.DataFrame(rows)
        for arm in ARMS:
            arm_frame = frame[frame.arm==arm]
            if len(arm_frame)!=len(ids) or arm_frame.participant_id.nunique()!=len(ids) or arm_frame.participant_id.duplicated().any():
                raise RuntimeError(f'OOF incompleto: {arm}, seed {seed}')
        salvar_csv(frame,pred_path)
        salvar_csv(pd.DataFrame(features,columns=['seed','fold','arm','tarefa','feature']),feat_path)
        salvar_csv(pd.DataFrame(folds),fold_path)
        print('Seed',seed,'completa:',', '.join(f'{arm} BA={metricas(frame.loc[frame.arm==arm,"y"],frame.loc[frame.arm==arm,"score"])["ba"]:.4f}'
                       for arm in ARMS),flush=True)
    pred = pd.concat([pd.read_csv(ck/f'predicoes_{seed}.csv')
                      for seed in range(a.primeira_seed,a.primeira_seed+a.repeticoes)],ignore_index=True)
    feat = pd.concat([pd.read_csv(ck/f'atributos_{seed}.csv')
                      for seed in range(a.primeira_seed,a.primeira_seed+a.repeticoes)],ignore_index=True)
    fl = pd.concat([pd.read_csv(ck/f'folds_{seed}.csv')
                    for seed in range(a.primeira_seed,a.primeira_seed+a.repeticoes)],ignore_index=True)
    perf=[]
    for (seed, arm),grp in pred.groupby(['seed','arm'],sort=True):
        perf.append({'seed':seed,'arm':arm,'participantes':len(grp),**metricas(grp.y,grp.score)})
    perf = pd.DataFrame(perf)
    summary = perf.groupby('arm',as_index=False).agg(ba_media=('ba','mean'),ba_min=('ba','min'),
                  ba_max=('ba','max'),ba_dp=('ba','std'),auc_media=('auc','mean'),
                  sens_media=('sensibilidade','mean'),esp_media=('especificidade','mean'))
    summary['ba_dp']=summary.ba_dp.fillna(0)
    ref=perf[perf.arm=='POOLED'].set_index('seed')
    diff=[]
    for arm in ARMS[1:]:
        other=perf[perf.arm==arm].set_index('seed')
        delta=other.ba-ref.ba
        diff.append({'alternativa':arm,'delta_ba_medio':float(delta.mean()),
                     'delta_ba_min':float(delta.min()),'delta_ba_max':float(delta.max()),
                     'repeticoes_com_ganho':int((delta>0).sum()),
                     'total_repeticoes':len(delta)})
    freq=feat.groupby(['arm','tarefa','feature'],as_index=False).size().rename(columns={'size':'selecoes'})
    parts=freq.feature.apply(agrupar_descritores)
    freq=pd.concat([freq,parts],axis=1)
    salvar_csv(pred,dest/'predicoes_oof_todas_seeds.csv')
    salvar_csv(perf,dest/'desempenho_por_seed.csv')
    salvar_csv(fl,dest/'desempenho_por_fold.csv')
    salvar_csv(summary,dest/'resumo_arquiteturas.csv')
    salvar_csv(pd.DataFrame(diff),dest/'diferencas_pareadas.csv')
    salvar_csv(freq.sort_values('selecoes',ascending=False),dest/'frequencia_features_selecionadas.csv')
    lines=['# AVFAD 09 — Identidade das frases na WPT sym5','',
           '- Objetivo: testar se preservar a identidade das seis tarefas melhora WPT temporal ja extraida.',
           '- Coorte: 410 pessoas (81 normais, 329 patologicas), as mesmas seis frases por pessoa.',
           f'- Seeds: {a.primeira_seed}..{a.primeira_seed+a.repeticoes-1}, 5 folds estratificados por pessoa.',
           '- Modelos PREDEFINIDOS: SVC linear C=1, class_weight=balanced, limiar zero.',
           '- POOLED: 128 features, treinamento por gravacao, media dos scores por pessoa.',
           '- FUSAO_TARDIA: 6 modelos separados, 128 features por tarefa, media igual das margens.',
           '- BLOCOS_POR_FRASE: 32 features escolhidas por tarefa, 192 concatenadas, modelo por pessoa.',
           '- Selecoes/escala/treino somente com participantes do treino de cada fold.',
           '- Todas as 3 arquiteturas compartilham exatamente as mesmas pessoas e folds em cada seed.',
           '- OOF significa cada pessoa testada fora do treino naquela divisao.',
           '- Seeds reutilizam pessoas: os resultados entre seeds nao sao amostras independentes.',
           '- Experimento exploratorio proposto apos inspecao do experimento 07. Nao e validacao externa nem estimativa livre de viés de multiplas tentativas.',
           '', '## Desempenho entre divisoes',
           '| Arquitetura | BA media | BA min | BA max | AUC media | Sens media | Esp media |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary.itertuples(index=False):
        lines.append(f'| {row.arm} | {row.ba_media:.4f} | {row.ba_min:.4f} | {row.ba_max:.4f} | {row.auc_media:.4f} | {row.sens_media:.4f} | {row.esp_media:.4f} |')
    lines += ['', '## Diferencas em relacao ao treinamento POOLED',
              '| Alternativa | Delta BA medio | Delta min | Delta max | Seeds com ganho |',
              '|---|---:|---:|---:|---:|']
    for z in diff:
        lines.append(f'| {z["alternativa"]} | {z["delta_ba_medio"]:+.4f} | {z["delta_ba_min"]:+.4f} | {z["delta_ba_max"]:+.4f} | {z["repeticoes_com_ganho"]}/{z["total_repeticoes"]} |')
    lines += ['', '## Frequencia de features selecionadas',
              'Veja frequencia_features_selecionadas.csv para bandas L5/L6, descritores e resumo temporal.',
              'As frequencias sao exploratorias e as selecoes entre seeds sao correlacionadas.',
              '', '## Interpretacao',
              'Resultados acima de 80% apenas em sementes selecionadas nao sao validacao clinica.',
              'Para uma estimativa confirmatoria, congelar arquitetura e utilizar coorte externa ou teste nao usado no desenvolvimento.']
    (dest/'RELATORIO_09.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(f'\n=== RESUMO EM {a.repeticoes} SEED(S) ===',flush=True)
    for r in summary.itertuples(index=False):
        print(f'{r.arm}: BA media {r.ba_media:.4f} ({r.ba_min:.4f}..{r.ba_max:.4f}), AUC {r.auc_media:.4f}, sens {r.sens_media:.3f}, esp {r.esp_media:.3f}',flush=True)
    for z in diff:
        print(f'  {z["alternativa"]}: delta BA vs POOLED {z["delta_ba_medio"]:+.4f}; ganho em {z["repeticoes_com_ganho"]}/{z["total_repeticoes"]} seeds',flush=True)
    print('Relatorio:',dest/'RELATORIO_09.md',flush=True)
    print('Nao houve extracao de audio. Saidas anteriores nao foram modificadas.',flush=True)

if __name__ == '__main__':
    main()
