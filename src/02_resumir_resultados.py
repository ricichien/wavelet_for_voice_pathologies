# -*- coding: utf-8 -*-
"""IC95% bootstrap por pessoa, deltas pareados, matriz de confusao e relatorio Markdown.

IMPORTANTE: os IC condicionam-se aos folds/algoritmos treinados e NAO sao
um intervalo de confianca de validacao externa independente.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from avfad_core import metrics


def boot_group(df,iterations,seed):
    rng=np.random.default_rng(seed)
    positive=np.flatnonzero(df.y.to_numpy()==1)
    negative=np.flatnonzero(df.y.to_numpy()==0)
    if len(positive)==0 or len(negative)==0:
        raise ValueError('Bootstrap exige duas classes.')
    ba=np.empty(iterations);auc=np.empty(iterations)
    y=df.y.to_numpy(int);s=df.score.to_numpy(float);thr=df.threshold.to_numpy(float)
    for i in range(iterations):
        ix=np.r_[rng.choice(positive,len(positive),replace=True),rng.choice(negative,len(negative),replace=True)]
        metric=metrics(y[ix],s[ix]-thr[ix],0.0)
        ba[i]=metric['ba']
        # AUC usa scores brutos; comparacoes entre folds dependem da escala das margens.
        auc[i]=metrics(y[ix],s[ix],0.0)['auc']
    return ba,auc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--saida',default='resultados_otimizacao_wavelet')
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261105)
    o=p.parse_args()
    root=asset(project_root(), o.saida)
    predfile=root/'predicoes_por_participante.csv'
    foldsfile=root/'escolhas_por_fold.csv'
    if not predfile.exists() or not foldsfile.exists():
        raise SystemExit('Execute 01_otimizar_wavelet_nested.py antes desta analise.')
    pds=pd.read_csv(predfile)
    folds=pd.read_csv(foldsfile)
    required={'participant_id','y','score','threshold','arm','fold'}
    if not required.issubset(pds.columns):raise SystemExit('Colunas esperadas ausentes nas predicoes.')
    arms=sorted(pds.arm.unique(),key=lambda a:(a!='WAVELET',a))
    n_by_arm=pds.groupby('arm').participant_id.nunique()
    if len(set(n_by_arm))!=1:
        raise SystemExit('Execucao parcial: quantidade de participantes difere entre bracos.')
    for arm in arms:
        part=pds[pds.arm==arm]
        if part.participant_id.duplicated().any():raise SystemExit(f'Predicao duplicada para {arm}.')
        if part.fold.nunique()!=folds[folds.arm==arm].fold.nunique():raise SystemExit('Folds incompletos.')
    tables=[];boot={};report=[]
    report.append('# Avaliacao aninhada — Wavelet AVFAD')
    report.append('')
    report.append('**Atencao:** hiperparametros, representacao e eventual limiar foram selecionados apenas nas dobras internas. Avaliacao externa por participante. Isto NAO equivale a validacao em outra base/centro.')
    report.append('')
    for i,arm in enumerate(arms):
        d=pds[pds.arm==arm].sort_values('participant_id').reset_index(drop=True)
        mm=metrics(d.y.to_numpy(),(d.score-d.threshold).to_numpy(),0.0)
        auc=metrics(d.y.to_numpy(),d.score.to_numpy(),0.0)['auc']
        vals=boot_group(d,o.bootstrap,o.seed+i*121)
        boot[arm]=vals
        row={'arm':arm,'participantes':len(d),**mm,'auc':auc,
             'ba_ci95_inf':float(np.quantile(vals[0],.025)),
             'ba_ci95_sup':float(np.quantile(vals[0],.975)),
             'auc_ci95_inf':float(np.quantile(vals[1],.025)),
             'auc_ci95_sup':float(np.quantile(vals[1],.975))}
        tables.append(row)
        report += [f'## {arm}',
                   f"- N={len(d)}, BA={mm['ba']:.4f} (IC95% {row['ba_ci95_inf']:.4f}–{row['ba_ci95_sup']:.4f}), AUC={auc:.4f}.",
                   f"- Sensibilidade={mm['sensitivity']:.4f}; especificidade={mm['specificity']:.4f}; acuracia={mm['accuracy']:.4f}; F1={mm['f1']:.4f}.",
                   f"- Matriz de confusao TN={mm['tn']}, FP={mm['fp']}, FN={mm['fn']}, TP={mm['tp']}.",'']
    summary=pd.DataFrame(tables)
    summary.to_csv(root/'resultado_final_resumo.csv',index=False,encoding='utf-8-sig')
    comparisons=[]
    if 'WAVELET' in arms:
        ref=pds[pds.arm=='WAVELET'].set_index('participant_id').sort_index()
        for idx,arm in enumerate(arms):
            if arm=='WAVELET':continue
            other=pds[pds.arm==arm].set_index('participant_id').sort_index()
            if not ref.index.equals(other.index) or not np.array_equal(ref.y.to_numpy(),other.y.to_numpy()):
                raise SystemExit('Nao ha pares de participantes identicos em Wavelet e '+arm)
            y=ref.y.to_numpy(int)
            pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
            s1=ref.score.to_numpy(float);t1=ref.threshold.to_numpy(float)
            s2=other.score.to_numpy(float);t2=other.threshold.to_numpy(float)
            rng=np.random.default_rng(o.seed+1000+idx)
            delta=np.empty(o.bootstrap);dauc=np.empty(o.bootstrap)
            for j in range(o.bootstrap):
                ix=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
                delta[j]=metrics(y[ix],s1[ix]-t1[ix])['ba']-metrics(y[ix],s2[ix]-t2[ix])['ba']
                dauc[j]=metrics(y[ix],s1[ix])['auc']-metrics(y[ix],s2[ix])['auc']
            d0=metrics(y,s1-t1)['ba']-metrics(y,s2-t2)['ba']
            row={'comparison':'WAVELET - '+arm,'delta_ba':d0,'delta_ba_ci95_low':float(np.quantile(delta,.025)),
                 'delta_ba_ci95_high':float(np.quantile(delta,.975)),
                 'delta_auc':metrics(y,s1)['auc']-metrics(y,s2)['auc'],
                 'delta_auc_ci95_low':float(np.quantile(dauc,.025)),
                 'delta_auc_ci95_high':float(np.quantile(dauc,.975))}
            comparisons.append(row)
            report += [f'## Comparacao pareada — WAVELET menos {arm}',
                       f"- ΔBA={d0:+.4f}; IC95% [{row['delta_ba_ci95_low']:+.4f}; {row['delta_ba_ci95_high']:+.4f}].",
                       f"- ΔAUC={row['delta_auc']:+.4f}; IC95% [{row['delta_auc_ci95_low']:+.4f}; {row['delta_auc_ci95_high']:+.4f}].",'']
    pd.DataFrame(comparisons).to_csv(root/'diferencas_pareadas.csv',index=False,encoding='utf-8-sig')
    report+=['## Escolhas por fold externo','', '| Braco | Fold | Representacao | Modelo | k | Limiar interno | BA externa |', '|---|---:|---|---|---:|---:|---:|']
    for _,r in folds.sort_values(['arm','fold']).iterrows():
        report.append(f"| {r['arm']} | {int(r['fold'])} | {r['representation']} | {r['model']} | {int(r['k'])} | {r['inner_threshold']:.3f} | {r['external_ba']:.3f} |")
    report+=['','## Limitacoes metodologicas',
             '- O objetivo de 80% de balanced accuracy nao deve ser atingido por escolha de resultados de folds externos ou por retirada oportunista de participantes.',
             '- A coorte HSA e desbalanceada e representa somente um centro, com possiveis confundidores adicionais.',
             '- Os baselines MFCC e ComParE foram extraidos do WAV inteiro. Wavelet temporal usa VAD e janelas de 400 ms: nao e comparacao isolada de transformada.',
             '- Os ICs bootstrap sao condicionais a estas predicoes OOF e nao consideram nova coleta nem total variabilidade de treinar novamente.',
             '- O AUC agregado usa scores provenientes de modelos de folds distintos e suas escalas podem diferir; interpretar com cautela.',
             '- Se posteriormente escolhermos entre diversos presets, coortes ou tarefas com base nestes resultados, surge novo vies de selecao. Validacao externa sera necessaria.',
             '- Nao utilizar os resultados como instrumento de diagnostico clinico.','']
    (root/'relatorio_final_wavelet.md').write_text('\n'.join(report),encoding='utf-8')
    print(summary.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    if comparisons:print('\n',pd.DataFrame(comparisons).to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print('\nRelatorio:',root/'relatorio_final_wavelet.md')


if __name__=='__main__':main()
