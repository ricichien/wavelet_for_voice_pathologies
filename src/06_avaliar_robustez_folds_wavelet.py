# -*- coding: utf-8 -*-
"""AVFAD HSA: estabilidade por repeticoes de CV por participante, sem ler WAVs.

Diagnostico exploratorio: modelos FIXOS. Sem escolher o melhor fold ou ajustar
hiperparametros/limiar nos dados de teste. Varios seeds NAO criam novos sujeitos
nem constituem teste externo independente.

Requer avfad_core.py do pacote anterior, junto deste arquivo.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC

from avfad_core import load_data, representations, transform, collapse, metrics

VERSAO = '1.0'
MODELOS = {
    'WPT_sym5_SVC_linear': ('WAVELET', 'linear', 1.0, 128),
    'MFCC_SVC_RBF': ('MFCC', 'rbf', 1.0, 128),
    'ComParE_SVC_RBF': ('COMPARE', 'rbf', 1.0, 128),
}


def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados', default='.', help='Diretorio contendo arquivos NPZ e CSV extraidos')
    p.add_argument('--saida', default='resultados_robustez_folds_wavelet', help='Diretorio de saida (novo)')
    p.add_argument('--repeticoes', type=int, default=10, help='Numero de seeds por esquema (padrao 10)')
    p.add_argument('--primeira-seed', type=int, default=101, help='Seed inicial (padrao 101; sem 42 historico)')
    p.add_argument('--esquemas', choices=['SKF','SGKF','ambos'], default='SKF')
    p.add_argument('--incluir-compare', action='store_true', help='Tambem executa ComParE, aumentando o tempo de treinamento')
    a=p.parse_args()
    if not (1<=a.repeticoes<=100):p.error('--repeticoes deve ser entre 1 e 100')
    if a.primeira_seed<0:p.error('Seed nao pode ser negativa')
    return a


def make_split(people, scheme, seed):
    if scheme=='SKF':
        cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        return list(cv.split(np.zeros(len(people)),people.y.to_numpy()))
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
    return list(cv.split(np.zeros(len(people)), people.y.to_numpy(),groups=people.participant_id.to_numpy()))


def signature(paths,options):
    records=[]
    for path in paths:
        p=Path(path)
        st=p.stat()
        records.append([p.name,int(st.st_size),int(st.st_mtime_ns)])
    return hashlib.sha256(json.dumps({'version':VERSAO,'inputs':records,'params':options},sort_keys=True).encode()).hexdigest()


def predict_one(model_name, context, folds, seed, scheme):
    meta,mats,idx,people=context
    mat_name,kernel,c,k=MODELOS[model_name]
    x=np.asarray(mats[mat_name][0],dtype=np.float32)
    if mat_name=='WAVELET':
        x=x[:,idx]
    else:
        x=x
    preds=[]
    fold_metrics=[]
    for fold,(p_train,p_test) in enumerate(folds,1):
        ids_train=set(people.iloc[p_train].participant_id)
        ids_test=set(people.iloc[p_test].participant_id)
        if ids_train & ids_test:
            raise RuntimeError('Vazamento de participantes detectado')
        tr=meta.participant_id.isin(ids_train).to_numpy()
        te=meta.participant_id.isin(ids_test).to_numpy()
        y_train=meta.loc[tr,'y'].to_numpy()
        Xt,Xv,n_sel=transform(x[tr],x[te],y_train,k)
        clf=SVC(kernel=kernel,C=c,gamma='scale',class_weight='balanced',cache_size=512)
        clf.fit(Xt,y_train)
        pred=collapse(meta.loc[te].reset_index(drop=True),clf.decision_function(Xv))
        if len(pred)!=len(ids_test):
            raise RuntimeError('Numero de participantes no fold nao confere')
        if len(np.unique(pred.y))!=2:
            raise RuntimeError(f'Fold sem ambas as classes: {scheme}, seed={seed}, fold={fold}')
        pred['fold']=fold
        pred['seed']=seed
        pred['esquema']=scheme
        pred['modelo']=model_name
        preds.append(pred)
        fm=metrics(pred.y.to_numpy(),pred.score.to_numpy(),0.0)
        fold_metrics.append({'esquema':scheme,'seed':seed,'modelo':model_name,'fold':fold,
                             'pessoas':len(pred),'normais':int((pred.y==0).sum()),
                             'patologicos':int((pred.y==1).sum()),
                             'features_selecionadas':n_sel,**fm})
    pred=pd.concat(preds,ignore_index=True)
    if len(pred)!=len(people) or pred.participant_id.duplicated().any():
        raise RuntimeError('Predicoes OOF incompletas ou duplicadas')
    all_m=metrics(pred.y.to_numpy(),pred.score.to_numpy(),0.0)
    result={'esquema':scheme,'seed':seed,'modelo':model_name,'participantes':len(pred),**all_m}
    return pred, fold_metrics,result


def atomic_write(frame, path):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    frame.to_csv(tmp,index=False,encoding='utf-8-sig')
    tmp.replace(path)


def build_summary(results):
    grouped=results.groupby(['esquema','modelo'],sort=True)
    rows=[]
    for (esquema,modelo),g in grouped:
        row={'esquema':esquema,'modelo':modelo,'repeticoes':len(g)}
        for col in ('ba','auc','sensitivity','specificity'):
            v=g[col].to_numpy(dtype=float)
            row[col+'_media']=v.mean()
            row[col+'_dp_seeds']=v.std(ddof=1) if len(v)>1 else np.nan
            row[col+'_min']=v.min();row[col+'_max']=v.max()
            row[col+'_q10']=np.quantile(v,.10)
            row[col+'_mediana']=np.median(v)
            row[col+'_q90']=np.quantile(v,.90)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    a=cli()
    root=Path(a.dados).resolve()
    dest=Path(a.saida)
    if not dest.is_absolute():dest=asset(root, dest)
    dest.mkdir(parents=True,exist_ok=True)
    parts_dir=dest/'checkpoints';parts_dir.mkdir(exist_ok=True)
    meta,mats,paths=load_data(root,require_baselines=True)
    people=meta[['participant_id','y']].drop_duplicates().sort_values('participant_id').reset_index(drop=True)
    idx=representations(mats['WAVELET'][1],'teste')['WPT_sym5']
    names=['WPT_sym5_SVC_linear','MFCC_SVC_RBF']
    if a.incluir_compare:names.append('ComParE_SVC_RBF')
    schemes=['SKF','SGKF'] if a.esquemas=='ambos' else [a.esquemas]
    options={'models':names,'schemes':schemes,'primeira_seed':a.primeira_seed,
             'seeds':list(range(a.primeira_seed,a.primeira_seed+a.repeticoes)),
             'threshold':0.0,'class_weight':'balanced','k':128}
    sig=signature(paths,options)
    meta_file=dest/'configuracao.json'
    if meta_file.exists():
        old=json.loads(meta_file.read_text(encoding='utf-8'))
        if old.get('assinatura')!=sig:
            raise RuntimeError(f'Pasta {dest} contem outra configuracao/arquivo de entrada. Use --saida com novo nome.\n'
                               'Nao misture predicoes de experimentos diferentes.')
    else:
        meta_file.write_text(json.dumps({'versao':VERSAO,'assinatura':sig,'opcoes':options,
                                         'arquivos':[p.name for p in paths]},indent=2,ensure_ascii=False),encoding='utf-8')
    context=(meta,mats,idx,people)
    print(f'Participantes: {len(people)}; normais={sum(people.y==0)}; patologicos={sum(people.y==1)}')
    print(f'Modelos FIXOS: {names}; folds: {schemes}; seeds: {options["seeds"]}')
    print('Selecao de features e scaler SOMENTE no treino. Limiar=zero. Sem reextrair WAVs.')
    for scheme in schemes:
        for seed in options['seeds']:
            folds=make_split(people,scheme,seed)
            for model_name in names:
                file=parts_dir/f'{scheme}_seed{seed}_{model_name}.csv'
                folds_file=parts_dir/f'{scheme}_seed{seed}_{model_name}_folds.csv'
                result_file=parts_dir/f'{scheme}_seed{seed}_{model_name}_resultado.csv'
                if all(p.exists() for p in (file,folds_file,result_file)):
                    print(f'{scheme} seed={seed} {model_name}: checkpoint existente, pulando',flush=True)
                    continue
                pred,fold_results,result=predict_one(model_name,context,folds,seed,scheme)
                atomic_write(pred,file)
                atomic_write(pd.DataFrame(fold_results),folds_file)
                atomic_write(pd.DataFrame([result]),result_file)
                print(f'{scheme} seed={seed} {model_name}: BA={result["ba"]:.4f}, AUC={result["auc"]:.4f}, '
                      f'Sens={result["sensitivity"]:.3f}, Esp={result["specificity"]:.3f}',flush=True)
    # Somente arquivos compatíveis com os modelos e seeds da configuração ativa.
    pred_frames=[];fold_frames=[];result_frames=[]
    for scheme in schemes:
        for seed in options['seeds']:
            for name in names:
                base=f'{scheme}_seed{seed}_{name}'
                for collection,suffix in ((pred_frames,'.csv'),(fold_frames,'_folds.csv'),(result_frames,'_resultado.csv')):
                    f=parts_dir/(base+suffix)
                    if not f.exists():raise RuntimeError(f'Checkpoint incompleto: {f}')
                    collection.append(pd.read_csv(f))
    pred=pd.concat(pred_frames,ignore_index=True)
    fold_tbl=pd.concat(fold_frames,ignore_index=True)
    result_tbl=pd.concat(result_frames,ignore_index=True)
    atomic_write(pred,dest/'predicoes_por_participante_todas_seeds.csv')
    atomic_write(fold_tbl,dest/'desempenho_por_fold.csv')
    atomic_write(result_tbl,dest/'desempenho_por_repeticao.csv')
    stats=build_summary(result_tbl)
    atomic_write(stats,dest/'resumo_distribuicao.csv')
    paired=[]
    for scheme in schemes:
        a1=result_tbl[(result_tbl.esquema==scheme)&(result_tbl.modelo=='WPT_sym5_SVC_linear')].set_index('seed')
        for other in names[1:]:
            b1=result_tbl[(result_tbl.esquema==scheme)&(result_tbl.modelo==other)].set_index('seed')
            for seed in options['seeds']:
                paired.append({'esquema':scheme,'seed':seed,'comparacao':f'WPT_sym5_menos_{other}',
                               'delta_ba':float(a1.at[seed,'ba']-b1.at[seed,'ba']),
                               'delta_auc':float(a1.at[seed,'auc']-b1.at[seed,'auc'])})
    paired=pd.DataFrame(paired)
    atomic_write(paired,dest/'diferencas_pareadas_por_seed.csv')
    md=['# Robustez Wavelet por repeticoes de CV em participantes', '',
        f'- Participantes: {len(people)} (82 normais, 329 patologicos).',
        f'- Repeticoes por esquema: {a.repeticoes}, seeds {a.primeira_seed}..{a.primeira_seed+a.repeticoes-1}.',
        '- Metodologia: cada pessoa aparece em teste uma vez por repeticao; folds comuns a todos os modelos em cada repeticao.',
        '- Modelos: parametros fixos, SelectKBest(128) e StandardScaler ajustados so no treino de cada fold.',
        '- BA, AUC, sensibilidade e especificidade: metricas por participante com limiar zero.',
        '- ATENCAO: repeticoes de CV nao sao conjuntos externos independentes; desvio entre seeds NAO e IC95%.',
        '- ATENCAO: WPT sym5 e hiperparametros dos modelos foram escolhidos anteriormente nesta mesma coorte, portanto estimativas exploratorias.',
        '- Nunca escolher a melhor seed para divulgar como resultado final.', '',
        '## Resumo por esquema (BA)', '',
        '| Esquema | Modelo | Media | Min | Max | DP entre seeds |',
        '|---|---|---:|---:|---:|---:|']
    for row in stats.itertuples(index=False):
        md.append(f'| {row.esquema} | {row.modelo} | {row.ba_media:.4f} | {row.ba_min:.4f} | {row.ba_max:.4f} | {row.ba_dp_seeds:.4f} |')
    md+=['','## Diferencas pareadas de BA (WPT menos baseline)','',
         '| Esquema | Comparacao | Media da diferenca | Seeds com WPT superior |',
         '|---|---|---:|---:|']
    for (scheme,comp),g in paired.groupby(['esquema','comparacao']):
        md.append(f'| {scheme} | {comp} | {g.delta_ba.mean():+.4f} | {int((g.delta_ba>0).sum())}/{len(g)} |')
    md+=['','## Interpretacao correta','',
         'Uma variacao consideravel entre as seeds indica instabilidade da estimativa neste conjunto de dados.',
         'Nao significa necessariamente que os classificadores sejam errados nem prova capacidade de generalizacao externa.',
         'A meta de BA > 0.80 deve ser verificada em participantes completamente independentes e com selecao de modelo previamente definida.',
         'A proxima etapa, dependendo da robustez, e revisar representacao, estratificacao por patologia e/ou coorte pareada, nao buscar uma seed favoravel.']
    (dest/'RELATORIO_ROBUSTEZ.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('\n=== DISTRIBUICAO ENTRE SEEDS ===')
    for row in stats.itertuples(index=False):
        print(f'{row.esquema} | {row.modelo}: BA media={row.ba_media:.4f}, min={row.ba_min:.4f}, max={row.ba_max:.4f}; '
              f'AUC media={row.auc_media:.4f}',flush=True)
    print('\nRelatorio:',dest/'RELATORIO_ROBUSTEZ.md')
    print('CSV:',dest/'resumo_distribuicao.csv')
    print('Sem extracao de audio; nenhum resultado historico alterado.')

if __name__=='__main__':main()
