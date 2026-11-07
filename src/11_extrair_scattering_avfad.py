# -*- coding: utf-8 -*-
"""AVFAD CAPE-V 004-009: scattering 1D ordem 1/2, checkpoint SQLite.
Nao altera os cinco ZIPs originais e nao mistura piloto com extracao completa.
"""
from __future__ import annotations
from avfad_paths import asset, project_root, raw_dir
import argparse, json, sqlite3, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scattering_avfad_common import (load_manifest,zip_index,signature,make_scattering,
                                     load_audio,extract_one,VERSION)


def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados',default='.',help='Pasta raiz do repositorio')
    p.add_argument('--piloto-pessoas',type=int,default=0,help='0 = 411 pessoas; ex. 10 = piloto equilibrado (NAO avaliar desempenho)')
    p.add_argument('--max-janelas',type=int,default=64,help='Limite de janelas por frase, com amostragem uniforme')
    p.add_argument('--saida',default=None,help='Pasta de saida; caso omitido cria pasta distinta para piloto e full')
    a=p.parse_args()
    if a.piloto_pessoas<0 or (a.piloto_pessoas!=0 and not (2<=a.piloto_pessoas<=200)):
        p.error('--piloto-pessoas deve ser 0, ou entre 2 e 200')
    if a.max_janelas<2: p.error('--max-janelas precisa ser >=2')
    return a


def atomic_csv(frame,path):
    temp=path.with_name(path.name+'.tmp')
    frame.to_csv(temp,index=False,encoding='utf-8-sig')
    temp.replace(path)


def main():
    a=args();root=Path(a.dados).resolve()
    meta=load_manifest(root)
    if a.piloto_pessoas:
        people=meta[['participant_id','y']].drop_duplicates()
        n0=a.piloto_pessoas//2;n1=a.piloto_pessoas-n0
        if n0>82 or n1>329:raise ValueError('Piloto excedeu uma das classes')
        chosen=set(people[people.y==0].sort_values('participant_id').participant_id.iloc[:n0])
        chosen.update(people[people.y==1].sort_values('participant_id').participant_id.iloc[:n1])
        meta=meta[meta.participant_id.isin(chosen)].reset_index(drop=True)
    default_dest=f'resultados_10_scattering_piloto_{a.piloto_pessoas}' if a.piloto_pessoas else 'resultados_10_scattering_full'
    dest=Path(a.saida) if a.saida else Path(default_dest)
    if not dest.is_absolute():dest=asset(root, dest)
    if dest.resolve() == root:raise ValueError('Saida nao pode ser diretorio raiz do projeto')
    dest.mkdir(parents=True,exist_ok=True)
    idx,zips=zip_index(root,set(meta.sample_id))
    params={'sr':16000,'J':9,'Q':(8,1),'max_order':2,'win_s':.4,'hop_s':.2,
            'vad_top_db':40,'vad_frame':688,'vad_hop':160,'vad_gap_s':.25,'vad_min_s':.25,
            'max_windows':a.max_janelas,'rms_target':.1,'log': 'log1p(x100)',
            'piloto_pessoas':a.piloto_pessoas,'metadata_count':len(meta)}
    metadata_path=asset(root, 'metadata_wavelet_temporal_avfad_reduzido.csv')
    sig=signature(root,[metadata_path]+zips,params)
    cfg_path=dest/'configuracao_scattering.json'
    if cfg_path.exists():
        cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
        if cfg.get('assinatura')!=sig:raise RuntimeError(f'Saida {dest} tem configuracao diferente. Use outra pasta --saida.')
    else:
        cfg_path.write_text(json.dumps({'versao':VERSION,'assinatura':sig,'parametros':params,
                                        'arquivos_zips':[z.name for z in zips]},indent=2,ensure_ascii=False),encoding='utf-8')
    t0=time.monotonic()
    scat,paths,names,layout=make_scattering()
    print(f'Kymatio: {len(paths)} caminhos ordem 1/2, {len(names)} caracteristicas/gravacao; RMS normalizado a 0.1')
    print(f'Gravacoes: {len(meta)} | pessoas: {meta.participant_id.nunique()} | max janelas: {a.max_janelas}')
    print(f'Saida: {dest}')
    db=sqlite3.connect(dest/'checkpoint_scattering.sqlite',timeout=120)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS feats (sample_id TEXT PRIMARY KEY, payload BLOB NOT NULL, janela_total INT, janela_usada INT, regioes INT, proporcao_ativa REAL, duracao_ativa REAL, sr_original INT)')
    db.execute('CREATE TABLE IF NOT EXISTS erros (sample_id TEXT PRIMARY KEY, mensagem TEXT NOT NULL)')
    db.commit()
    done=set(row[0] for row in db.execute('SELECT sample_id FROM feats'))
    print(f'Checkpoint: {len(done)} gravacoes anteriores')
    expected_outputs=[dest/'features_scattering_avfad.npz',dest/'metadata_scattering_avfad.csv',dest/'caminhos_scattering.csv']
    if set(meta.sample_id).issubset(done) and all(p.exists() for p in expected_outputs):
        print('Extracao ja finalizada. Arquivos anteriores preservados (sem regravar ou mudar timestamps).')
        db.close()
        return
    for i,row in enumerate(meta.itertuples(index=False),1):
        if row.sample_id in done:continue
        try:
            x,sr=load_audio(idx[row.sample_id])
            vec,extra=extract_one(x,scat,paths,max_windows=a.max_janelas)
            if len(vec)!=len(names):raise RuntimeError('Vetores com comprimento errado')
            db.execute('INSERT OR REPLACE INTO feats VALUES (?,?,?,?,?,?,?,?)',(
                row.sample_id,sqlite3.Binary(vec.tobytes()),int(extra['n_windows_possible']),
                int(extra['n_windows']),int(extra['n_regions']),float(extra['ratio_active']),
                float(extra['duration_active_s']),int(sr)))
            db.execute('DELETE FROM erros WHERE sample_id=?',(row.sample_id,))
            db.commit()
        except KeyboardInterrupt:
            db.commit();raise
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}'
            db.execute('INSERT OR REPLACE INTO erros VALUES (?,?)',(row.sample_id,error));db.commit()
            print(f'  ERRO {row.sample_id}: {error}')
        if i%10==0 or i==len(meta):
            count=db.execute('SELECT COUNT(*) FROM feats').fetchone()[0]
            print(f'  {i}/{len(meta)}; concluidas no checkpoint: {count}; decorrido: {(time.monotonic()-t0)/60:.1f} min',flush=True)
    all_rows=db.execute('SELECT sample_id, payload, janela_total, janela_usada, regioes, proporcao_ativa, duracao_ativa, sr_original FROM feats').fetchall()
    sample_dict={r[0]:r for r in all_rows}
    failed=[sid for sid in meta.sample_id if sid not in sample_dict]
    if failed:
        rows=db.execute('SELECT * FROM erros').fetchall()
        atomic_csv(pd.DataFrame(rows,columns=['sample_id','erro']),dest/'erros_scattering.csv')
        db.close()
        raise RuntimeError(f'Extracao incompleta: {len(failed)} gravacoes sem vetor; consulte erros_scattering.csv. Pode executar novamente para retomar.')
    mat=np.stack([np.frombuffer(sample_dict[s][1],dtype='<f4') for s in meta.sample_id]).astype(np.float32)
    if mat.shape != (len(meta),len(names)) or not np.isfinite(mat).all():raise ValueError('Matriz final invalida')
    npz_path=dest/'features_scattering_avfad.npz'
    tmp=npz_path.with_suffix('.npz.tmp')
    with open(tmp,'wb') as f:np.savez_compressed(f,scattering=mat,feature_names=names)
    tmp.replace(npz_path)
    extra_df=pd.DataFrame([{'sample_id':sid,'janelas_candidatas':sample_dict[sid][2],
                            'janelas_usadas':sample_dict[sid][3],'regioes_voz':sample_dict[sid][4],
                            'proporcao_ativa':sample_dict[sid][5],'duracao_ativa_s':sample_dict[sid][6],
                            'sr_original':sample_dict[sid][7]} for sid in meta.sample_id])
    atomic_csv(meta.merge(extra_df,on='sample_id',validate='one_to_one'),dest/'metadata_scattering_avfad.csv')
    atomic_csv(layout,dest/'caminhos_scattering.csv')
    counts=layout.order.value_counts().to_dict()
    report=(f'AVFAD Scattering completo | modo={"PILOTO" if a.piloto_pessoas else "COORTE 411"}\n'
            f'Gravacoes={len(meta)}, pessoas={meta.participant_id.nunique()}, dimensao={mat.shape}\n'
            f'Kymatio J=9 Q=(8,1), ordens={counts}, sr=16000\n'
            f'Janelas por gravacao: mediana {extra_df.janelas_usadas.median():.0f}, max {extra_df.janelas_usadas.max()}\n'
            f'Gravacoes truncadas a {a.max_janelas} janelas: {(extra_df.janelas_candidatas>a.max_janelas).sum()}\n'
            'Sem modelagem, sem avaliacao e sem leitura de diagnosticos na extracao.\n')
    (dest/'RESUMO_EXTRACAO.md').write_text(report,encoding='utf-8')
    print(report)
    print('ARQUIVO:',npz_path)
    db.close()

if __name__=='__main__':main()
