# -*- coding: utf-8 -*-
"""Roda 01 para tarefas CAPE-V separadas, em pastas isoladas.

Analise exploratoria. Escolher a melhor tarefa pelos folds externos e depois
chama-la de avaliacao independente cria vies de selecao.
"""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dados',default='.')
    p.add_argument('--saida',default='resultados_tarefas_wavelet')
    p.add_argument('--tarefas',nargs='+',required=True,choices=['004','005','006','007','008','009'])
    p.add_argument('--preset',choices=['teste','rapido','completo'],default='rapido')
    p.add_argument('--baselines',choices=['nenhum','mfcc','compare','todos'],default='nenhum')
    o=p.parse_args()
    script=Path(__file__).parent/'01_otimizar_wavelet_nested.py'
    for task in o.tarefas:
        dest=Path(o.saida)/task
        cmd=[sys.executable,str(script),'--dados',o.dados,'--saida',str(dest),
             '--preset',o.preset,'--baselines',o.baselines,'--tarefa',task]
        print('Executando tarefa',task,flush=True)
        subprocess.run(cmd,check=True)
        report=[sys.executable,str(Path(__file__).parent/'02_resumir_resultados.py'),'--saida',str(Path(o.dados)/dest)]
        subprocess.run(report,check=True)
    print('Analise de tarefas concluida. Nao escolher vencedor sem nova validacao independente.')


if __name__=='__main__':main()
