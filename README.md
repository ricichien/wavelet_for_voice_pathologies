# Deteccao experimental de patologias vocais por wavelets

Projeto de pesquisa e TCC para investigar representacoes acusticas de voz com
Transformada Wavelet Discreta (DWT), Wavelet Packet Transform (WPT) e Wavelet
Scattering. MFCC e ComParE sao usados como referencias comparativas. Ha tambem
experimentos com redes temporais e representacoes auto-supervisionadas
(wav2vec/XLS-R).

Este repositorio contem codigo e documentacao, nao os corpora privados nem os
resultados locais. Os experimentos sao exploratorios e **nao constituem
validacao ou ferramenta de diagnostico clinico**.

## Metodologia em resumo

- Extracao de atributos acusticos e wavelet a partir das bases locais AVFAD e
  SVD; alguns experimentos tambem comparam representacoes ComParE e MFCC.
- Avaliacao com separacao e agregacao por participante, incluindo validacao
  cruzada aninhada em partes dos experimentos, para reduzir vazamento entre
  gravacoes da mesma pessoa.
- Comparacao de tarefas, familias wavelet, normalizacao, selecao de atributos,
  classificadores e modelos temporais/auto-supervisionados.
- Os detalhes de cada protocolo, criterios de inclusao e configuracoes estao
  nos scripts correspondentes. Resultados de experimentos locais nao sao
  distribuidos neste repositorio.

## Estrutura

| Caminho | Conteudo | Publicacao |
|---|---|---|
| `src/` | Extracao, analise, modelagem, verificacoes e testes | Codigo |
| `SVD/` | Pipelines e rotinas de analise da base SVD | Codigo |
| `docs/` | Documentacao segura do projeto | Documentacao |
| `requirements*.txt` | Dependencias separadas por pipeline | Configuracao |
| `data/` | Dados locais e derivados | Ignorado; nao publicar |
| `models/` | Pesos de modelos locais | Ignorado; licenca/procedencia a confirmar |
| `checkpoints/` | Checkpoints e caches de execucao | Ignorado; nao publicar automaticamente |
| `results/` | Resultados e artefatos experimentais | Ignorado; revisar antes de qualquer compartilhamento |
| `.venv*/` | Ambientes Python locais | Ignorado; recriar em cada maquina |
| `backups/` | Copias e arquivos antigos locais | Ignorado |

O caminho da base nao e enviado ao Git. Para os scripts que permitem variaveis
de ambiente, configure os caminhos locais conforme a secao de privacidade.

## Catalogo de scripts

### Experimentos numerados em `src/`

| Script | Finalidade |
|---|---|
| `00_verificar_entradas.py` | Verificar entradas e manifesto antes da execucao |
| `01_otimizar_wavelet_nested.py` | Otimizacao wavelet com validacao aninhada |
| `02_resumir_resultados.py` | Consolidar resultados e comparacoes pareadas |
| `03_criar_coortes_pareadas.py` | Construir coortes pareadas para analise |
| `04_varrer_tarefas_exploratorio.py` | Explorar tarefas e configuracoes |
| `05_diagnosticar_queda_nested.py` | Diagnosticar diferencas entre protocolos nested |
| `06_avaliar_robustez_folds_wavelet.py` | Avaliar robustez entre folds |
| `07_investigar_tarefas_wavelet.py` | Investigar tarefas wavelet |
| `08_erros_por_patologia_avfad.py` | Analise de erros por classe na AVFAD |
| `09_fusao_frases_wavelet.py` | Fusao de predicoes por frase |
| `10_verificar_scattering.py` | Verificar dependencias/configuracao de scattering |
| `11_extrair_scattering_avfad.py` | Extrair scattering para AVFAD |
| `12_avaliar_scattering_avfad.py` | Avaliar atributos de scattering |
| `13_otimizar_wpt_sym5.py` | Otimizar configuracao WPT sym5 |
| `14_ablacao_busca_limiar_wpt.py` | Ablacao e busca de limiar WPT |
| `15_avaliar_wpt_compacta_participante.py` | Avaliar WPT compacta por participante |
| `16_avaliar_svd_wavelet_participante.py` | Avaliar wavelet SVD por participante |
| `17_robustez_distribuicao_wpt_avfad.py` | Analisar distribuicao/robustez WPT |
| `18_extrair_wpt_16khz_avfad.py` | Extrair WPT a 16 kHz |
| `19_avaliar_wpt_16khz_avfad.py` | Avaliar atributos WPT a 16 kHz |
| `20_extrair_wav2vec_avfad.py` | Extrair representacoes wav2vec |
| `21_avaliar_wpt_wav2vec_avfad.py` | Avaliar combinacoes WPT/wav2vec |
| `22_regularizar_wpt_wav2vec_avfad.py` | Regularizar modelos WPT/wav2vec |
| `23_extrair_wpt_rms_sequencias_avfad.py` | Extrair sequencias WPT/RMS |
| `24_avaliar_normalizacao_wpt_avfad.py` | Avaliar normalizacao WPT |
| `25_treinar_wavelet_cnn_tcn_avfad.py` | Treinar CNN/TCN com atributos wavelet |
| `26_hibrido_wavelet_temporal_wav2vec.py` | Avaliar modelo hibrido temporal/wav2vec |
| `27_extrair_camadas_ssl_avfad.py` | Extrair camadas de modelo SSL |
| `28_avaliar_camadas_ssl_wavelet_avfad.py` | Avaliar camadas SSL e wavelet |
| `29_extrair_vogal_wavelet_ssl_avfad.py` | Extrair atributos de vogais com SSL |
| `30_avaliar_frase_vogal_wavelet_avfad.py` | Avaliar frase/vogal com SSL e wavelet |
| `31_prefixo_xlsr_lora.py` | Experimentos de prefixo e LoRA com XLS-R |
| `32_adaptar_prefixo_wavelet_avfad.py` | Adaptar prefixo wavelet para AVFAD |

### Modulos e utilitarios em `src/`

- **Nucleo e caminhos:** `avfad_core.py`, `avfad_paths.py`,
  `scattering_avfad_common.py`, `wavelet_participant_selection.py` e
  `svd_cohort_manifest.py`.
- **Extracao:** `extrair_features_avfad.py`,
  `extrair_features_capev_avfad.py`,
  `extrair_features_capev_compare_avfad.py`, `extrair_features_prt_avfad.py`,
  `extrair_wavelet_temporal_avfad.py`,
  `extrair_wavelet_temporal_avfad_reduzido.py`,
  `preparar_modelo_wav2vec.py`.
- **Modelagem e analise:** `modelar_avfad.py`,
  `modelar_capev_compare_avfad.py`, `modelar_classificadores_avfad.py`,
  `modelar_pareamento_prt_avfad.py`, `modelar_wavelet_temporal_avfad.py`,
  `modelar_wavelet_temporal_avfad_corrigido.py`, `analise_hsa.py`,
  `avaliar_pareamento_avfad.py`, `avaliar_praat_avfad.py`,
  `auditar_alvo_tarefas27.py`, `auditar_compare_avfad.py`, `auditar_wavs.py`.
- **Validacao, diagnostico e campanhas:** `validar_dataset.py`,
  `verificar_avfad_para_tcc.py`, `verificar_integridade.py`,
  `verificar_lista.py`, `diagnosticar_prt_avfad.py`, `diagnostico_avfad.py`,
  `diagnostico_manifesto.py`, `diagnostico_zips.py`, `inventario_audio.py`,
  `inspecionar.py`, `resumo_avfad.py`, `congelar_fontes_experimento.py`,
  `executar_campanha_25_26.py`, `executar_campanha_27_28.py`,
  `executar_campanha_29_30.py`, `gerar_documentacao_23_26.py`.
- `test_*.py` contem testes automatizados e verificacoes de regressao para os
  experimentos numerados e modulos compartilhados.

### Pipelines em `SVD/`

- `svd_pipeline_wpt_temporal.py` e
  `svd_pipeline_wpt_temporal_HEALTHY.py`: extracao/avaliacao WPT temporal.
- `main.py`, `experimentos_finais.py`, `checagens_defesa.py` e
  `checagens_idade.py`: execucoes e verificacoes de experimentos SVD.
- `rodar_*.bat`: atalhos Windows locais; revise os argumentos e caminhos antes
  de executar em outra maquina.

Os nomes dos scripts indicam a finalidade geral; consulte o codigo e os
argumentos de cada script para os detalhes especificos e requisitos de entrada.

## Instalacao e dependencias

Use uma versao de Python compativel com o pipeline escolhido. Os scripts
principais e de SVD requerem Python 3.10 ou superior; a configuracao SVD foi
documentada para Python 3.12. Pipelines distintos podem exigir versoes de
dependencias diferentes. Crie ambientes separados em vez de combinar todos os
requisitos:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- `requirements.txt`: dependencias numericas e de classificacao do nucleo.
- `requirements_scattering.txt`: dependencias adicionais do pipeline
  Scattering; instale em ambiente isolado se houver conflitos.
- `requirements_wav2vec_extracao.txt`: versoes fixadas para o pipeline
  wav2vec/XLS-R. O PyTorch deve ser instalado separadamente conforme a versao
  de Python e o indice CPU/CUDA compativel, usando as instrucoes oficiais do
  PyTorch.
- `SVD/requirements.txt`: dependencias do pipeline SVD, incluindo audio,
  wavelets e classificadores opcionais.

Os manifests nao cobrem necessariamente todos os pacotes opcionais de todos
os scripts, e nao ha lockfile completo para reproducoes bit a bit. Registre a
versao de Python, sistema operacional, bibliotecas e configuracoes efetivamente
usadas em cada execucao.

## Execucao dos experimentos

1. Obtenha acesso autorizado aos dados e confirme que sua licenca permite o
   uso pretendido. Mantenha-os fora do repositorio.
2. Crie o ambiente correspondente ao pipeline e instale somente o manifest
   necessario.
3. Configure caminhos de dados e modelos localmente. Quando disponivel, prefira
   argumentos CLI ou variaveis de ambiente a caminhos absolutos no codigo.
4. Leia o script escolhido e confirme entradas, saidas, protocolo e custo antes
   de iniciar. Extracoes de audio/modelos podem consumir muita memoria, disco,
   tempo e GPU.
5. Execute primeiro verificacoes e testes aplicaveis. Guarde logs e resultados
   em pastas locais ignoradas pelo Git.

Exemplos de entrada:

```powershell
python src\00_verificar_entradas.py
python src\20_extrair_wav2vec_avfad.py
python SVD\main.py
```

Esses comandos nao fornecem dados nem garantem execucao sem configuracao; os
scripts podem exigir argumentos, arquivos locais, modelo baixado e recursos
especificos. Consulte `--help` quando suportado e o cabecalho do script antes
de rodar um experimento completo.

Os scripts SVD aceitam configurar os dados localmente pelas variaveis
`AVFAD_SVD_DATA_DIR`, `AVFAD_SVD_SOURCE_ROOT`,
`AVFAD_SVD_HEALTHY_ZIP` e `AVFAD_SVD_FEATURES_CSV`, conforme o pipeline. O
manifesto de coorte tambem exige `AVFAD_EXPECTED_MIXED_IDS` para a verificacao
contra a lista privada congelada. Defina-a apenas no ambiente local autorizado;
nao grave identificadores de participantes no Git, em logs publicos ou neste
README.

## Testes

Instale `pytest` no ambiente de desenvolvimento e, na raiz do repositorio,
execute:

```powershell
python -m pytest src -q
```

Os testes que usam dados, pesos ou configuracao local podem precisar desses
recursos e nao devem ser executados contra bases sem autorizacao. Os testes
sinteticos nao substituem a validacao ponta a ponta nem a revisao do protocolo.

## Resultados e reprodutibilidade

Resultados, caches, checkpoints, features extraidas e pesos locais nao sao
incluidos. Para reproduzir, sao necessarios os dados licenciados originais, os
modelos externos pertinentes, as dependencias e os parametros documentados em
cada experimento. Alguns processos sao longos e podem nao ser deterministas
entre hardware, CUDA, bibliotecas ou versoes do sistema.

Nao interprete resultados exploratorios como desempenho clinico. Evite
selecionar participantes, folds, limiares ou configuracoes em funcao do
resultado de teste. Uma validacao clinica ou externa requer protocolo e dados
independentes apropriados.

## Privacidade, licencas e limitacoes

- Nao publique gravacoes, arquivos originais AVFAD/SVD, arquivos compactados
  dos datasets, metadados identificaveis, IDs individuais, atributos
  derivados, predicoes por pessoa, checkpoints ou resultados que possam
  revelar condicao clinica.
- Nao publique credenciais, caminhos pessoais, ambientes virtuais, caches ou
  artefatos cuja procedencia/licenca nao autorize redistribuicao.
- `data/`, `results/`, `models/`, `checkpoints/`, `backups/` e ambientes locais
  sao protegidos pelo `.gitignore`. Verifique o staging e os caminhos antes de
  cada commit; regras de ignore nao removem arquivos ja rastreados.
- Pesos de modelos externos so podem ser redistribuidos depois de confirmar
  origem oficial, licenca, integridade e limites da plataforma. Git LFS nao
  torna dados sensiveis ou licenciados apropriados para publicacao.
- Este projeto e destinado a pesquisa. Nao e dispositivo medico, nao fornece
  diagnostico e nao deve orientar atendimento individual.

Consulte [docs/ESTRUTURA_LOCAL.md](docs/ESTRUTURA_LOCAL.md) para notas de
organizacao local e configuracao de caminhos.
