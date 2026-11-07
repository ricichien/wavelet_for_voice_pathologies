# Organizacao local

O repositorio deve conter codigo, configuracoes de dependencias e
documentacao segura. Dados, modelos, caches e resultados sao recursos locais e
nao devem ser adicionados ao Git sem uma avaliacao especifica de privacidade,
licenca e necessidade.

## Pastas locais

- `data/`: bases, metadados e atributos derivados. Mantenha acesso restrito e
  siga as condicoes de uso/licenca de cada conjunto.
- `models/`: pesos externos. Confirme a origem, licenca e checksum antes de
  baixar ou reutilizar.
- `checkpoints/`: estado de execucoes e caches. Alguns podem ser caros de
  reconstruir; preserve em backup privado se forem importantes.
- `results/`: predicoes, relatorios, matrizes, features e resultados locais.
- `backups/`: versoes antigas e pacotes locais; nao e substituto de backup
  externo validado.
- `.venv/` e outros ambientes virtuais: recrie a partir dos manifests de
  dependencias; nao os versione.

## Caminhos configuraveis

Nao fixe caminhos de perfil de usuario ou de uma maquina no codigo. Use
argumentos de linha de comando ou variaveis de ambiente locais. Os scripts da
SVD usam, conforme o pipeline:

- `AVFAD_SVD_DATA_DIR`: diretorio local com arquivos do pipeline SVD.
- `AVFAD_SVD_SOURCE_ROOT`: raiz dos arquivos de origem usados pelo manifesto.
- `AVFAD_SVD_HEALTHY_ZIP`: caminho local opcional para o arquivo de controle.
- `AVFAD_SVD_FEATURES_CSV`: arquivo local de atributos para a avaliacao.
- `AVFAD_EXPECTED_MIXED_IDS`: lista privada necessaria para validar o conjunto
  congelado de participantes mistos no manifesto.

Configure esses valores apenas no ambiente local autorizado. Nao os grave em
arquivos rastreados, logs publicos ou comandos compartilhados. O manifesto
falha explicitamente se a lista privada de verificacao nao estiver definida.

## Regras de publicacao

O `.gitignore` exclui as pastas de dados, modelos, resultados, checkpoints,
backups e ambientes virtuais, alem de extensoes comuns de dados, features e
pesos. Antes de publicar, revise `git status`, `git diff --cached --stat` e a
lista de arquivos staged. Ignorar um caminho nao remove do historico um arquivo
que ja tenha sido commitado.
