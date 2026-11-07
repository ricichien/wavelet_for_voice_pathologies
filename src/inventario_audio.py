from avfad_paths import asset, project_root, raw_dir
import os
import re
import csv
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# CONFIGURACAO
# ============================================================

PASTA = project_root()
PASTA_SAIDA = asset(PASTA, "saida")

ZIPS = [
    "AVFAD_01_00_00_1_README.zip",
    "AVFAD_01_00_00_2_A_to_C.zip",
    "AVFAD_01_00_00_3_D_to_L.zip",
    "AVFAD_01_00_00_4_M.zip",
    "AVFAD_01_00_00_5_N_to_Z.zip",
]

TAREFAS = {
    "001": "Vogal /i/",
    "002": "Vogal /a/",
    "003": "Vogal /u/",
    "004": "Frase CAPE-V (a confirmar)",
    "005": "Frase CAPE-V (a confirmar)",
    "006": "Frase CAPE-V (a confirmar)",
    "007": "Frase CAPE-V (a confirmar)",
    "008": "Frase CAPE-V (a confirmar)",
    "009": "Frase CAPE-V (a confirmar)",
    "010": "The North Wind and the Sun (a confirmar)",
    "011": "Fala espontanea (a confirmar)",
}

PADRAO_AUDIO = re.compile(r"^([A-Z]{3})(\d{3})\.wav$", re.IGNORECASE)


# ============================================================
# LEITURA DO CABECALHO WAV
# ============================================================

def ler_cabecalho_wav(arquivo_zip, info):
    """
    Le apenas o inicio do WAV dentro do ZIP.
    Nao extrai o arquivo inteiro.

    Importante:
    O chunk 'data' pode ser muito maior que os bytes lidos.
    Por isso, seu tamanho declarado nao pode ser usado
    para interromper a leitura antes de identifica-lo.
    """

    limite = 1024 * 1024

    with arquivo_zip.open(info, "r") as f:
        dados = f.read(limite)

    resultado = {
        "formato": "",
        "codec": "",
        "canais": "",
        "taxa_hz": "",
        "bits": "",
        "frames": "",
        "duracao_s": "",
        "data_size": "",
        "data_offset": "",
        "erro": "",
        "data_size_confere": "",
    }

    if len(dados) < 12:
        resultado["erro"] = "cabecalho muito pequeno"
        return resultado

    if dados[:4] not in (b"RIFF", b"RF64"):
        resultado["erro"] = "assinatura RIFF/RF64 ausente"
        return resultado

    if dados[8:12] != b"WAVE":
        resultado["erro"] = "assinatura WAVE ausente"
        return resultado

    pos = 12
    formato_tag = None
    canais = None
    taxa = None
    bits = None
    block_align = None
    data_offset = None
    data_size = None

    while pos + 8 <= len(dados):

        chunk_id = dados[pos:pos + 4]
        chunk_size = struct.unpack(
            "<I", dados[pos + 4:pos + 8]
        )[0]

        corpo = pos + 8

        # Identifica o bloco data ANTES de verificar
        # se ele inteiro cabe nos bytes lidos.
        if chunk_id == b"data":
            data_offset = corpo
            data_size = chunk_size
            break

        # Bloco de formato
        if chunk_id == b"fmt ":

            if chunk_size < 16:
                resultado["erro"] = "chunk fmt invalido"
                return resultado

            if corpo + 16 > len(dados):
                resultado["erro"] = "chunk fmt incompleto na leitura"
                return resultado

            formato_tag, canais, taxa, byte_rate, block_align, bits = (
                struct.unpack("<HHIIHH", dados[corpo:corpo + 16])
            )

            # WAVE_FORMAT_EXTENSIBLE
            if formato_tag == 0xFFFE:

                if chunk_size >= 40 and corpo + 40 <= len(dados):
                    subformat = struct.unpack(
                        "<H", dados[corpo + 24:corpo + 26]
                    )[0]

                    formato_tag = subformat

        # Para outros blocos, precisamos avançar ate o proximo.
        proxima_pos = corpo + chunk_size + (chunk_size & 1)

        if proxima_pos > len(dados):
            break

        pos = proxima_pos

    if formato_tag is None:
        resultado["erro"] = "chunk fmt nao encontrado"
        return resultado

    if data_offset is None:
        resultado["erro"] = "chunk data nao encontrado nos bytes lidos"
        return resultado

    # Nomes dos formatos mais comuns
    formatos = {
        1: "PCM",
        3: "IEEE_FLOAT",
        6: "A_LAW",
        7: "MU_LAW",
        0xFFFE: "WAVE_FORMAT_EXTENSIBLE",
    }

    nome_formato = formatos.get(
        formato_tag, f"DESCONHECIDO_{formato_tag}"
    )

    resultado["formato"] = nome_formato
    resultado["codec"] = formato_tag
    resultado["canais"] = canais
    resultado["taxa_hz"] = taxa
    resultado["bits"] = bits
    resultado["data_size"] = data_size
    resultado["data_offset"] = data_offset

    # Confere se o tamanho declarado do audio cabe no arquivo.
    disponivel = info.file_size - data_offset

    confere = (
        data_size >= 0
        and data_size <= disponivel
    )

    resultado["data_size_confere"] = confere

    if not confere:
        resultado["erro"] = "tamanho do chunk data inconsistente"
        return resultado

    if not block_align or not taxa:
        resultado["erro"] = "block_align ou taxa de amostragem invalida"
        return resultado

    frames = data_size // block_align
    duracao = frames / taxa

    resultado["frames"] = frames
    resultado["duracao_s"] = round(duracao, 4)

    return resultado


# ============================================================
# FUNCOES AUXILIARES
# ============================================================

def identificar_audio(nome):
    base = os.path.basename(nome)
    match = PADRAO_AUDIO.match(base)

    if not match:
        return None, None

    participante = match.group(1).upper()
    tarefa = match.group(2)

    return participante, tarefa


def identificar_pasta(nome):
    partes = nome.replace("\\", "/").split("/")

    if len(partes) < 2:
        return ""

    pasta = partes[-2].upper()

    if re.fullmatch(r"[A-Z]{3}", pasta):
        return pasta

    return ""


def formatar_duracao(valor):
    if valor == "":
        return ""

    return f"{float(valor):.2f}"


# ============================================================
# INVENTARIO
# ============================================================

def main():

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    arquivos_audio = []
    arquivos_outros = []

    extensoes_por_zip = defaultdict(Counter)

    participantes_tarefas = defaultdict(list)
    formatos = Counter()

    erros_cabecalho = []
    tamanhos_inconsistentes = []

    total_entradas = 0
    total_wav = 0
    total_prt = 0

    print("\nINICIANDO INVENTARIO AVFAD\n")

    for nome_zip in ZIPS:

        caminho_zip = asset(PASTA, nome_zip)

        if not caminho_zip.exists():
            print(f"\nERRO: arquivo nao encontrado: {nome_zip}")
            continue

        print(f"\nLendo {nome_zip} ...")

        with zipfile.ZipFile(caminho_zip, "r") as z:

            entradas = z.infolist()
            total = len(entradas)

            for indice, info in enumerate(entradas, start=1):

                if info.is_dir():
                    continue

                total_entradas += 1

                nome_interno = info.filename
                extensao = Path(nome_interno).suffix.lower()

                extensoes_por_zip[nome_zip][extensao] += 1

                if indice % 1000 == 0:
                    print(f"  {indice}/{total} entradas")

                # ------------------------------------------------
                # ARQUIVOS WAV
                # ------------------------------------------------

                if extensao == ".wav":

                    total_wav += 1

                    participante, tarefa = identificar_audio(
                        nome_interno
                    )

                    pasta_id = identificar_pasta(nome_interno)

                    cabecalho = ler_cabecalho_wav(z, info)

                    if cabecalho["erro"]:
                        erros_cabecalho.append(
                            (nome_interno, cabecalho["erro"])
                        )

                    if cabecalho["data_size_confere"] is False:
                        tamanhos_inconsistentes.append(nome_interno)

                    if cabecalho["formato"]:
                        chave = (
                            cabecalho["formato"],
                            cabecalho["canais"],
                            cabecalho["taxa_hz"],
                            cabecalho["bits"],
                        )
                        formatos[chave] += 1

                    if participante and tarefa:
                        participantes_tarefas[participante].append(
                            tarefa
                        )

                    id_confere = (
                        pasta_id == participante
                        if pasta_id and participante
                        else ""
                    )

                    registro = {
                        "pacote": nome_zip,
                        "arquivo": nome_interno,
                        "participante": participante or "",
                        "tarefa": tarefa or "",
                        "conteudo_provisorio": (
                            TAREFAS.get(tarefa, "")
                            if tarefa else ""
                        ),
                        "identificado": bool(participante and tarefa),
                        "pasta_id": pasta_id,
                        "id_confere_pasta": id_confere,
                        "tamanho_bytes": info.file_size,
                        "formato": cabecalho["formato"],
                        "codec": cabecalho["codec"],
                        "canais": cabecalho["canais"],
                        "taxa_hz": cabecalho["taxa_hz"],
                        "bits": cabecalho["bits"],
                        "frames": cabecalho["frames"],
                        "duracao_s": cabecalho["duracao_s"],
                        "data_size": cabecalho["data_size"],
                        "data_size_confere": cabecalho[
                            "data_size_confere"
                        ],
                        "erro": cabecalho["erro"],
                    }

                    arquivos_audio.append(registro)

                # ------------------------------------------------
                # ARQUIVOS PRT
                # ------------------------------------------------

                elif extensao == ".prt":

                    total_prt += 1

                    arquivos_outros.append({
                        "pacote": nome_zip,
                        "arquivo": nome_interno,
                        "extensao": extensao,
                        "tamanho_bytes": info.file_size,
                    })

                # ------------------------------------------------
                # OUTROS ARQUIVOS
                # ------------------------------------------------

                else:

                    arquivos_outros.append({
                        "pacote": nome_zip,
                        "arquivo": nome_interno,
                        "extensao": extensao,
                        "tamanho_bytes": info.file_size,
                    })

    # ============================================================
    # CSV DE AUDIO
    # ============================================================

    caminho_csv_audio = PASTA_SAIDA.joinpath( "inventario_audio.csv")

    if arquivos_audio:

        with open(
            caminho_csv_audio,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            escritor = csv.DictWriter(
                f,
                fieldnames=arquivos_audio[0].keys(),
                delimiter=";"
            )

            escritor.writeheader()
            escritor.writerows(arquivos_audio)

    # ============================================================
    # CSV DE OUTROS ARQUIVOS
    # ============================================================

    caminho_csv_outros = (
        PASTA_SAIDA.joinpath( "inventario_outros_arquivos.csv")
    )

    with open(
        caminho_csv_outros,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        campos = [
            "pacote",
            "arquivo",
            "extensao",
            "tamanho_bytes",
        ]

        escritor = csv.DictWriter(
            f,
            fieldnames=campos,
            delimiter=";"
        )

        escritor.writeheader()
        escritor.writerows(arquivos_outros)

    # ============================================================
    # ANALISE DE PARTICIPANTES E TAREFAS
    # ============================================================

    participantes = set(participantes_tarefas.keys())

    completos = 0
    faltando = []

    duplicados = []

    for participante, tarefas in participantes_tarefas.items():

        contagem = Counter(tarefas)

        esperadas = set(TAREFAS.keys())

        if esperadas.issubset(set(tarefas)):
            completos += 1
        else:
            faltantes = sorted(esperadas - set(tarefas))

            faltando.append(
                (participante, faltantes)
            )

        for tarefa, quantidade in contagem.items():

            if quantidade > 1:
                duplicados.append(
                    (participante, tarefa, quantidade)
                )

    ids_divergentes = [
        r for r in arquivos_audio
        if r["id_confere_pasta"] is False
    ]

    fora_padrao = [
        r for r in arquivos_audio
        if not r["identificado"]
    ]

    # ============================================================
    # DURACAO POR TAREFA
    # ============================================================

    duracoes = defaultdict(list)

    for r in arquivos_audio:

        tarefa = r["tarefa"]
        duracao = r["duracao_s"]

        if tarefa and duracao != "":
            duracoes[tarefa].append(float(duracao))

    # ============================================================
    # RELATORIO TXT
    # ============================================================

    linhas = []

    def escrever(texto=""):
        linhas.append(str(texto))

    escrever("=" * 72)
    escrever("RESUMO DO INVENTARIO DE AUDIO -- AVFAD")
    escrever("=" * 72)

    escrever()
    escrever("-- Arquivos por pacote e extensao --")

    for nome_zip in ZIPS:

        contagem = extensoes_por_zip[nome_zip]

        partes = [
            f"{ext or '[sem extensao]'}={qtd}"
            for ext, qtd in sorted(contagem.items())
        ]

        escrever(
            f"  {nome_zip:<40} "
            f"{sum(contagem.values()):>5} arquivos "
            f"({', '.join(partes)})"
        )

    escrever()
    escrever(f"  TOTAL: {total_entradas} arquivos "
             f"({total_wav} WAV + {total_entradas - total_wav} outros)")

    escrever()
    escrever("-- Comparacao com a documentacao --")
    escrever(f"  WAVs encontrados: {total_wav}")
    escrever(f"  Arquivos PRT encontrados: {total_prt}")
    escrever(f"  WAV + PRT: {total_wav + total_prt}")
    escrever("  Referencia documental: 8.648 arquivos")
    escrever(
        f"  Diferenca para WAV + PRT: "
        f"{8648 - (total_wav + total_prt)}"
    )
    escrever(
        f"  Outros arquivos nao WAV/PRT: "
        f"{total_entradas - total_wav - total_prt}"
    )

    escrever()
    escrever("-- WAVs fora do padrao <ID><tarefa>.wav --")
    escrever(f"  {len(fora_padrao)} arquivos")

    for r in fora_padrao[:15]:
        escrever(f"    {r['arquivo']}")

    if len(fora_padrao) > 15:
        escrever(
            "    ... lista completa em inventario_audio.csv"
        )

    escrever()
    escrever("-- Participantes e tarefas --")
    escrever(f"  Participantes distintos: {len(participantes)}")
    escrever(
        f"  Participantes com tarefas 001-011 completas: {completos}"
    )
    escrever(
        f"  Participantes com tarefas faltando: {len(faltando)}"
    )
    escrever(
        f"  Pares participante/tarefa duplicados: {len(duplicados)}"
    )
    escrever(
        f"  IDs diferentes entre arquivo e pasta: "
        f"{len(ids_divergentes)}"
    )

    if faltando:
        escrever("  Exemplos de participantes com tarefas faltando:")

        for participante, tarefas in faltando[:10]:
            escrever(
                f"    {participante}: {', '.join(tarefas)}"
            )

    if duplicados:
        escrever("  Exemplos de duplicados:")

        for participante, tarefa, quantidade in duplicados[:10]:
            escrever(
                f"    {participante} tarefa {tarefa}: "
                f"{quantidade} arquivos"
            )

    escrever()
    escrever("-- Consistencia do formato "
             "(formato / canais / taxa / bits) --")

    if formatos:

        for chave, quantidade in formatos.most_common():
            formato, canais, taxa, bits = chave

            escrever(
                f"  {formato:<14} "
                f"canais={canais} "
                f"taxa={taxa} Hz "
                f"bits={bits} : {quantidade} arquivos"
            )

    escrever(
        f"  Arquivos com erro no cabecalho: "
        f"{len(erros_cabecalho)}"
    )

    for nome, erro in erros_cabecalho[:10]:
        escrever(f"    {nome} -> {erro}")

    escrever(
        f"  Arquivos com tamanho do chunk data inconsistente: "
        f"{len(tamanhos_inconsistentes)}"
    )

    for nome in tamanhos_inconsistentes[:10]:
        escrever(f"    {nome}")

    escrever()
    escrever("-- Duracao por tarefa (segundos) --")
    escrever(
        f"  {'tarefa':<8} {'conteudo':<42} "
        f"{'n':>5} {'min':>8} {'mediana':>9} {'max':>8}"
    )

    for tarefa in sorted(duracoes):

        valores = sorted(duracoes[tarefa])
        n = len(valores)

        minimo = valores[0]
        maximo = valores[-1]

        if n % 2:
            mediana = valores[n // 2]
        else:
            mediana = (
                valores[n // 2 - 1] +
                valores[n // 2]
            ) / 2

        escrever(
            f"  {tarefa:<8} "
            f"{TAREFAS.get(tarefa, ''):<42} "
            f"{n:>5} "
            f"{minimo:>8.2f} "
            f"{mediana:>9.2f} "
            f"{maximo:>8.2f}"
        )

    escrever()
    escrever("=" * 72)
    escrever("Inventario concluido.")
    escrever(f"Arquivos gerados em: {PASTA_SAIDA}")
    escrever("=" * 72)

    caminho_resumo = PASTA_SAIDA.joinpath( "resumo_inventario.txt")

    with open(
        caminho_resumo,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("\n".join(linhas))

    # ============================================================
    # EXIBICAO NO TERMINAL
    # ============================================================

    print()
    print("\n".join(linhas))


if __name__ == "__main__":
    main()