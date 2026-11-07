from avfad_paths import asset, project_root, raw_dir
import csv
import re
import zipfile
from pathlib import Path, PurePosixPath
from collections import Counter, defaultdict


# ============================================================
# CONFIGURACAO
# ============================================================

PASTA = project_root()
SAIDA = asset(PASTA, "saida")

CSV_AUDIO = SAIDA / "inventario_audio.csv"

ZIPS = [
    "AVFAD_01_00_00_1_README.zip",
    "AVFAD_01_00_00_2_A_to_C.zip",
    "AVFAD_01_00_00_3_D_to_L.zip",
    "AVFAD_01_00_00_4_M.zip",
    "AVFAD_01_00_00_5_N_to_Z.zip",
]

FORMATO_PADRAO = ("PCM", "1", "48000", "16")


# ============================================================
# FUNCOES AUXILIARES
# ============================================================

def ler_csv(caminho):
    with open(caminho, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def salvar_csv(caminho, registros, campos):
    with open(
        caminho,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        escritor = csv.DictWriter(
            f,
            fieldnames=campos,
            delimiter=";",
            extrasaction="ignore"
        )

        escritor.writeheader()
        escritor.writerows(registros)


def identificar_id_prt(nome):
    """
    Tenta identificar o participante pelo nome do PRT.
    Se nao conseguir, procura uma pasta com ID de 3 letras.
    """

    caminho = PurePosixPath(nome)
    stem = caminho.stem.upper()

    resultado = re.match(
        r"^([A-Z]{3})(?:\d{3})?(?:[_-].*)?$",
        stem
    )

    if resultado:
        return resultado.group(1)

    for parte in reversed(caminho.parts[:-1]):
        parte = parte.upper()

        if re.fullmatch(r"[A-Z]{3}", parte):
            return parte

    return ""


def numero(valor):
    try:
        return float(valor)
    except (ValueError, TypeError):
        return None


# ============================================================
# CARREGAR INVENTARIO WAV
# ============================================================

print("\nLendo inventario de audio...")

if not CSV_AUDIO.exists():
    raise FileNotFoundError(
        f"Arquivo nao encontrado: {CSV_AUDIO}"
    )

audios = ler_csv(CSV_AUDIO)

ids_wav = {
    r["participante"]
    for r in audios
    if r.get("participante")
}

# ============================================================
# ANALISAR FORMATOS E DURACOES
# ============================================================

formatos_diferentes = []
duracoes_revisar = []
duracoes_curta = []
duracoes_longas = []

for r in audios:

    formato = (
        r.get("formato", ""),
        str(r.get("canais", "")),
        str(r.get("taxa_hz", "")),
        str(r.get("bits", ""))
    )

    if formato != FORMATO_PADRAO:

        item = dict(r)
        item["motivo"] = (
            "Diferente do padrao PCM mono 48000 Hz 16 bits"
        )

        formatos_diferentes.append(item)

    duracao = numero(r.get("duracao_s"))
    tarefa = r.get("tarefa", "")

    if duracao is None:
        continue

    motivos = []

    if duracao < 1.0:
        motivos.append("Menos de 1 segundo")
        duracoes_curta.append(r)

    if tarefa in {
        "001", "002", "003", "004", "005",
        "006", "007", "008", "009"
    } and duracao > 60:
        motivos.append("Mais de 60 segundos nas tarefas 001-009")

    if duracao > 120:
        motivos.append("Mais de 120 segundos")

    data_size = numero(r.get("data_size"))

    if data_size is not None and data_size <= 0:
        motivos.append("Chunk de audio vazio")

    if r.get("erro"):
        motivos.append("Erro no cabecalho")

    if motivos:
        item = dict(r)
        item["motivo_revisao"] = " | ".join(motivos)

        duracoes_revisar.append(item)

        if duracao > 120:
            duracoes_longas.append(item)


# ============================================================
# ANALISAR ARQUIVOS PRT
# ============================================================

print("Procurando arquivos PRT nos ZIPs...")

prt_registros = []
prt_por_id = defaultdict(list)
prt_por_pacote = Counter()
nomes_prt = Counter()

for nome_zip in ZIPS:

    caminho_zip = asset(PASTA, nome_zip)

    if not caminho_zip.exists():
        print(f"AVISO: ZIP nao encontrado: {nome_zip}")
        continue

    with zipfile.ZipFile(caminho_zip, "r") as z:

        for info in z.infolist():

            if info.is_dir():
                continue

            if not info.filename.lower().endswith(".prt"):
                continue

            participante = identificar_id_prt(info.filename)

            registro = {
                "pacote": nome_zip,
                "arquivo": info.filename,
                "nome_base": PurePosixPath(
                    info.filename
                ).name,
                "participante": participante,
                "tamanho_bytes": info.file_size,
            }

            prt_registros.append(registro)
            prt_por_pacote[nome_zip] += 1

            nomes_prt[
                PurePosixPath(info.filename).name.lower()
            ] += 1

            if participante:
                prt_por_id[participante].append(
                    info.filename
                )


# ============================================================
# INVESTIGAR PRTs
# ============================================================

ids_prt = set(prt_por_id.keys())

ids_sem_prt = sorted(ids_wav - ids_prt)
ids_sem_wav = sorted(ids_prt - ids_wav)

prt_multiplos = {
    participante: arquivos
    for participante, arquivos in prt_por_id.items()
    if len(arquivos) > 1
}

prt_sem_id = [
    r for r in prt_registros
    if not r["participante"]
]

nomes_duplicados = {
    nome: quantidade
    for nome, quantidade in nomes_prt.items()
    if quantidade > 1
}


# ============================================================
# GERAR CSVs
# ============================================================

print("Gerando relatorios...")

if formatos_diferentes:
    campos = list(formatos_diferentes[0].keys())

    salvar_csv(
        SAIDA / "wav_formatos_diferentes.csv",
        formatos_diferentes,
        campos
    )

if duracoes_revisar:
    campos = list(duracoes_revisar[0].keys())

    salvar_csv(
        SAIDA / "wav_duracoes_para_revisar.csv",
        duracoes_revisar,
        campos
    )

campos_prt = [
    "pacote",
    "arquivo",
    "nome_base",
    "participante",
    "tamanho_bytes"
]

salvar_csv(
    SAIDA / "inventario_prt_detalhado.csv",
    prt_registros,
    campos_prt
)


# ============================================================
# RELATORIO TXT
# ============================================================

linhas = []


def escrever(texto=""):
    linhas.append(str(texto))


escrever("=" * 68)
escrever("DIAGNOSTICO DE EXCECOES -- AVFAD")
escrever("=" * 68)

escrever()
escrever("-- FORMATOS DIFERENTES --")

escrever(
    f"Total: {len(formatos_diferentes)} arquivos"
)

for r in formatos_diferentes:
    escrever(
        f"  {r['participante']} tarefa {r['tarefa']} | "
        f"{r['formato']} | canais={r['canais']} | "
        f"taxa={r['taxa_hz']} | bits={r['bits']} | "
        f"{r['arquivo']}"
    )

escrever()
escrever("-- GRAVACOES COM MENOS DE 1 SEGUNDO --")

escrever(f"Total: {len(duracoes_curta)}")

for r in duracoes_curta:
    escrever(
        f"  ID={r['participante']} | "
        f"tarefa={r['tarefa']} | "
        f"duracao={r['duracao_s']} s | "
        f"data_size={r['data_size']} bytes | "
        f"{r['arquivo']}"
    )

escrever()
escrever("-- GRAVACOES LONGAS --")

escrever(f"Total acima de 120 segundos: {len(duracoes_longas)}")

for r in sorted(
    duracoes_longas,
    key=lambda x: numero(x.get("duracao_s")) or 0,
    reverse=True
):
    escrever(
        f"  ID={r['participante']} | "
        f"tarefa={r['tarefa']} | "
        f"duracao={r['duracao_s']} s | "
        f"{r['arquivo']}"
    )

escrever()
escrever("-- TOTAL DE GRAVACOES PARA REVISAR --")
escrever(f"  {len(duracoes_revisar)}")

escrever()
escrever("-- ARQUIVOS PRT --")

escrever(f"  Total de PRT: {len(prt_registros)}")
escrever(f"  IDs identificados: {len(ids_prt)}")
escrever(f"  IDs sem PRT correspondente: {len(ids_sem_prt)}")
escrever(f"  IDs de PRT sem WAV correspondente: {len(ids_sem_wav)}")
escrever(f"  PRTs sem ID identificado: {len(prt_sem_id)}")
escrever(
    f"  Participantes com mais de um PRT: {len(prt_multiplos)}"
)
escrever(
    f"  Nomes de arquivo PRT repetidos: {len(nomes_duplicados)}"
)

escrever()
escrever("-- PRT POR PACOTE --")

for pacote, quantidade in sorted(prt_por_pacote.items()):
    escrever(f"  {pacote}: {quantidade}")

if prt_multiplos:

    escrever()
    escrever("-- PARTICIPANTES COM MULTIPLOS PRTs --")

    for participante, arquivos in sorted(prt_multiplos.items()):

        escrever(f"  {participante}:")

        for arquivo in arquivos:
            escrever(f"    {arquivo}")

if ids_sem_wav:

    escrever()
    escrever("-- IDs DE PRT SEM PARTICIPANTE WAV --")

    for participante in ids_sem_wav:
        escrever(f"  {participante}")

if prt_sem_id:

    escrever()
    escrever("-- PRTs SEM ID IDENTIFICADO --")

    for r in prt_sem_id:
        escrever(f"  {r['arquivo']}")

if nomes_duplicados:

    escrever()
    escrever("-- NOMES PRT REPETIDOS --")

    for nome, quantidade in sorted(nomes_duplicados.items()):
        escrever(f"  {nome}: {quantidade} vezes")

escrever()
escrever("=" * 68)
escrever("DIAGNOSTICO CONCLUIDO")
escrever("=" * 68)

caminho_relatorio = SAIDA / "diagnostico_avfad.txt"

with open(caminho_relatorio, "w", encoding="utf-8") as f:
    f.write("\n".join(linhas))

print()
print("\n".join(linhas))

print()
print("Arquivos gerados em:")
print(SAIDA)


if __name__ == "__main__":
    pass