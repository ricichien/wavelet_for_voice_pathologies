
from avfad_paths import asset, project_root, raw_dir
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
from openpyxl import load_workbook
from collections import Counter
import io
import wave

ROOT = project_root()
PLANILHA = (
    asset(ROOT, "metadata") / "AVFAD_01_00_00_1_README"
    / "AVFAD_01_00_00.xlsx"
)

# Lê os IDs dos participantes na planilha
wb = load_workbook(PLANILHA, read_only=True, data_only=True)
ws = wb["AVFAD"]

cabecalho = [str(c.value).strip() if c.value else "" for c in ws[1]]
idx_id = cabecalho.index("File ID")

ids = {
    str(row[idx_id]).strip().upper()
    for row in ws.iter_rows(min_row=2, values_only=True)
    if row[idx_id] is not None
}
wb.close()

# Registra as gravações encontradas em cada ZIP
gravacoes = {pid: set() for pid in ids}
arquivos_por_zip = Counter()
amostra = None

zips = sorted(raw_dir(ROOT).glob("AVFAD_01_00_00_*.zip"))

for caminho_zip in zips:
    with ZipFile(caminho_zip, "r") as arquivo:
        for info in arquivo.infolist():
            if not info.filename.lower().endswith(".wav"):
                continue

            arquivos_por_zip[caminho_zip.name] += 1

            caminho = PurePosixPath(info.filename)
            pasta = caminho.parent.name.strip().upper()

            if pasta not in ids:
                continue

            nome = caminho.stem.upper()
            sufixo = nome[len(pasta):] if nome.startswith(pasta) else ""

            if sufixo.isdigit():
                numero = int(sufixo)
                gravacoes[pasta].add(numero)

                if numero == 2 and amostra is None:
                    amostra = (caminho_zip, info.filename)

print("\n======= ARQUIVOS WAV POR PACOTE =======")
for nome, total in arquivos_por_zip.items():
    print(f"{nome}: {total} WAVs")

encontrados = [p for p in ids if gravacoes[p]]
com_a = [p for p in ids if 2 in gravacoes[p]]
completos = [
    p for p in ids
    if set(range(1, 12)).issubset(gravacoes[p])
]

faltantes_a = sorted(p for p in ids if 2 not in gravacoes[p])

print("\n======= COBERTURA DOS PARTICIPANTES =======")
print("Participantes na planilha:", len(ids))
print("Com alguma gravação:", len(encontrados))
print("Com gravação 002.wav:", len(com_a))
print("Com todas as gravações 001 a 011:", len(completos))
print("Sem gravação 002.wav:", len(faltantes_a))

if faltantes_a:
    print("Primeiros IDs sem 002.wav:", faltantes_a[:30])

# Lê somente o cabeçalho de um áudio de dentro do ZIP
if amostra:
    caminho_zip, membro = amostra

    with ZipFile(caminho_zip, "r") as arquivo:
        dados = arquivo.read(membro)

    try:
        with wave.open(io.BytesIO(dados), "rb") as wav:
            print("\n======= FORMATO DE UMA AMOSTRA =======")
            print("Arquivo:", membro)
            print("Canais:", wav.getnchannels())
            print("Taxa de amostragem:", wav.getframerate(), "Hz")
            print("Bits por amostra:", wav.getsampwidth() * 8)
            print("Duração:", round(
                wav.getnframes() / wav.getframerate(), 2
            ), "segundos")
    except wave.Error as erro:
        print("\nNão foi possível ler o cabeçalho WAV:", erro)
else:
    print("\nNenhum arquivo 002.wav foi localizado nos ZIPs.")
