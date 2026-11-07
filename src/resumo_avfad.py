from avfad_paths import asset, project_root, raw_dir
from openpyxl import load_workbook
from collections import Counter
from pathlib import Path

arquivo = Path(
    asset(project_root(), "metadata") / "AVFAD_01_00_00_1_README" / "AVFAD_01_00_00.xlsx"
)

wb = load_workbook(arquivo, read_only=True, data_only=True)
ws = wb["AVFAD"]

cabecalho = [c.value for c in ws[1]]
col = {str(nome).strip(): i for i, nome in enumerate(cabecalho) if nome}

participantes = list(ws.iter_rows(min_row=2, values_only=True))
participantes = [r for r in participantes if r[col["File ID"]] is not None]

diagnosticos = Counter()
sexos = Counter()
grupos = Counter()
idades = []

for r in participantes:
    codigo = r[col["CMVD-I Dimension 1 (numeric system)"]]
    nome = r[col["CMVD-I Dimension 1 (word system)"]]
    sexo = str(r[col["Sex"]]).strip()

    diagnosticos[str(nome).strip()] += 1
    sexos[sexo] += 1

    grupo = "Normal" if codigo == 0 else "Patológico"
    grupos[(sexo, grupo)] += 1

    idade = r[col["Age"]]
    if isinstance(idade, (int, float)):
        idades.append(idade)

print("\nTOTAL DE PARTICIPANTES:", len(participantes))

print("\nCLASSIFICAÇÃO BINÁRIA:")
print("Normais:", grupos[("M", "Normal")] + grupos[("F", "Normal")])
print("Patológicos:", grupos[("M", "Patológico")] + grupos[("F", "Patológico")])

print("\nDISTRIBUIÇÃO POR SEXO:")
for sexo, total in sorted(sexos.items()):
    print(sexo, total)

print("\nSEXO X DIAGNÓSTICO:")
for chave, total in sorted(grupos.items()):
    print(chave, total)

print("\nIDADE:")
print("Mínima:", min(idades))
print("Máxima:", max(idades))

print("\nDIAGNÓSTICOS:")
for nome, total in diagnosticos.most_common():
    print(f"{nome}: {total}")

wb.close()