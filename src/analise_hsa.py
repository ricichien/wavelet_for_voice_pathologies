
from avfad_paths import asset, project_root, raw_dir
import csv
import io
import zipfile

from pathlib import Path
from collections import Counter, defaultdict
from statistics import median
from openpyxl import load_workbook

ROOT = project_root()

CSV_PATH = asset(ROOT, "validacao_tarefa_002.csv")

# Carrega a data de visita diretamente da planilha
metadata = {}

for zip_path in raw_dir(ROOT).glob("AVFAD_01_00_00_*.zip"):
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.filename.lower().endswith("avfad_01_00_00.xlsx"):
                wb = load_workbook(
                    io.BytesIO(z.read(info)),
                    read_only=True,
                    data_only=True
                )

                ws = wb["AVFAD"]
                rows = ws.iter_rows(values_only=True)

                headers = next(rows)
                idx = {name: headers.index(name) for name in (
                    "File ID", "Visit date"
                )}

                for row in rows:
                    pid = str(row[idx["File ID"]]).strip().upper()
                    date = row[idx["Visit date"]]

                    if pid:
                        metadata[pid] = date.year if date else "Sem data"

                wb.close()
                break

# Carrega o CSV validado
with open(CSV_PATH, encoding="utf-8-sig") as f:
    records = list(csv.DictReader(f))

hsa = [
    r for r in records
    if r["visit_place"] == "HSA"
    and r["label"] in ("Normal", "Pathological")
]

print("\n=== HSA: DISTRIBUIÇÃO GERAL ===")
print("Total:", len(hsa))

print(Counter(r["label"] for r in hsa))

print("\n=== SEXO POR CLASSE ===")

table = defaultdict(Counter)

for r in hsa:
    table[r["sex"]][r["label"]] += 1

for sex, counts in sorted(table.items()):
    print(sex, dict(counts))

print("\n=== IDADE: FAIXAS DE 5 ANOS E SEXO ===")

table = defaultdict(Counter)

for r in hsa:
    age = int(float(r["age"]))
    start = 18 + ((age - 18) // 5) * 5
    age_group = f"{start}-{start + 4}"

    key = (r["sex"], age_group)
    table[key][r["label"]] += 1

possible_pairs = 0

for (sex, age_group), counts in sorted(table.items()):
    normal = counts["Normal"]
    path = counts["Pathological"]

    if normal or path:
        print(
            f"{sex} | {age_group}: "
            f"Normal={normal}, Patológico={path}"
        )

    possible_pairs += min(normal, path)

print(
    "\nMáximo de pares possíveis por sexo "
    "e faixa de 5 anos:",
    possible_pairs
)

print("\n=== ANO DE GRAVAÇÃO ===")

table = defaultdict(Counter)

for r in hsa:
    year = metadata.get(r["participant_id"], "Sem data")
    table[year][r["label"]] += 1

for year, counts in sorted(table.items()):
    print(year, dict(counts))

print("\n=== DURAÇÃO DOS ÁUDIOS ===")

for label in ("Normal", "Pathological"):
    durations = sorted(
        float(r["duration_s"])
        for r in hsa
        if r["label"] == label
    )

    print(
        label,
        "| N =", len(durations),
        "| Mínimo =", round(min(durations), 2),
        "| Mediana =", round(median(durations), 2),
        "| Máximo =", round(max(durations), 2)
    )

print("\nAnálise concluída.")
