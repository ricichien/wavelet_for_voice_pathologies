
from avfad_paths import asset, project_root, raw_dir
import zipfile
import io
from pathlib import Path
from openpyxl import load_workbook

ROOT = project_root()

with open(asset(ROOT, "resumo_curto.txt"), "w", encoding="utf-8") as out:

    for zip_path in raw_dir(ROOT).glob("AVFAD_01_00_00_*.zip"):
        with zipfile.ZipFile(zip_path) as z:

            for info in z.infolist():
                if info.is_dir():
                    continue

                nome = info.filename.lower()

                if nome.endswith(".xlsx"):
                    out.write(f"\nPLANILHA: {info.filename}\n")

                    wb = load_workbook(
                        io.BytesIO(z.read(info)),
                        read_only=True,
                        data_only=True
                    )

                    for ws in wb.worksheets:
                        out.write(f"\nABA: {ws.title}\n")
                        out.write(f"Dimensão: {ws.max_row} x {ws.max_column}\n")

                        for row in ws.iter_rows(
                            min_row=1,
                            max_row=min(ws.max_row, 5),
                            values_only=True
                        ):
                            out.write(str(row) + "\n")

                    wb.close()

                elif nome.endswith(".prt"):
                    if Path(nome).name in (
                        "imn002.prt",
                        "imn010.prt"
                    ):
                        out.write(f"\nPRT: {info.filename}\n")
                        out.write(
                            z.read(info).decode(
                                "utf-8",
                                errors="replace"
                            )[:1500]
                        )
                        out.write("\n")

                elif "readme" in nome and nome.endswith(".txt"):
                    out.write(f"\nREADME: {info.filename}\n")
                    out.write(
                        z.read(info).decode(
                            "utf-8",
                            errors="replace"
                        )[:4000]
                    )
                    out.write("\n")

print("Relatorio curto criado em resumo_curto.txt na raiz do repositorio.")
