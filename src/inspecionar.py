from avfad_paths import asset, project_root, raw_dir
from openpyxl import load_workbook

arquivo = asset(project_root(), "metadata") / "AVFAD_01_00_00_1_README" / "AVFAD_01_00_00.xlsx"

wb = load_workbook(arquivo, read_only=True, data_only=True)

for ws in wb.worksheets:
    print("\n" + "=" * 70)
    print("PLANILHA:", ws.title)
    print("DIMENSÕES:", ws.max_row, "linhas x", ws.max_column, "colunas")

    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i >= 6:
            break
        print(row)

wb.close()