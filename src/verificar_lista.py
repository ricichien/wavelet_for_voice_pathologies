from avfad_paths import asset, project_root, raw_dir
from pathlib import Path
from collections import Counter

arquivo = Path(
    asset(project_root(), "metadata") / "AVFAD_01_00_00_1_README" / "ListFiles.txt"
)

print("Arquivo encontrado:", arquivo.exists())

if arquivo.exists():
    linhas = arquivo.read_text(
        encoding="utf-8",
        errors="replace"
    ).splitlines()

    print("Total de linhas:", len(linhas))

    extensoes = Counter(
        Path(l.strip()).suffix.lower()
        for l in linhas if l.strip()
    )

    print("\nExtensões:")
    for ext, total in extensoes.most_common():
        print(ext or "(sem extensão)", total)

    print("\nPrimeiras 25 linhas:")
    for linha in linhas[:25]:
        print(linha)

    print("\nÚltimas 15 linhas:")
    for linha in linhas[-15:]:
        print(linha)