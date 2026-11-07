
from avfad_paths import asset, project_root, raw_dir
from pathlib import PurePosixPath
from zipfile import ZipFile
from pathlib import Path

root = project_root()

for zip_path in sorted(raw_dir(root).glob("AVFAD_01_00_00_*.zip")):
    participantes = set()
    exemplos = []

    with ZipFile(zip_path) as z:
        for info in z.infolist():
            if not info.filename.lower().endswith(".wav"):
                continue

            partes = PurePosixPath(info.filename).parts

            if len(partes) >= 2:
                pasta = partes[-2]

                if (
                    len(pasta) == 3
                    and pasta.isalpha()
                    and pasta.isupper()
                ):
                    participantes.add(pasta)

                    if len(exemplos) < 8:
                        exemplos.append(info.filename)

    print("\n" + "=" * 60)
    print(zip_path.name)
    print("Participantes encontrados:", len(participantes))
    print("Primeiros IDs:", sorted(participantes)[:20])
    print("Exemplos de arquivos:")
    for exemplo in exemplos:
        print(" ", exemplo)
