from avfad_paths import asset, project_root, raw_dir
from pathlib import Path
from zipfile import ZipFile, BadZipFile

root = project_root()

for arquivo in sorted(raw_dir(root).glob("AVFAD_01_00_00_*.zip")):
    print("\n" + "=" * 60)
    print(arquivo.name)
    print("Tamanho:", round(arquivo.stat().st_size / (1024**3), 2), "GB")

    try:
        with ZipFile(arquivo) as z:
            problema = z.testzip()

            if problema:
                print("ERRO: arquivo corrompido:", problema)
            else:
                print("OK: ZIP íntegro")

    except BadZipFile:
        print("ERRO: não é um ZIP válido")
    except Exception as e:
        print("ERRO:", e)