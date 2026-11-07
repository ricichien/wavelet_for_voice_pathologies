"""Baixa somente o checkpoint publico fixado; nao recebe audio ou rotulos."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL = "facebook/wav2vec2-xls-r-300m"
REVISION = "1a640f32ac3e39899438a2931f9924c02f080a54"
FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md")
DEFAULT = ROOT / "models" / "wav2vec2-xls-r-300m" / REVISION


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AVFAD-research-local/1.0"}), timeout=60)


def digests(path):
    sha256 = hashlib.sha256()
    git = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            git.update(block)
    return sha256.hexdigest(), git.hexdigest()


def inspect_versions():
    for package, version in (("torch", "2.10.0"), ("transformers", "4.57.6")):
        with request(f"https://pypi.org/pypi/{package}/{version}/json") as response:
            info = json.load(response)["info"]
        print(package, info["version"], info["requires_python"], flush=True)
    with request("https://download.pytorch.org/whl/cu128/torch/") as response:
        lines = response.read().decode().splitlines()
    matches = [line for line in lines if "torch-2.10.0" in line and "cp310-cp310-win_amd64" in line]
    print("torch2.10_CUDA128_Python310_Windows_available", bool(matches), flush=True)
    if not matches:
        raise RuntimeError("Pacote CUDA esperado ausente; nao instalar outro automaticamente")


def download(out):
    if not out.is_relative_to(ROOT / "models"):
        raise ValueError("Destino precisa permanecer dentro de models no projeto")
    with request(f"https://huggingface.co/api/models/{MODEL}/tree/{REVISION}") as response:
        tree = {row["path"]: row for row in json.load(response)}
    if not set(FILES).issubset(tree):
        raise RuntimeError("Revisao nao contem todos os arquivos esperados")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"model_id": MODEL, "model_revision": REVISION, "files": {}}
    for name in FILES:
        entry, target = tree[name], out / name
        if not target.exists():
            temporary = target.with_name(name + ".part")
            # Uma transferencia parcial anterior nunca e tratada como completa.
            received, last = 0, -1
            with request(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}") as response, temporary.open("wb") as stream:
                while block := response.read(4 * 1024 * 1024):
                    stream.write(block)
                    received += len(block)
                    percent = int(received * 10 / entry["size"])
                    if percent > last:
                        last = percent
                        print(f"{name}: {received}/{entry['size']} bytes", flush=True)
            if temporary.stat().st_size != entry["size"]:
                raise RuntimeError(f"Download incompleto: {name}")
            digest, git_digest = digests(temporary)
            if entry.get("lfs"):
                if digest != entry["lfs"]["oid"]:
                    raise RuntimeError(f"SHA256 publicado diverge: {name}")
            elif git_digest != entry["oid"]:
                raise RuntimeError(f"ObjetoGit publicado diverge: {name}")
            temporary.replace(target)
        digest, git_digest = digests(target)
        valid = digest == entry["lfs"]["oid"] if entry.get("lfs") else git_digest == entry["oid"]
        if target.stat().st_size != entry["size"] or not valid:
            raise RuntimeError(f"Arquivo existente difere da origem: {name}")
        manifest["files"][name] = {"size_bytes": target.stat().st_size, "sha256": digest,
                                    "repository_oid": entry["oid"], "lfs_sha256": entry.get("lfs", {}).get("oid")}
        print(f"VERIFICADO {name}: SHA256={digest}", flush=True)
    provenance = out / "ORIGEM_MODELO.json"
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if provenance.exists() and provenance.read_text(encoding="utf8") != text:
        raise RuntimeError("Manifesto local divergente; preservar e revisar")
    if not provenance.exists():
        provenance.write_bytes(text.encode("utf8"))
    print("Modelo local completo e verificado:", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verificar-versoes", action="store_true")
    parser.add_argument("--baixar-modelo", action="store_true")
    parser.add_argument("--saida", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.verificar_versoes:
        inspect_versions()
    if args.baixar_modelo:
        download(args.saida.resolve())
    if not (args.verificar_versoes or args.baixar_modelo):
        parser.print_help()


if __name__ == "__main__":
    main()
