#!/usr/bin/env python3
"""Install hash-pinned KeiReality 384×192 runtime files from GitHub Releases."""
import argparse
from contextlib import ExitStack
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
from shutil import copyfileobj
import sys
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
# The research scorers pin this specific baseline, including the manifest's file hash.
MANIFEST = ROOT / "configs/reality/releases/keireality-384-v1.0.0.json"
CURRENT_TAG = "keireality-384-v1.1.0"


def verify(path, item):
    digest = hashlib.sha256()
    if path.is_file() and path.stat().st_size == item["bytes"]:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() == item["sha256"]:
            return
    raise ValueError(f'{item["asset"]} のサイズまたは SHA-256 が配布版と異なります ({path})')


def install(root, manifest, assets_dir=None):
    base = f'https://github.com/{manifest["repository"]}/releases/download/{manifest["tag"]}'
    pending = []
    with ExitStack() as cleanup:
        for item in manifest["files"]:
            dest = root / item["path"]
            if dest.exists() or dest.is_symlink():
                verify(dest, item)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f'取得中  {item["asset"]}', flush=True)
            with NamedTemporaryFile(dir=dest.parent, prefix=".keireality-", delete=False) as output:
                temp = Path(output.name)
                cleanup.callback(temp.unlink, missing_ok=True)
                if assets_dir is None:
                    request = Request(f'{base}/{item["asset"]}', headers={"User-Agent": "KeiReality-release"})
                    source = urlopen(request, timeout=90)
                else:
                    source = (assets_dir / item["asset"]).open("rb")
                with source:
                    copyfileobj(source, output, length=1024 * 1024)
            verify(temp, item)
            pending.append((temp, dest, item))

        # Validate all downloads before publishing them. A concurrent install can win this race.
        for temp, dest, item in pending:
            try:
                os.link(temp, dest)
            except FileExistsError:
                verify(dest, item)
    print(f'{manifest["tag"]}  {len(manifest["files"])} ファイルの SHA-256 を確認しました。')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", choices=(MANIFEST.stem, CURRENT_TAG), default=CURRENT_TAG,
                        help="release to install (default: %(default)s)")
    parser.add_argument("--from-dir", type=Path, help="install previously downloaded release assets from this directory")
    args = parser.parse_args()
    try:
        manifest = MANIFEST.with_name(args.tag + ".json")
        install(ROOT, json.loads(manifest.read_text(encoding="utf-8")), args.from_dir)
    except (OSError, HTTPException, ValueError) as error:
        sys.exit(str(error))
