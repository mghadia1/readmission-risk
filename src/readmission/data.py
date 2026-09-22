"""Download and verify the UCI "Diabetes 130-US Hospitals for Years 1999-2008" files."""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

# SHA-256 of the extracted files, recorded 2026-09-22. Hashing the CSVs rather
# than the zip means a re-zipped but identical release still verifies.
FILES = {
    "diabetic_data.csv": "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
    "IDS_mapping.csv": "f1bb82b471cb34649352597572c9b1fb00bd27f77b9f5a22a03dc3eb1039749e",
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path.name}: {actual}")


def download(destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    if not all((destination / name).exists() for name in FILES):
        archive = destination / "diabetes.zip"
        urllib.request.urlretrieve(DATASET_URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            for name in FILES:
                zipped.extract(name, destination)
        archive.unlink()
    for name, expected in FILES.items():
        verify(destination / name, expected)
    return [destination / name for name in FILES]
