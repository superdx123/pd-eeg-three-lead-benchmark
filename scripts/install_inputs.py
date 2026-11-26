"""Install the versioned benchmark inputs after verifying each file."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile

from pd_eeg.io import read_json, safe_relative_file, sha256


def install(archive, manifest, destination):
    archive, destination = Path(archive), Path(destination)
    if archive.stat().st_size != manifest["bytes"] or sha256(archive) != manifest["sha256"]:
        raise ValueError("Input archive size or SHA-256 differs from the release manifest")
    members = manifest["members"]
    outputs = {name: safe_relative_file(destination, name) for name in members}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("An input already exists; select a fresh working directory")
    with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory() as temporary:
        records = source.infolist()
        if len(records) != len(members) or {entry.filename for entry in records} != set(members):
            raise ValueError("Archive contains missing, duplicate or unexpected entries")
        for entry in records:
            name = entry.filename
            expected = members[name]
            if entry.is_dir() or entry.file_size != expected["bytes"]:
                raise ValueError(f"Invalid archive member: {name}")
            staged = safe_relative_file(temporary, name)
            staged.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with source.open(entry) as stream, staged.open("xb") as out:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    out.write(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {name}")
        for name, output in outputs.items():
            output.parent.mkdir(parents=True, exist_ok=True)
            with safe_relative_file(temporary, name).open("rb") as stream, output.open("xb") as out:
                shutil.copyfileobj(stream, out)
    return len(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("configs/inputs.json"))
    parser.add_argument("--destination", type=Path, default=Path("."))
    args = parser.parse_args()
    print(f"Installed {install(args.archive, read_json(args.manifest), args.destination)} verified files.")


if __name__ == "__main__":
    main()
