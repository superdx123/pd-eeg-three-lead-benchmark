"""Prepare/check a fixed source subset; never import or execute archived code."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
ROOT_SNAPSHOT_FILES = (
    "README.md", "ENVIRONMENT.md", "LICENSE_STATUS.md", "INFERENCE_ADAPTER.md",
    "ADAPTER_V3_CHANGELOG.md", "primary_metrics.json",
    "predict_saved_weights.py", "verify_scores.py",
)
FROZEN_MODULES = (
    "dsdata/__init__.py", "dsdata/data_io.py", "dsdata/metrics.py", "dsdata/protocol.py",
    "dsmodel/__init__.py", "dsmodel/models.py", "dsmodel/ssl.py", "dsmodel/transforms.py",
    "dstraining/__init__.py", "dstraining/engine.py", "dstraining/trainer.py",
    "run/__init__.py", "run/analyze_formal.py", "run/common.py", "run/formal.py",
)
RUNNER_FILES = ("common.py", "preflight.py", "run_fold.py", "summarize.py", "validate_fold.py")
CONFIG_FILES = ("protocol.json", "tasks.json")
MANAGED_FILES = {"repo_files_manifest.json", "local_ready_receipt.json"}
SUPPORT_FILES = {
    "README.md", ".gitignore", "archive.lock.json", "PUBLICATION_STATUS.json",
    "tools/prepare_source_snapshot.py",
}
BLOCKED_PARTS = {"weights", "data", "metadata", "provenance", "internal", "private", "ops", "replay_outputs"}
BLOCKED_EXTENSIONS = {".csv", ".tsv", ".npz", ".npy", ".pt", ".pth", ".ckpt", ".zip", ".7z", ".tar", ".gz", ".edf", ".bdf", ".set", ".fdt", ".log"}
SECRET_PATTERNS = (
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def snapshot_paths() -> list[str]:
    paths = list(ROOT_SNAPSHOT_FILES)
    for protocol in ("S", "W"):
        prefix = f"experiments/{protocol}/"
        paths.extend(prefix + "config/" + name for name in CONFIG_FILES)
        paths.extend(prefix + "scripts/" + name for name in RUNNER_FILES)
        paths.extend(prefix + "frozen_ds/" + name for name in FROZEN_MODULES)
    require(len(paths) == len(set(paths)) == 52, "Unexpected source whitelist")
    return sorted(paths)


def inventory() -> dict[str, Path]:
    items = {}
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts:
            continue
        require(not path.is_symlink(), "Repository symlinks are not allowed")
        if path.is_file():
            items[relative.as_posix()] = path
    return items


def verify_archive(path: Path | None) -> bool:
    if path is None:
        return False
    lock = json.loads((ROOT / "archive.lock.json").read_text(encoding="utf-8"))
    require(path.is_file(), "Archive must be a regular file")
    require(path.name == lock["archive_name"], "Archive filename differs from lock")
    require(path.stat().st_size == lock["bytes"], "Archive byte size differs from lock")
    require(digest(path) == lock["sha256"], "Archive SHA256 differs from lock")
    return True


def static_checks(payload: Path | None) -> dict:
    sources = snapshot_paths()
    expected = SUPPORT_FILES | MANAGED_FILES | {"source_snapshot/" + name for name in sources}
    items = inventory()
    require(set(items) <= expected, "Unexpected file(s) outside the explicit repository whitelist")
    require(expected - MANAGED_FILES <= set(items), "Required repository file(s) missing")
    parsed_count = 0
    for relative, path in items.items():
        parts = Path(relative).parts
        require(not BLOCKED_PARTS.intersection(parts), "Archive-only directory found in repository")
        require(path.suffix.lower() not in BLOCKED_EXTENSIONS, "Archive-only file type found")
        require(path.name != "input_origins.json", "Machine-origin manifest is not allowed")
        content = path.read_bytes()
        require(not any(pattern.search(content) for pattern in SECRET_PATTERNS), "Credential-pattern check failed; values suppressed")
        if path.suffix == ".py":
            ast.parse(content, filename=relative)
            parsed_count += 1
        elif path.suffix == ".json":
            json.loads(content)
    source_comparison = False
    if payload is not None:
        payload = payload.resolve(strict=True)
        for relative in sources:
            original = (payload / relative).resolve(strict=True)
            require(original.is_relative_to(payload), "Source escaped the selected payload")
            require(digest(original) == digest(ROOT / "source_snapshot" / relative), "Snapshot source-byte comparison failed")
        source_comparison = True
    return {
        "source_snapshot_files": len(sources),
        "python_files_statically_parsed": parsed_count,
        "explicit_file_whitelist_passed": True,
        "archive_only_file_exclusion_passed": True,
        "credential_pattern_findings": 0,
        "source_snapshot_byte_comparison_performed": source_comparison,
    }


def emit_json(path: Path, value: dict) -> None:
    require(not path.exists(), "Refusing to overwrite an existing generated artifact")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def build(payload: Path, archive: Path) -> None:
    require(not (ROOT / "source_snapshot").exists(), "Refusing to overwrite a snapshot")
    require(not any((ROOT / name).exists() for name in MANAGED_FILES), "Generated artifacts already exist")
    require(verify_archive(archive), "The locked archive is required for preparation")
    payload = payload.resolve(strict=True)
    for relative in snapshot_paths():
        original = (payload / relative).resolve(strict=True)
        require(original.is_file() and original.is_relative_to(payload), "Invalid source whitelist member")
    for relative in snapshot_paths():
        destination = ROOT / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(payload / relative, destination)
    checks = static_checks(payload)
    entries = {
        relative: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for relative, path in sorted(inventory().items()) if relative not in MANAGED_FILES
    }
    manifest = {
        "schema_version": 1,
        "scope": "Local static source/configuration/aggregate-result subset, not full payload",
        "source_snapshot_files": 52,
        "hash_exempt_generated_artifacts": sorted(MANAGED_FILES),
        "files": entries,
    }
    emit_json(ROOT / "repo_files_manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "status": "PASS_LOCAL_SOURCE_SUBSET_STATIC_CHECKS_LICENSE_PENDING",
        "PUBLICATION_STATUS": "local only / license pending",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_files_manifest_sha256": digest(ROOT / "repo_files_manifest.json"),
        "manifest_hashed_files": len(entries),
        "repository_files_including_manifest_and_receipt": len(entries) + 2,
        "archive_name": archive.name,
        "locked_archive_sha256_verified": True,
        "locked_archive_byte_size_verified": True,
        "checks": checks,
        "signal_or_model_object_loaded": False,
        "training_called": False,
        "forward_inference_called": False,
        "metrics_recomputed": False,
        "threshold_selection_called": False,
        "frozen_source_or_config_modified": False,
        "full_frozen_input_bindings_claimed_for_subset": False,
        "reduced_weight_forward_replay_claimed": False,
        "git_initialized_or_committed": False,
        "external_operations_performed": False,
        "public_release_authorized": False,
    }
    emit_json(ROOT / "local_ready_receipt.json", receipt)
    verify(payload, archive)


def verify(payload: Path | None, archive: Path | None) -> None:
    checks = static_checks(payload)
    manifest_path = ROOT / "repo_files_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = inventory()
    expected = set(manifest["files"]) | MANAGED_FILES
    require(set(items) == expected, "Repository files differ from the recorded manifest")
    for relative, entry in manifest["files"].items():
        path = items[relative]
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "Repository file digest check failed")
    receipt = json.loads((ROOT / "local_ready_receipt.json").read_text(encoding="utf-8"))
    require(receipt["repo_files_manifest_sha256"] == digest(manifest_path), "Manifest digest differs from local receipt")
    archive_checked = verify_archive(archive)
    print(json.dumps({"status": "PASS_LOCAL_STATIC_REPOSITORY_VERIFICATION", "archive_sha256_verified_this_run": archive_checked, "repository_files": len(items), "checks": checks}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="Maintainer-only: populate a fresh source_snapshot once")
    mode.add_argument("--verify", action="store_true", help="Read-only static repository and optional archive verification")
    parser.add_argument("--payload", type=Path, help="Optional original payload for byte-comparison; required to build")
    parser.add_argument("--archive", type=Path, help="Optional exact ZIP for byte hashing; required to build")
    args = parser.parse_args()
    if args.build:
        require(args.payload is not None and args.archive is not None, "Build requires explicit payload and archive paths")
        build(args.payload, args.archive)
    else:
        verify(args.payload, args.archive)


if __name__ == "__main__":
    main()
