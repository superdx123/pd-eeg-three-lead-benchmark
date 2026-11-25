"""Offline, standard-library-only checks; never import the frozen experiment code.

Run with Python 3.10+:
    python -B tests/test_repository_integrity.py --repository REPOSITORY -v
or set PD_EEG_REPOSITORY and use:
    python -B -m unittest discover -s tests -v

Only synthetic, temporary files are used by archive tests. This suite does not
load signals or weights, execute training/inference, or recompute study metrics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


_CLI_REPOSITORY: str | None = None
MANAGED_FILES = {"repo_files_manifest.json", "local_ready_receipt.json"}
SUPPORT_FILES = {
    ".editorconfig", ".gitignore", "LICENSE", "NOTICE.md", "CONTRIBUTING.md",
    "README.md", "PUBLICATION_STATUS.json", "archive.lock.json",
    "docs/REPRODUCIBILITY.md", "docs/RELEASE_CHECKLIST.md",
    "docs/ENTRYPOINTS_AND_ATTRIBUTION.md", "tools/prepare_source_snapshot.py",
    "tests/test_repository_integrity.py",
}

# Independent, immutable baseline from the 2026-11-09 source-only manifest.
# Rehashing a changed snapshot into a new manifest must not hide source changes.
_ROOT_SOURCE_SHA256 = {
    "ADAPTER_V3_CHANGELOG.md": "9227a0c49ea1df9862ec5e82d6ff0173e1edb6dcefcee694e64499de7f7b8ad6",
    "ENVIRONMENT.md": "5dcbc221eb7ecacc02ed34216e12b946c6160eee351ef5731b23f17e4d7d482c",
    "INFERENCE_ADAPTER.md": "4c76f9d45ca23ca4a1fc141858d271c181b1f3a2c27bfe850951a56c3efa2c02",
    "LICENSE_STATUS.md": "ca917e4c7bf6c732e8d9efa845c494c658970b1104a1885a9e9332a001f54313",
    "README.md": "ffb53dc0d6fbc1ad6aec942a190f94b45cfbf7dfe8eda64ccd4a396c58599ec7",
    "predict_saved_weights.py": "7b0ecfa3fc37b132d2985521d60c830838805d4110c3fa5dad6130bf6edb7717",
    "primary_metrics.json": "050c6acc1e8bcacceac4115a8f0585d2c7bab3522d7341f51de42ffbb237811a",
    "verify_scores.py": "aafafefdd48449ebf3284d6a7f7310af25cbe90474e63a58f97eec36c51d354c",
}
_SHARED_FROZEN_SHA256 = {
    "dsdata/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dsdata/data_io.py": "e66fa17eaf524cd99dbac83972a7ba4677039cf2908efefff83435504bbab030",
    "dsdata/metrics.py": "b097a47f4c0cb66fc14e4fb5e3d78d4e6a56ab49921c24c4574c196078c7f0cf",
    "dsdata/protocol.py": "688dce781de86f9845df6040376d5826a87c1b0ed552b966e4aec09118fcc560",
    "dsmodel/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dsmodel/models.py": "b370476ca5c211bf9e5fd380a7c0ad25ec42a3b989cb2d56c8e3592f8654ffe5",
    "dsmodel/ssl.py": "17b2fee125d2104764f9ac902602749ed779fced8f6b86391342cfb2f98553af",
    "dsmodel/transforms.py": "3a133ad6aad417295e04db4c0356d2a2b6dcd68b57451c5c894dccc59915f199",
    "dstraining/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "dstraining/engine.py": "46985303654a0fa15c5e5d4836c82662c72b66b7f334881540e12c600b39eb20",
    "dstraining/trainer.py": "dc162a1c48163203c1b9521942d4bf25a8f127c1e9f1b950925f5aa26cecb9e8",
    "run/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "run/analyze_formal.py": "73d730aaa2a4c84c108d87ff8a37547480a9833b0a721c55b22da53a1ffb7ef9",
    "run/common.py": "70dc897bb2c2abf87e9c02d8a1cf172439aee3dd6907352dcdbd128312f5a98c",
    "run/formal.py": "5b96ffa1e6b84de58ad2ffceed0b80f8570f5af18ea8b6a6f70ff1e6a56bdfec",
}
_PROTOCOL_SHA256 = {
    "S": {
        "config/protocol.json": "156a5c6f236d8a8bff6a8506e4ef6ee815e2b582e4e8d558e419f174aa8d501b",
        "config/tasks.json": "2f1575d4d2ec4d6f72e1170b7206f45b42615f708229d77bb960abcc92714074",
        "scripts/common.py": "052f7fe0742de574b4623f01f51a2df6bb68a03aa868400ef4131cc6e26db3e9",
        "scripts/preflight.py": "4a9e8e523514e9c1c8d8c2ca2b8041ad5bcd4da5d1680b8079bea2e2e5a4532c",
        "scripts/run_fold.py": "0d2107fb59c545ca99523c81b360351745abafd1ebef85910223011681040677",
        "scripts/summarize.py": "90ecde6cab14897204cf0a6300c8a7cafbac05243f6744aa8feb0a2d69877346",
        "scripts/validate_fold.py": "f048af56c55c7638da95bcb4ec9c909d58032922a5a856bcfa06c8846a98922b",
    },
    "W": {
        "config/protocol.json": "5789e4d25fb5be95a7b77de82baa283724bbf3d3c0f689695704b288dbfb7c39",
        "config/tasks.json": "9e759a64bb05ab2d888e21dfba5da0dec03e854a0028e4e579eeebbebd81e938",
        "scripts/common.py": "f4a52476e077968fa559934b9d56ef9d105a65d0308ccf91961597110cb44be6",
        "scripts/preflight.py": "ad5eacd859f972ef51a6f96482f369ded652c990198b27971d068ac3cdb7e257",
        "scripts/run_fold.py": "19f393f3733b7e5e7affa4557ddce6e574000cc734e52e07a5176cd0a337d801",
        "scripts/summarize.py": "5f99f6e34287e9adfc5b691401182dd853a5e8aa296ea4243fa4a22820b6887b",
        "scripts/validate_fold.py": "46ffa76cdbb8a96f7020eafb854edb59378e0a1bb2ecaef36f589606e8bd289a",
    },
}
SOURCE_SHA256 = {"source_snapshot/" + name: value for name, value in _ROOT_SOURCE_SHA256.items()}
for _protocol, _entries in _PROTOCOL_SHA256.items():
    SOURCE_SHA256.update({
        f"source_snapshot/experiments/{_protocol}/" + name: value
        for name, value in _entries.items()
    })
    SOURCE_SHA256.update({
        f"source_snapshot/experiments/{_protocol}/frozen_ds/" + name: value
        for name, value in _SHARED_FROZEN_SHA256.items()
    })
EXPECTED_FILES = SUPPORT_FILES | MANAGED_FILES | set(SOURCE_SHA256)
BLOCKED_PARTS = {
    "data", "weights", "metadata", "provenance", "internal", "private", "ops",
    "replay_outputs", ".venv", "venv", "__pycache__",
}
BLOCKED_EXTENSIONS = {
    ".csv", ".tsv", ".npz", ".npy", ".pt", ".pth", ".ckpt", ".safetensors",
    ".zip", ".7z", ".tar", ".gz", ".edf", ".bdf", ".set", ".fdt", ".log",
    ".h5", ".hdf5", ".mat", ".pkl", ".pickle", ".onnx", ".bin", ".pyc",
}
SECRET_PATTERNS = (
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}"),
    re.compile(rb"AKIA[A-Z0-9]{16}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    re.compile(rb'''(?im)\b(?:password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*["'][^"'\r\n]{8,}["']'''),
)


def repository_root() -> Path:
    chosen = _CLI_REPOSITORY or os.environ.get("PD_EEG_REPOSITORY")
    root = Path(chosen).expanduser() if chosen else Path(__file__).resolve().parents[1]
    root = root.resolve(strict=True)
    if not (root / "repo_files_manifest.json").is_file():
        raise ValueError("Set --repository or PD_EEG_REPOSITORY to the prepared repository root")
    return root


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def is_link_or_reparse(path: Path) -> bool:
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def safe_repository_file(root: Path, relative: str) -> Path:
    """Validate every ancestor before opening an approved static file."""
    if relative not in EXPECTED_FILES:
        raise ValueError(relative)
    path = root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if is_link_or_reparse(path):
            raise ValueError(relative)
    if not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
        raise ValueError(relative)
    return path


def inventory(root: Path) -> tuple[dict[str, Path], list[str]]:
    """Do not follow links, including Windows junction/reparse directories."""
    files: dict[str, Path] = {}
    rejected: list[str] = []
    expected_dirs = {
        parent.as_posix()
        for name in EXPECTED_FILES
        for parent in PurePosixPath(name).parents
        if parent.as_posix() != "."
    }
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative == ".git":
                directories.remove(name)
            elif is_link_or_reparse(path) or relative not in expected_dirs:
                rejected.append(relative)
                directories.remove(name)
        for name in names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative == ".git":
                continue  # A Git worktree can use a root .git file.
            if is_link_or_reparse(path) or not path.is_file():
                rejected.append(relative)
            else:
                files[relative] = path
    return files, rejected


def load_maintenance_tool(root: Path):
    """Only the standard-library maintenance tool is imported, never a snapshot."""
    path = safe_repository_file(root, "tools/prepare_source_snapshot.py")
    specification = importlib.util.spec_from_file_location("pd_eeg_static_maintenance", path)
    if specification is None or specification.loader is None:
        raise ValueError("tools/prepare_source_snapshot.py")
    module = importlib.util.module_from_spec(specification)
    with mock.patch.object(sys, "dont_write_bytecode", True):
        specification.loader.exec_module(module)
    return module


class RepositoryIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = repository_root()

    def assert_paths_clean(self, paths) -> None:
        paths = sorted(set(paths))
        if paths:
            self.fail("\n".join(paths))  # Paths only; never print content/secret matches.

    def read_json(self, relative: str):
        try:
            return json.loads(safe_repository_file(self.root, relative).read_bytes())
        except (OSError, ValueError, UnicodeError):
            self.fail(relative)

    def test_explicit_repository_whitelist(self) -> None:
        files, rejected = inventory(self.root)
        self.assert_paths_clean(rejected + sorted(set(files) ^ EXPECTED_FILES))
        self.assertEqual(len(SOURCE_SHA256), 52)
        self.assertEqual(len(EXPECTED_FILES), 67)

    def test_no_archive_payloads_or_secrets(self) -> None:
        files, rejected = inventory(self.root)
        findings = list(rejected)
        for relative, path in files.items():
            parts = {part.lower() for part in PurePosixPath(relative).parts}
            if (
                parts & BLOCKED_PARTS
                or path.suffix.lower() in BLOCKED_EXTENSIONS
                or path.name.lower().startswith(".env")
                or path.name.lower() == "input_origins.json"
                or relative not in EXPECTED_FILES
            ):
                findings.append(relative)
                continue  # Unknown/payload files are never opened.
            content = path.read_bytes()
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                findings.append(relative)
        self.assert_paths_clean(findings)

    def test_python_and_json_parse_without_importing_frozen_source(self) -> None:
        findings = []
        for relative in sorted(EXPECTED_FILES):
            try:
                path = safe_repository_file(self.root, relative)
                if path.suffix == ".py":
                    ast.parse(path.read_bytes(), filename=relative)
                elif path.suffix == ".json":
                    json.loads(path.read_bytes())
            except (OSError, SyntaxError, ValueError, UnicodeError):
                findings.append(relative)
        self.assert_paths_clean(findings)

    def test_frozen_52_source_files_match_independent_sha256(self) -> None:
        findings = []
        for relative, expected in SOURCE_SHA256.items():
            try:
                path = safe_repository_file(self.root, relative)
                if digest(path) != expected:
                    findings.append(relative)
            except (OSError, ValueError):
                findings.append(relative)
        self.assert_paths_clean(findings)

    def test_manifest_exact_whitelist_size_and_sha256(self) -> None:
        manifest = self.read_json("repo_files_manifest.json")
        self.assertEqual(manifest.get("schema_version"), 2)
        self.assertEqual(manifest.get("source_snapshot_files"), 52)
        self.assertEqual(set(manifest.get("hash_exempt_generated_artifacts", [])), MANAGED_FILES)
        entries = manifest.get("files", {})
        expected_hashed = EXPECTED_FILES - MANAGED_FILES
        self.assert_paths_clean(set(entries) ^ expected_hashed)
        findings = []
        for relative in sorted(expected_hashed):
            try:
                path = safe_repository_file(self.root, relative)
                entry = entries.get(relative, {})
                if path.stat().st_size != entry.get("bytes") or digest(path) != entry.get("sha256"):
                    findings.append(relative)
            except (OSError, ValueError):
                findings.append(relative)
        self.assert_paths_clean(findings)

    def test_receipt_binds_manifest_without_new_experiment_claims(self) -> None:
        receipt = self.read_json("local_ready_receipt.json")
        manifest_path = safe_repository_file(self.root, "repo_files_manifest.json")
        if receipt.get("repo_files_manifest_sha256") != digest(manifest_path):
            self.fail("repo_files_manifest.json")
        for key in (
            "training_called", "forward_inference_called", "metrics_recomputed",
            "signal_or_model_object_loaded", "threshold_selection_called",
            "frozen_source_or_config_modified", "external_operations_performed",
        ):
            self.assertIs(receipt.get(key), False, "local_ready_receipt.json")
        status = self.read_json("PUBLICATION_STATUS.json")
        self.assertIs(receipt.get("public_release_authorized"), status.get("public_release_authorized"))

    def test_mit_license_present(self) -> None:
        text = safe_repository_file(self.root, "LICENSE").read_text(encoding="utf-8")
        for phrase in (
            "MIT License", "Copyright", "Permission is hereby granted, free of charge",
            'THE SOFTWARE IS PROVIDED "AS IS"',
        ):
            self.assertTrue(phrase in text, "LICENSE")

    def test_publication_status_matches_code_only_scope(self) -> None:
        status = self.read_json("PUBLICATION_STATUS.json")
        self.assertEqual(status.get("verified_repository_visibility"), "public", "PUBLICATION_STATUS.json")
        self.assertIs(status.get("public_release_authorized"), True, "PUBLICATION_STATUS.json")
        self.assertIs(status.get("visibility_change_requires_explicit_owner_instruction"), True, "PUBLICATION_STATUS.json")
        self.assertEqual(status.get("code_license"), "MIT", "PUBLICATION_STATUS.json")
        self.assertIs(status.get("project_authored_code_MIT_approved"), True, "PUBLICATION_STATUS.json")
        text = str(status.get("PUBLICATION_STATUS", "")).lower()
        self.assertIn("public", text, "PUBLICATION_STATUS.json")
        self.assertIs(status.get("weights_or_participant_records_published"), False, "PUBLICATION_STATUS.json")
        self.assertIn("ds002778", status.get("publication_scope", ""))


class SyntheticArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = load_maintenance_tool(repository_root())

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="pd_eeg_synthetic_test_")
        self.addCleanup(self.temporary.cleanup)
        self.fixture_root = Path(self.temporary.name)
        self.archive = self.fixture_root / "synthetic_archive.zip"
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("synthetic_fixture.txt", date_time=(2020, 1, 1, 0, 0, 0))
            archive.writestr(info, b"synthetic fixture only\n")
        self.lock = {
            "archive_name": self.archive.name,
            "bytes": self.archive.stat().st_size,
            "sha256": digest(self.archive),
        }
        self.write_lock()
        self.root_patch = mock.patch.object(self.tool, "ROOT", self.fixture_root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def write_lock(self) -> None:
        (self.fixture_root / "archive.lock.json").write_text(json.dumps(self.lock), encoding="utf-8")

    def test_synthetic_archive_correct_hash(self) -> None:
        self.assertIs(self.tool.verify_archive(self.archive), True)

    def test_synthetic_archive_wrong_hash_rejected(self) -> None:
        self.lock["sha256"] = "0" * 64
        self.write_lock()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.tool.verify_archive(self.archive)

    def test_synthetic_archive_wrong_filename_rejected(self) -> None:
        self.lock["archive_name"] = "other_synthetic_archive.zip"
        self.write_lock()
        with self.assertRaisesRegex(ValueError, "filename"):
            self.tool.verify_archive(self.archive)

    def test_synthetic_archive_wrong_size_rejected(self) -> None:
        self.lock["bytes"] += 1
        self.write_lock()
        with self.assertRaisesRegex(ValueError, "size"):
            self.tool.verify_archive(self.archive)

    def test_no_archive_requested_is_not_a_verification_claim(self) -> None:
        self.assertIs(self.tool.verify_archive(None), False)

    def test_synthetic_small_plain_file_can_be_hashed(self) -> None:
        fixture = self.fixture_root / "synthetic_small_file.txt"
        fixture.write_bytes(b"abc")
        self.assertEqual(self.tool.digest(fixture), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def guarded_mock_path(self):
        path = mock.Mock(spec=["is_symlink", "is_junction", "lstat", "resolve"])
        path.is_symlink.return_value = False
        path.is_junction.return_value = False
        path.lstat.return_value = mock.Mock(st_file_attributes=0)
        path.resolve.return_value.is_relative_to.return_value = True
        return path

    def test_path_guard_rejects_symlink_before_resolving(self) -> None:
        path = self.guarded_mock_path()
        path.is_symlink.return_value = True
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.tool.require_local_regular_path(path, self.fixture_root)
        path.resolve.assert_not_called()

    def test_path_guard_rejects_windows_junction_before_resolving(self) -> None:
        path = self.guarded_mock_path()
        path.is_junction.return_value = True
        with self.assertRaisesRegex(ValueError, "junction"):
            self.tool.require_local_regular_path(path, self.fixture_root)
        path.resolve.assert_not_called()

    def test_path_guard_rejects_windows_reparse_before_resolving(self) -> None:
        path = self.guarded_mock_path()
        path.lstat.return_value = mock.Mock(st_file_attributes=0x400)
        with self.assertRaisesRegex(ValueError, "reparse"):
            self.tool.require_local_regular_path(path, self.fixture_root)
        path.resolve.assert_not_called()

    def test_path_guard_rejects_resolved_root_escape(self) -> None:
        path = self.guarded_mock_path()
        path.resolve.return_value.is_relative_to.return_value = False
        with self.assertRaisesRegex(ValueError, "escapes root"):
            self.tool.require_local_regular_path(path, self.fixture_root)
        path.resolve.assert_called_once_with(strict=True)
        path.resolve.return_value.is_relative_to.assert_called_once_with(self.fixture_root)

    def test_path_guard_accepts_temporary_regular_file(self) -> None:
        self.tool.require_local_regular_path(self.archive, self.fixture_root.resolve())

    def test_maintenance_inventory_checks_root_and_each_member(self) -> None:
        guard = self.tool.require_local_regular_path
        with mock.patch.object(self.tool, "require_local_regular_path", wraps=guard) as wrapped:
            items = self.tool.inventory()
        root = self.fixture_root.resolve()
        wrapped.assert_any_call(self.fixture_root, root)
        wrapped.assert_any_call(self.archive, root)
        wrapped.assert_any_call(self.fixture_root / "archive.lock.json", root)
        self.assertEqual(set(items), {"archive.lock.json", "synthetic_archive.zip"})

    def test_maintenance_main_rejects_optimized_mode(self) -> None:
        with mock.patch.object(self.tool.sys, "flags", mock.Mock(optimize=1)):
            with self.assertRaisesRegex(ValueError, "Optimized Python mode"):
                self.tool.main()

    def test_maintenance_inventory_rejects_symlink_without_reading_target(self) -> None:
        target = self.fixture_root / "synthetic_target.txt"
        target.write_bytes(b"synthetic fixture only\n")
        link = self.fixture_root / "synthetic_link.txt"
        try:
            link.symlink_to(target)
        except (NotImplementedError, OSError):
            self.skipTest("This platform/account cannot create a temporary symlink")
        with self.assertRaises(ValueError):
            self.tool.inventory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repository", metavar="ROOT", help="Prepared repository root; overrides PD_EEG_REPOSITORY")
    known, remaining = parser.parse_known_args()
    _CLI_REPOSITORY = known.repository
    unittest.main(argv=[sys.argv[0], *remaining])
