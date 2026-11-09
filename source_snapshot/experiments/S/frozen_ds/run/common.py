"""Common entry helpers: env, device, output dirs, dataset loading."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_ROOT = Path(os.environ.get("DS_OUTPUT_ROOT", "/home/jerry/dx/deepseek/outputs"))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="DS0", help="registered model name")
    parser.add_argument("--dataset", default="ds002778", help="ds002778 | ds003490")
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default=None, help="output directory (defaults under outputs/)")
    parser.add_argument("--no-deterministic", action="store_true")


def resolve_output_dir(args, subdir: str) -> Path:
    if args.output_dir:
        path = Path(args.output_dir)
    else:
        path = OUTPUT_ROOT / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
