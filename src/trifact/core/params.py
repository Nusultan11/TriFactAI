"""Read params.yaml (the single source of numbers) and load .env into the process environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]


def load_params(path: Path | None = None) -> dict[str, Any]:
    load_dotenv(ROOT / ".env", override=False)
    params: dict[str, Any] = yaml.safe_load((path or ROOT / "params.yaml").read_text(encoding="utf-8"))
    return params
