"""Fitness tests: the repository stays consistent with docs/architecture (ADR-0001, §11 of the architecture)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import yaml

from trifact.core.schemas import Answer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "trifact"
DOCS = ROOT / "docs"
ARCH_DOC = DOCS / "architecture" / "docs" / "01-architecture.md"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_core_depends_on_nothing_else_in_trifact() -> None:
    for path in (SRC / "core").glob("*.py"):
        bad = {m for m in _imports(path) if m.startswith("trifact.") and not m.startswith("trifact.core")}
        assert not bad, f"{path.name} imports {bad}"


def test_every_dvc_stage_points_to_an_existing_module() -> None:
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    for name, stage in stages.items():
        cmd = stage.get("do", stage)["cmd"]
        module = re.search(r"python -m ([\w.]+)", cmd).group(1)
        path = ROOT / "src" / Path(*module.split(".")).with_suffix(".py")
        assert path.is_file(), f"stage {name}: {module} not found"


def test_acquired_sources_are_declared_in_params() -> None:
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    acquired = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]["acquire"]["foreach"]
    assert set(acquired) <= set(params["sources"])
    assert params["scale"]["profile"] in {"pilot", "full"}


def test_every_adr_is_referenced_from_another_document() -> None:
    texts = {p: p.read_text(encoding="utf-8") for p in DOCS.rglob("*.md")}
    for adr in (DOCS / "architecture" / "adr").glob("*.md"):
        others = (t for p, t in texts.items() if p != adr)
        assert any(adr.name in t for t in others), f"{adr.name} is not referenced"


def test_answer_contract_in_architecture_matches_schema() -> None:
    doc = ARCH_DOC.read_text(encoding="utf-8")
    block = re.search(r"\*\*Response contract\*\*\s*```json\n(.*?)```", doc, re.DOTALL).group(1)
    example = json.loads(block)
    claim = example["claims"][0]
    claim["verdict"] = "supported"  # the doc lists the enum as "a | b | c"
    Answer.model_validate(example)


def test_dvc_dependencies_use_lf_line_endings() -> None:
    # dvc.lock hashes bytes: a CRLF working copy locks hashes that a fresh LF checkout never reproduces
    files = [ROOT / "params.yaml", ROOT / "dvc.yaml", *(ROOT / "src").rglob("*.py")]
    assert [str(f.relative_to(ROOT)) for f in files if b"\r\n" in f.read_bytes()] == []
