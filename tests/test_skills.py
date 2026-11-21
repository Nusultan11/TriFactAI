"""Agent Skills in `.agents/skills` follow the Agent Skills spec and are mirrored to `.claude/skills` unchanged."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / ".agents" / "skills"
SPEC_FIELDS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
LINK = re.compile(r"\]\(([^)\s]+)\)")

skill_dirs = sorted(p for p in SKILLS.glob("trifact-*") if p.is_dir())


def _frontmatter(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path}: frontmatter must open on the first line"
    head, body = text[4:].split("\n---\n", 1)
    return yaml.safe_load(head), body


def test_skills_exist() -> None:
    assert len(skill_dirs) >= 10


@pytest.mark.parametrize("skill", skill_dirs, ids=lambda p: p.name)
def test_skill_frontmatter_follows_spec(skill: Path) -> None:
    meta, body = _frontmatter(skill / "SKILL.md")
    assert set(meta) <= SPEC_FIELDS, f"non-spec fields: {set(meta) - SPEC_FIELDS}"
    name = meta["name"]
    assert isinstance(name, str) and NAME.match(name) and len(name) <= 64
    assert name == skill.name
    description = meta["description"]
    assert isinstance(description, str) and 0 < len(description) <= 1024
    assert "Use " in description, "description must say when to use the skill"
    assert len(body.splitlines()) < 500


@pytest.mark.parametrize("doc", sorted(SKILLS.rglob("*.md")), ids=lambda p: p.relative_to(SKILLS).as_posix())
def test_relative_links_resolve(doc: Path) -> None:
    for target in LINK.findall(doc.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = (doc.parent / target.split("#", 1)[0]).resolve()
        assert path.exists(), f"{doc.relative_to(ROOT)}: broken link {target}"


def test_claude_mirror_is_in_sync() -> None:
    spec = importlib.util.spec_from_file_location("sync_skills", ROOT / "scripts" / "sync_skills.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.diff() == [], "run: python scripts/sync_skills.py"
