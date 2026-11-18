"""Wikitext -> plain prose paragraphs with link spans (silver NER, ADR-0015).

Only running prose is kept: templates, tables, references, HTML, headings, list and table lines,
file / category / interwiki links are removed. Internal links `[[Target|anchor]]trail` become their
anchor text (plus the word trail) and are returned as (start, end, target title) spans.
"""

from __future__ import annotations

import re

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_REF = re.compile(r"<ref[^>/]*?/>|<ref[^>]*?>.*?</ref>", re.DOTALL | re.IGNORECASE)
_BLOCK = re.compile(
    r"<(math|gallery|timeline|score|syntaxhighlight|source|pre|code|graph|mapframe|imagemap|poem|chem|ce)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG = re.compile(r"</?[a-zA-Z][^>]*?/?>")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_TABLE = re.compile(r"\{\|[^{}]*?\|\}", re.DOTALL)
_MEDIA = re.compile(
    r"\[\[\s*(?:file|image|media|category|файл|изображение|категория|сурет|санат|медиа|категория)\s*:[^\[\]]*"
    r"(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]",
    re.IGNORECASE,
)
_INTERWIKI = re.compile(r"\[\[\s*:?[a-z]{2,3}(?:-[a-z]+)?\s*:[^\[\]]*\]\]")
_MARKUP = re.compile(
    r"\[\[([^\[\]|]+)(?:\|([^\[\]]*))?\]\](\w*)|\[(?:https?:)?//[^\s\]]+(?:\s([^\]]*))?\]|'{2,}"
)
_SKIP_LINE = re.compile(r"^\s*(?:[=*#:;|!{}]|__|\[\[)")
_LEFTOVER = re.compile(r"\{\{|\}\}|\[\[|\]\]|\{\||\|\}|<|>|&[a-z]+;")


def _strip_nested(pattern: re.Pattern[str], text: str, rounds: int = 12) -> str:
    for _ in range(rounds):
        new = pattern.sub("", text)
        if new == text:
            return new
        text = new
    return text


def normalize_title(title: str) -> str:
    """MediaWiki title form: no anchor, spaces for underscores, first letter upper-cased."""
    t = title.split("#", 1)[0].replace("_", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t[:1].upper() + t[1:] if t else t


def prose(wikitext: str) -> list[tuple[str, list[tuple[int, int, str]]]]:
    """Paragraphs of running text with their internal link spans (start, end, normalized target)."""
    t = _COMMENT.sub("", wikitext)
    t = _REF.sub("", t)
    t = _BLOCK.sub("", t)
    t = _strip_nested(_TEMPLATE, t)
    t = _strip_nested(_TABLE, t)
    t = _MEDIA.sub("", t)
    t = _INTERWIKI.sub("", t)
    t = _TAG.sub("", t)
    out = []
    for line in t.split("\n"):
        if not line.strip() or _SKIP_LINE.match(line):
            continue
        buf: list[str] = []
        links: list[tuple[int, int, str]] = []
        pos = 0
        n = 0
        for m in _MARKUP.finditer(line):
            buf.append(line[pos : m.start()])
            n += m.start() - pos
            pos = m.end()
            if m.group(0).startswith("'"):
                continue
            if m.group(1) is not None:
                target = m.group(1)
                anchor = (m.group(2) if m.group(2) is not None else target) + m.group(3)
                anchor = anchor.strip()
                if anchor:
                    links.append((n, n + len(anchor), normalize_title(target)))
                    buf.append(anchor)
                    n += len(anchor)
            else:
                label = (m.group(4) or "").strip()
                buf.append(label)
                n += len(label)
        buf.append(line[pos:])
        para = "".join(buf)
        if _LEFTOVER.search(para):
            continue
        # collapse spaces left by removed templates; shift link offsets accordingly
        para, links = _collapse(para, links)
        if para:
            out.append((para, links))
    return out


def _collapse(text: str, links: list[tuple[int, int, str]]) -> tuple[str, list[tuple[int, int, str]]]:
    keep = []
    mapping = []
    prev_space = True
    for ch in text:
        sp = ch.isspace()
        if sp and prev_space:
            mapping.append(len(keep))
            continue
        mapping.append(len(keep))
        keep.append(" " if sp else ch)
        prev_space = sp
    mapping.append(len(keep))
    s = "".join(keep)
    s = s.rstrip()
    new = [(mapping[a], min(mapping[b], len(s)), t) for a, b, t in links if mapping[a] < mapping[b]]
    return s, [(a, b, t) for a, b, t in new if a < b and s[a:b].strip()]
