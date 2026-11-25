"""Offline review pages for the human review sheets (stage 2 quality, stage 3 silver NER and eval MT items).

python -m trifact.evaluation.review_app [--sheet quality|ner|eval]   # -> data/review/<dir>/review.html

One self-contained HTML file per sheet (`review_app.html` + the sheet rows + the machine pre-labels of
`trifact.evaluation.prelabel`), opened from disk in any browser, no server and no network. The reviewer enters
a name, labels with the keyboard, progress stays in the browser (localStorage), and "Export CSV" writes the
file the scoring command reads, in its exact columns:

* quality -> `labels.csv` for `python -m trifact.evaluation.review_quality score --review-dir data/review/quality_human`;
* ner     -> `labels.csv` (sheet columns filled) for `python -m trifact.evaluation.review_ner score`;
* eval    -> `sheet-01.csv` (verdict, fixed_text, fixed_source, reviewer) for `python -m trifact.data.build_eval freeze`.

Exported rows carry `assist` (`machine_shown` or `blind`: whether the machine suggestion was visible) and the
time of the label. The machine suggestion is shown as such and can be hidden; export refuses an empty reviewer
name or one that names a model. A row counts as human only when a person picked its label.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.evaluation.prelabel import LABELS, SHEETS, en_nli, machine_dir, read_sheet

TEMPLATE = Path(__file__).with_name("review_app.html")
HELP = {
    "quality": """<b>Проверка фильтров качества (этап 2).</b> Прочитайте начало документа (600 символов) и решите:
<b>good</b> — естественный текст на указанном языке, который стоит использовать для обучения и поиска (проза, новости,
энциклопедия, книги, посты; короткий тоже годится); <b>bad</b> — шаблон или навигация сайта, списки и таблицы без прозы,
битая кодировка или мусор OCR, спам и SEO, машинная бессмыслица, обрывок без смысла, в основном другой язык.
Решение фильтра (оставлен или отброшен) намеренно не показано. Клавиши: <kbd>G</kbd> good, <kbd>B</kbd> bad,
<kbd>J</kbd>/<kbd>K</kbd> — следующий/предыдущий. Готовый файл сохраните как <code>labels.csv</code> в
<code>data/review/quality_human/</code>.""",
    "ner": """<b>Проверка silver NER (этап 3).</b> Сущности размечены автоматически по ссылкам Википедии и типам Wikidata:
PER — человек, ORG — организация, LOC — место (страна, город, регион, объект, здание). В kk/ru падежное окончание
входит в спан. Даты, национальности, языки, события, произведения — не сущности. Посчитайте: сущности с неверным
типом, с неверными границами, отмеченное не-сущность, пропущенные PER/ORG/LOC. Если ошибок нет — оставьте нули и
нажмите <kbd>Enter</kbd>. Готовый файл сохраните как <code>labels.csv</code> в <code>data/review/ner_silver/</code>.""",
    "eval": """<b>Проверка машинных переводов eval (этап 3, ROADMAP 3.5).</b> Запросы retrieval: <kbd>O</kbd> ok —
перевод точен и естественен (имена, числа, даты сохранены); <kbd>F</kbd> fix — впишите правильный перевод;
<kbd>D</kbd> drop — исходный запрос сам бессмыслен. NLI kk: метка должна оставаться верной для казахской пары;
fix — исправьте гипотезу и/или посылку; drop — метку не спасти. Для заморозки нужно проверить весь пакет из 3 500 элементов.
Готовый файл сохраните как <code>sheet-01.csv</code> в <code>data/eval/review/</code> (заменив пустой лист).""",
}
EXPORT = {
    "quality": ("labels.csv", ["doc_id", "review_lang", "status", "stratum", "label", "note", "reviewer", "assist", "reviewed_at"]),
    "ner": ("labels.csv", ["sent_id", "lang", "text", "entities", "wrong_type", "wrong_span", "not_entity", "missed",
                           "reviewer", "note", "assist", "reviewed_at"]),
    "eval": ("sheet-01.csv", ["item_id", "set", "kind", "direction", "source_text", "candidate_text", "context", "label",
                              "verdict", "fixed_text", "fixed_source", "reviewer", "note", "assist", "took_machine_fix",
                              "reviewed_at"]),
}  # fmt: skip
TITLES = {
    "quality": "Этап 2 — качество документов",
    "ner": "Этап 3 — silver NER",
    "eval": "Этап 3 — переводы eval",
}


def clean(v: Any) -> Any:
    return "" if v is None else v


def build(name: str, params: dict[str, Any], root: Path) -> Path:
    sheet = read_sheet(name)
    id_col = SHEETS[name][1]
    mfile = machine_dir(name) / LABELS
    machine: dict[str, dict[str, Any]] = {}
    if mfile.exists():
        m = pl.read_csv(mfile, infer_schema_length=10000)
        machine = {r[id_col]: {k: clean(v) for k, v in r.items()} for r in m.iter_rows(named=True)}
    en = en_nli(root) if name == "eval" else {}
    items = []
    for r in sheet.iter_rows(named=True):
        row = {k: clean(v) for k, v in r.items()}
        if name == "eval" and row["kind"] == "nli_translation":
            row["en_premise"], row["en_hypothesis"] = en.get(
                row["item_id"].removeprefix("nli/xnli/kk/"), ("", "")
            )
        lang = row.get("review_lang") or row.get("lang") or row.get("direction", "").split("-")[0]
        items.append({"id": row[id_col], "lang": lang, "r": row, "m": machine.get(row[id_col], {})})
    labeler = next(iter(machine.values()), {}).get("labeler", "нет машинной предразметки")
    export_name, columns = EXPORT[name]
    data = {"sheet": name, "version": str(mfile.stat().st_mtime_ns if mfile.exists() else 0), "kind": name,
            "title": TITLES[name], "help": HELP[name], "id_col": id_col, "labeler": labeler,
            "export_name": export_name, "columns": columns, "items": items}  # fmt: skip
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out = machine_dir(name) / "review.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload), encoding="utf-8", newline="\n"
    )
    return out


def build_hub(out_dir: Path, links: dict[str, str], stats: dict[str, dict[str, Any]]) -> Path:
    """Build a single consolidated HTML hub linking all 3 human-review packets with exact save/CLI instructions."""
    cards = []
    meta = {
        "quality": (
            "1. Stage 2 — Качество документов (1 500)",
            "data/review/quality_human/labels.csv",
            'uv run --no-sync python -m trifact.evaluation.review_quality score --review-dir data/review/quality_human --reviewer "<Ваше имя>"',
        ),
        "ner": (
            "2. Stage 3 — Silver NER аудит (600)",
            "data/review/ner_silver/labels.csv",
            "uv run --no-sync python -m trifact.evaluation.review_ner score",
        ),
        "eval": (
            "3. Stage 3 — Eval MT & NLI (3 500)",
            "data/eval/review/sheet-01.csv",
            "uv run --no-sync python -m trifact.data.build_eval freeze",
        ),
    }
    for key in ("quality", "ner", "eval"):
        if key not in links:
            continue
        title, save_to, cmd = meta[key]
        st = stats.get(key, {})
        u = st.get("uncertainty", {})
        cards.append(
            f"""<div class="card">
  <h2>{title}</h2>
  <p>Элементов: <b>{st.get("items", 0)}</b> · Машинных оценок: <b>{st.get("judged", 0)}/{st.get("items", 0)}</b> ·
  Риск: <span class="hi">high {u.get("high", 0)}</span> / <span class="med">medium {u.get("medium", 0)}</span> / <span class="lo">low {u.get("low", 0)}</span></p>
  <p><a class="btn" href="{links[key]}">Открыть интерфейс проверки →</a></p>
  <p class="hint">Сохранить экспорт как: <code>{save_to}</code><br>Команда после проверки: <code>{cmd}</code></p>
</div>"""
        )
    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TriFact AI — Единый пакет проверки (Stages 2–3, 5 600 элементов)</title>
<style>
body {{ font:15px/1.5 system-ui, sans-serif; max-width:900px; margin:24px auto; padding:0 16px; background:#fafaf9; color:#1c1917; }}
.card {{ background:#fff; border:1px solid #e7e5e4; border-radius:10px; padding:16px; margin-bottom:16px; }}
.btn {{ display:inline-block; background:#2563eb; color:#fff; text-decoration:none; padding:8px 14px; border-radius:6px; font-weight:600; }}
.hint {{ color:#57534e; font-size:13px; }}
code {{ background:#f5f5f4; padding:2px 5px; border-radius:4px; }}
.hi {{ color:#b91c1c; font-weight:600; }} .med {{ color:#b45309; font-weight:600; }} .lo {{ color:#15803d; font-weight:600; }}
</style>
</head>
<body>
<h1>TriFact AI — Единый пакет проверки (Stages 2–3)</h1>
<p>Все <b>5 600 / 5 600</b> элементов имеют полные машинные оценки, рекомендации, исправления и уровни неопределённости.
Машинные метки <b>не являются человеческой проверкой</b> (ADR-0022): откройте каждый пакет, введите своё имя, проверьте спорные случаи (high/medium) и подтвердите корректные рекомендации, затем экспортируйте CSV.</p>
{"".join(cards)}
</body></html>
"""
    out = out_dir / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8", newline="\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", choices=list(SHEETS), action="append")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    built: dict[str, Path] = {}
    stats: dict[str, dict[str, Any]] = {}
    for name in args.sheet or list(SHEETS):
        p = build(name, params, root)
        built[name] = p
        mfile = machine_dir(name) / LABELS
        if mfile.exists():
            m = pl.read_csv(mfile, infer_schema_length=10000)
            vcol = {"quality": "judge_label", "ner": "judge_ok", "eval": "judge_verdict"}[name]
            judged = int(m[vcol].sum()) if name == "ner" else int(m[vcol].is_not_null().sum())
            unc = dict(m.group_by("uncertainty").len().iter_rows()) if "uncertainty" in m.columns else {}
            stats[name] = {"items": m.height, "judged": judged, "uncertainty": unc}
        print(name, "->", p)
    if len(built) == len(SHEETS):
        hub_links = {
            "quality": "quality_human/review.html",
            "ner": "ner_silver/review.html",
            "eval": "eval_mt/review.html",
        }
        hub = build_hub(ROOT / "data" / "review", hub_links, stats)
        print("hub ->", hub)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
