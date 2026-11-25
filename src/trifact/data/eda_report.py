"""EDA report: figures, comparison tables and findings from the EDA profiles (ROADMAP 2.6, 3.7).

python -m trifact.data.eda_report [--layers raw,clean,curated,entities,task]

Reads `reports/eda/<layer>.json` (written by `trifact.data.eda` and `trifact.data.task_eda`) and the stage
reports (`reports/clean.json`, `dedup.json`, `decontamination.json`, `corpus_stats.json`,
`quality_review.json`, `langid.json`), writes `reports/eda/figures/*.png` and `reports/eda/report.md`, and
logs both to MLflow (experiment `eda`, run `eda-report`). Automatic findings (thresholds below) are listed
per profile; each finding is closed by a decision recorded in docs/PROGRESS.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Any

import numpy as np

from trifact.core.params import ROOT

log = logging.getLogger("trifact.eda_report")
EDA = ROOT / "reports" / "eda"
FIG = EDA / "figures"
LOG_BINS = np.concatenate([[0], np.unique(np.round(np.logspace(0, 8, 161)).astype(np.int64))])
# automatic findings: share thresholds over documents of a profile
FINDINGS = {
    "empty": 0.001, "docs_replacement_char": 0.001, "docs_control_chars": 0.001, "docs_mojibake": 0.001,
    "docs_email": 0.01, "docs_phone": 0.01, "docs_iin": 0.0001, "mostly_latin": 0.05, "null_share": 0.05,
}  # fmt: skip


def load(name: str) -> dict[str, Any] | None:
    p = (
        EDA / f"{name}.json"
        if name in ("raw", "clean", "curated", "entities", "task")
        else ROOT / "reports" / f"{name}.json"
    )
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def fmt(n: Any) -> str:
    if isinstance(n, float):
        return f"{n:.4g}"
    if isinstance(n, int):
        return f"{n:,}".replace(",", " ")
    return str(n)


def table(rows: list[list[Any]], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def js_divergence(p: list[float], q: list[float]) -> float:
    a, b = np.asarray(p, float), np.asarray(q, float)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = lambda x, y: float(np.nansum(np.where(x > 0, x * np.log2(x / y), 0)))
    return round(0.5 * kl(a, m) + 0.5 * kl(b, m), 4)


# ----------------------------------------------------------------------------- figures


def _plt() -> Any:
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    return plt


def fig_length(name: str, series: dict[str, list[int]], xlabel: str) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    centers = np.sqrt(np.maximum(LOG_BINS[:-1], 1) * LOG_BINS[1:])
    for label, h in series.items():
        h = np.asarray(h, float)
        if h.sum() == 0:
            continue
        ax.plot(centers, h / h.sum(), label=label, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("share of documents")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return f"figures/{path.name}"


def fig_bars(name: str, labels: list[str], values: list[float], xlabel: str, log: bool = False) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, max(2.0, 0.22 * len(labels) + 0.8)))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4C72B0")
    ax.set_yticks(y, labels, fontsize=7)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return f"figures/{path.name}"


def fig_heatmap(name: str, rows: list[str], cols: list[str], m: list[list[float]], title: str) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(1.2 + 0.55 * len(cols), 1.0 + 0.4 * len(rows)))
    arr = np.asarray(m, float)
    im = ax.imshow(arr, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if math.isfinite(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2g}", ha="center", va="center", fontsize=6, color="w")
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return f"figures/{path.name}"


def fig_zipf(name: str, curves: dict[str, list[list[int]]]) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for label, rf in curves.items():
        if rf:
            r, f = zip(*rf, strict=True)
            ax.loglog(r, f, marker=".", lw=1, label=label)
    ax.set_xlabel("rank")
    ax.set_ylabel("frequency (sample)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return f"figures/{path.name}"


# ----------------------------------------------------------------------------- sections


def text_findings(key: str, col: str, t: dict[str, Any]) -> list[str]:
    n = max(t["docs"], 1)
    out = []
    checks = {
        "empty": t["empty"], "docs_replacement_char": t["encoding"]["docs_replacement_char"],
        "docs_control_chars": t["encoding"]["docs_control_chars"], "docs_mojibake": t["encoding"]["docs_mojibake"],
        "docs_email": t["pii"]["docs_email"], "docs_phone": t["pii"]["docs_phone"], "docs_iin": t["pii"]["docs_iin"],
    }  # fmt: skip
    for k, v in checks.items():
        if v / n > FINDINGS[k]:
            out.append(f"`{key}.{col}`: {k} = {fmt(v)} ({100 * v / n:.2f}% of documents)")
    lang = key.split("/")[1] if "/" in key else ""
    if lang in ("kk", "ru") and t["script"]["docs_mostly_latin"] / n > FINDINGS["mostly_latin"]:
        out.append(
            f"`{key}.{col}`: {100 * t['script']['docs_mostly_latin'] / n:.1f}% documents mostly Latin script in a Cyrillic language"
        )
    if lang == "ru" and t["script"]["kk_letters"] / max(t["script"]["cyrillic_letters"], 1) > 0.002:
        out.append(
            f"`{key}.{col}`: Kazakh-specific letters {100 * t['script']['kk_letters'] / max(t['script']['cyrillic_letters'], 1):.2f}% of Cyrillic letters in ru"
        )
    if lang == "kk" and t["script"]["kk_letters"] / max(t["script"]["cyrillic_letters"], 1) < 0.03:
        out.append(
            f"`{key}.{col}`: only {100 * t['script']['kk_letters'] / max(t['script']['cyrillic_letters'], 1):.2f}% Kazakh-specific letters (Russian text or Latin transliteration?)"
        )
    return out


def layer_section(name: str, rep: dict[str, Any]) -> tuple[list[str], list[str]]:
    md = [
        f"## Layer `{name}`",
        "",
        f"Profiled in {rep['meta']['seconds']} s on {rep['meta']['workers']} workers, peak RSS {rep['meta']['peak_rss_mb']} MB.",
        "",
    ]
    findings: list[str] = []
    prof = rep["profiles"]
    rows = []
    for k, p in prof.items():
        drift = "yes" if p.get("schema_drift") else "no"
        idd = p.get("id", {})
        nulls = {c: v["null_share"] for c, v in p["columns"].items() if v["null_share"] > 0}
        const = [c for c, v in p["columns"].items() if v.get("constant") and not c.startswith("_")]
        empty = {c: v.get("empty_share", 0) for c, v in p["columns"].items() if v.get("empty_share", 0) > 0}
        rows.append([k, p["rows"], p["shards"], round(p["bytes"] / 2**20, 1), drift,
                     idd.get("duplicates_within_shards", "-"), idd.get("duplicates_across_shards", "-"),
                     ", ".join(f"{c} {v:.3f}" for c, v in nulls.items()) or "-", ", ".join(const) or "-",
                     ", ".join(f"{c} {v:.3f}" for c, v in empty.items()) or "-"])  # fmt: skip
        for c, v in nulls.items():
            if v > FINDINGS["null_share"] and not c.startswith("_"):
                findings.append(f"`{k}`: column `{c}` null in {100 * v:.1f}% rows")
        if p.get("schema_drift"):
            findings.append(f"`{k}`: {p['schemas']} different shard schemas")
        if idd and (idd.get("duplicates_within_shards") or idd.get("duplicates_across_shards")):
            findings.append(
                f"`{k}`: duplicate `{idd['column']}` values ({idd['duplicates_within_shards']} within, {idd['duplicates_across_shards']} across shards)"
            )
    md += ["### Schema and integrity", "", table(rows, ["profile", "rows", "shards", "MB", "schema drift", "dup ids in shard",
           "dup ids across shards", "null share", "constant columns", "empty-string share"]), ""]  # fmt: skip
    # volumes and lengths
    vrows = []
    lens: dict[str, list[int]] = {}
    script_rows, enc_rows, pii_rows = [], [], []
    zipf: dict[str, list[list[int]]] = {}
    vocab_rows, lex = [], []
    for k, p in prof.items():
        for c, t in p.get("text", {}).items():
            q = t["q_chars"]
            vrows.append([f"{k}.{c}", t["docs"], t["chars"], t["words"], t["paragraphs"], t["mean_chars"], q["p01"], q["p05"],
                          q["p50"], q["p95"], q["p99"], t["q_words"]["p50"]])  # fmt: skip
            if t["docs"] >= 1000 and c in ("text", "context", "kk", "ru", "en", "premise"):
                lens[f"{k}.{c}"] = t["hist_chars"]
            s = t["script"]
            script_rows.append([f"{k}.{c}", round(s["cyrillic_letters"] / max(s["all_letters"], 1), 4),
                                round(s["latin_letters"] / max(s["all_letters"], 1), 4),
                                round(s["kk_letters"] / max(s["cyrillic_letters"], 1), 4),
                                round(s["docs_with_kk_letters"] / max(t["docs"], 1), 4),
                                round(s["docs_mostly_latin"] / max(t["docs"], 1), 4)])  # fmt: skip
            e = t["encoding"]
            enc_rows.append(
                [
                    f"{k}.{c}",
                    t["empty"],
                    e["docs_replacement_char"],
                    e["docs_control_chars"],
                    e["docs_mojibake"],
                ]
            )
            pi = t["pii"]
            pii_rows.append([f"{k}.{c}", round(pi["docs_email"] / max(t["docs"], 1), 5), round(pi["docs_phone"] / max(t["docs"], 1), 5),
                             pi["docs_iin"], pi["card_numbers"], round(pi["docs_url"] / max(t["docs"], 1), 4)])  # fmt: skip
            findings += text_findings(k, c, t)
            v = t.get("vocab") or {}
            if v:
                vocab_rows.append([f"{k}.{c}", v["sample_docs"], v["tokens"], v["types"], v["type_token_ratio"],
                                   v["hapax_share_of_types"], v["hapax_share_of_tokens"], v["zipf_exponent"]])  # fmt: skip
                if c in ("text", "context") and v["tokens"] > 10000:
                    zipf[f"{k}.{c}"] = v["rank_freq"]
            if t.get("top_words"):
                lex.append(
                    (
                        f"{k}.{c}",
                        t["top_words"][:25],
                        v.get("top_bigrams", [])[:10],
                        t.get("top_lines", [])[:10],
                    )
                )
    md += ["### Volumes and lengths (characters; quantiles from log-binned full-volume histograms)", "",
           table(vrows, ["profile.column", "docs", "chars", "words", "paragraphs", "mean chars", "p01", "p05", "p50", "p95", "p99", "words p50"]), ""]  # fmt: skip
    if lens:
        md += [f"![lengths]({fig_length(f'{name}_lengths', lens, 'characters per document')})", ""]
    md += ["### Script and Kazakh letters", "", "Shares of letters; `kk letters` = ә ғ қ ң ө ұ ү һ і among Cyrillic letters.", "",
           table(script_rows, ["profile.column", "Cyrillic", "Latin", "kk letters / Cyrillic", "docs with kk letters", "docs mostly Latin"]), ""]  # fmt: skip
    md += [
        "### Encoding errors",
        "",
        table(
            enc_rows,
            [
                "profile.column",
                "empty docs",
                "docs with U+FFFD",
                "docs with control chars",
                "docs with mojibake",
            ],
        ),
        "",
    ]
    md += ["### Sensitive content (PII, regex over the full volume)", "",
           table(pii_rows, ["profile.column", "docs with e-mail", "docs with phone", "valid IIN", "Luhn card numbers", "docs with URL"]), ""]  # fmt: skip
    if vocab_rows:
        md += ["### Vocabulary (deterministic 2% document sample per shard)", "",
               table(vocab_rows, ["profile.column", "sample docs", "tokens", "types", "TTR", "hapax / types", "hapax / tokens", "Zipf exponent"]), ""]  # fmt: skip
    if zipf:
        md += [f"![zipf]({fig_zipf(f'{name}_zipf', zipf)})", ""]
    for k, words, bi, lines in lex[:40]:
        md += [f"**{k}** — top words: " + ", ".join(f"{w} ({fmt(n)})" for w, n in words[:20]), ""]
        if bi:
            md += ["top bigrams: " + ", ".join(f"{w} ({n})" for w, n in bi), ""]
        if lines:
            md += [
                "most repeated short lines (boilerplate candidates): "
                + "; ".join(f"`{ln[:60]}` ({fmt(n)})" for ln, n in lines[:6]),
                "",
            ]
    # domains, dates
    for k, p in prof.items():
        if p.get("hosts"):
            hosts = sorted(p["hosts"].items(), key=lambda x: -x[1])[:25]
            tld = sorted(p["tlds"].items(), key=lambda x: -x[1])[:15]
            md += [f"### Domains of `{k}`", "", f"![hosts]({fig_bars(f'{name}_{k.replace('/', '_')}_hosts', [h for h, _ in hosts], [v for _, v in hosts], 'documents', log=True)})", "",
                   "TLD: " + ", ".join(f".{t} {fmt(n)}" for t, n in tld), ""]  # fmt: skip
        if p.get("months"):
            ms = sorted(p["months"].items())
            years: dict[str, int] = {}
            for m, n in ms:
                years[m[:4]] = years.get(m[:4], 0) + n
            md += [
                f"Dates of `{k}` by year: " + ", ".join(f"{y}: {fmt(n)}" for y, n in sorted(years.items())),
                "",
            ]
        if p.get("groups"):
            g = sorted(p["groups"].items(), key=lambda x: -x[1])[:30]
            md += [f"Groups of `{k}`: " + ", ".join(f"{a}: {fmt(n)}" for a, n in g), ""]
    # extremes
    md += ["### Anomalous lengths (examples)", ""]
    for k, p in list(prof.items())[:60]:
        for c, t in p.get("text", {}).items():
            for kind, ex in t.get("extremes", {}).items():
                if ex:
                    e = ex[0]
                    md.append(
                        f"- `{k}` {kind}: {fmt(e['chars'])} chars, id `{e['id']}`: «{(e['text'] or '')[:160].replace(chr(10), ' ')}»"
                    )
    md.append("")
    # clean/curated specifics
    qrows, lrows = [], []
    for k, p in prof.items():
        if p.get("quality"):
            q = p["quality"]
            qrows.append([k, q["rows"], *[q["signals"].get(s, {}).get("mean", "-") for s in ("alpha_ratio", "stopword_ratio", "dup_line_ratio", "dup_ngram_ratio", "mean_word_len")],
                          ", ".join(f"{r}: {fmt(n)}" for r, n in sorted((q.get("reasons") or {}).items(), key=lambda x: -x[1])[:6]) or "-"])  # fmt: skip
            corr = q.get("correlation")
            if corr and k.endswith("/part") and k.count("/") == 2:
                md += [
                    f"![corr]({fig_heatmap(f'{name}_{k.replace("/", "_")}_corr', corr['cols'], corr['cols'], corr['spearman'], f'Spearman of quality signals, {k}')})",
                    "",
                ]
        if p.get("langid"):
            li = p["langid"]
            lrows.append([k, li["docs_with_other_target_lang_paragraph"], li["docs_with_non_target_paragraph"], li["paragraphs"],
                          ", ".join(f"{a} {fmt(b)}" for a, b in sorted(li["moves"].items(), key=lambda x: -x[1])[:5])])  # fmt: skip
    if qrows:
        md += [
            "### Quality signals (means) and drop reasons",
            "",
            table(
                qrows,
                [
                    "profile",
                    "docs",
                    "alpha",
                    "stopwords",
                    "dup lines",
                    "dup 5-grams",
                    "word len",
                    "drop reasons",
                ],
            ),
            "",
        ]
    if lrows:
        md += [
            "### Language mixing",
            "",
            table(
                lrows,
                [
                    "profile",
                    "docs with paragraph in another target language",
                    "docs with non-target paragraph",
                    "paragraphs",
                    "language moves src->detected",
                ],
            ),
            "",
        ]
    return md, findings


def entities_section(rep: dict[str, Any]) -> tuple[list[str], list[str]]:
    e = rep["profiles"]["wikidata/entities"]
    c = e["coverage"]
    n = c["entities"]
    md = [
        "## Wikidata `entities`",
        "",
        f"{fmt(n)} entities ({fmt(c['items'])} items, {fmt(c['properties'])} properties).",
        "",
    ]
    rows = []
    for lg in ("kk", "ru", "en", "mul"):
        rows.append([lg, c[f"label_{lg}"], round(c[f"label_{lg}"] / n, 4), c[f"description_{lg}"], round(c[f"description_{lg}"] / n, 4),
                     c[f"with_aliases_{lg}"], c[f"aliases_{lg}"]])  # fmt: skip
    md += [
        table(rows, ["lang", "labels", "share", "descriptions", "share", "entities with aliases", "aliases"]),
        "",
    ]
    md += [(f"Label in at least one of kk/ru/en: {fmt(c['label_any_target'])} ({100 * c['label_any_target'] / n:.1f}%); in all three: {fmt(c['label_all_three'])} "
           f"({100 * c['label_all_three'] / n:.2f}%). Sitelinks: kkwiki {fmt(c['sitelink_kkwiki'])}, ruwiki {fmt(c['sitelink_ruwiki'])}, enwiki {fmt(c['sitelink_enwiki'])}. "
           f"Without P31: {fmt(c['no_p31'])}."), ""]  # fmt: skip
    top = e["top_p31"][:30]
    md += [
        f"![p31]({fig_bars('entities_top_p31', [f'{r["p31"]} {r.get("label_en") or ""}'[:40] for r in top], [r['len'] for r in top], 'entities', log=True)})",
        "",
    ]
    md += [
        "Sitelinks per entity: " + ", ".join(f"{k} {v}" for k, v in e["n_sitelinks_quantiles"].items()),
        "",
    ]
    g = e["class_graph"]
    md += [(f"Class graph (P279): {fmt(g['classes'])} classes, {fmt(g['edges'])} edges, {fmt(g['components'])} weakly connected components, "
           f"largest {100 * g['largest_component_share']:.1f}% of classes; {fmt(g['instance_links_to_graph_classes'])} of {fmt(g['instance_links'])} P31 links point into the class graph."), ""]  # fmt: skip
    kk = e["kk_labelled_top_p31"][:15]
    md += [
        "Most frequent first P31 of kk-labelled entities: "
        + ", ".join(f"{r['p31']} ({fmt(r['len'])})" for r in kk),
        "",
    ]
    findings = []
    if c["label_kk"] / max(c["label_en"], 1) < 0.1:
        findings.append(
            f"Wikidata: kk labels cover only {100 * c['label_kk'] / n:.2f}% of entities vs en {100 * c['label_en'] / n:.1f}% — kk linking must fall back to ru/en labels and aliases"
        )
    return md, findings


def comparison_section(reps: dict[str, dict[str, Any]]) -> list[str]:
    """raw -> clean -> curated: documents, characters and length distributions of the corpus sources."""
    md = ["## raw → clean → curated", ""]
    rows = []
    for src in ("wikipedia", "fineweb2_kk", "multidomain_kk"):
        for lg in ("kk", "ru", "en"):
            r = (
                reps.get("raw", {})
                .get("profiles", {})
                .get(f"{src}/{lg}/part", {})
                .get("text", {})
                .get("text")
            )
            c = (
                reps.get("clean", {})
                .get("profiles", {})
                .get(f"{src}/{lg}/part", {})
                .get("text", {})
                .get("text")
            )
            cu = reps.get("curated", {}).get("profiles", {}).get(f"corpus/{src}/{lg}/part", {}) or reps.get(
                "curated", {}
            ).get("profiles", {}).get(f"corpus/{lg}/part", {})
            cu = cu.get("text", {}).get("text") if cu else None
            if not r:
                continue
            row = [f"{src}/{lg}", r["docs"], c["docs"] if c else "-", cu["docs"] if cu else "-",
                   r["chars"], c["chars"] if c else "-", cu["chars"] if cu else "-",
                   r["q_chars"]["p50"], c["q_chars"]["p50"] if c else "-", cu["q_chars"]["p50"] if cu else "-",
                   js_divergence(r["hist_chars"], c["hist_chars"]) if c else "-",
                   js_divergence(r["hist_chars"], cu["hist_chars"]) if cu else "-"]  # fmt: skip
            rows.append(row)
    md += [table(rows, ["source/lang", "raw docs", "clean docs", "curated docs", "raw chars", "clean chars", "curated chars",
                        "raw p50", "clean p50", "curated p50", "JS raw-clean", "JS raw-curated"]), ""]  # fmt: skip
    md += [
        "JS = Jensen-Shannon divergence (bits) of the log-binned character-length distributions: 0 = unchanged.",
        "",
    ]
    return md


def stage_section() -> list[str]:
    md = ["## Stage reports", ""]
    for name in ("langid", "quality_review", "clean", "dedup", "decontamination", "corpus_stats"):
        r = load(name)
        if r is None:
            md.append(f"- `{name}`: not produced yet")
            continue
        md.append(f"- `reports/{name}.json`: present")
    cs = load("corpus_stats")
    if cs and cs.get("fertility"):
        rows = []
        for tok, f in cs["fertility"].items():
            for r in f["by_lang"]:
                rows.append(
                    [tok, r["lang"], r["docs"], r["tokens_per_word"], r["tokens_per_1k_chars"], r["unk_rate"]]
                )
        md += [
            "",
            "### Tokenizer fertility (curated sample)",
            "",
            table(rows, ["tokenizer", "lang", "docs", "tokens/word", "tokens/1k chars", "unk rate"]),
        ]
    if cs and cs.get("toxicity"):
        tx = cs["toxicity"]
        rows = [[r["source"], r["lang"], r["docs"], r["toxic"], r["share"], f"{r['ci95'][0]}-{r['ci95'][1]}",
                 "yes" if r["validated_language"] else "no (kk not in training data)"] for r in tx["by_source_lang"]]  # fmt: skip
        md += [
            "",
            f"### Toxicity (curated sample, `{tx['model']}`, p >= {tx['threshold']})",
            "",
            table(rows, ["source", "lang", "docs", "toxic", "share", "95% CI", "model trained on lang"]),
        ]
    md.append("")
    return md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", default="raw,clean,curated,entities,task")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    FIG.mkdir(parents=True, exist_ok=True)
    md = ["# TriFact AI — EDA report", "", ("Generated by `python -m trifact.data.eda_report` from `reports/eda/*.json`. "
          "Findings are automatic; the decision for each is recorded in docs/PROGRESS.md."), ""]  # fmt: skip
    findings: list[str] = []
    reps = {}
    for layer in args.layers.split(","):
        rep = load(layer)
        if rep is None:
            md += [f"## Layer `{layer}`", "", "Not profiled yet.", ""]
            continue
        reps[layer] = rep
        if layer == "entities":
            s, f = entities_section(rep)
        elif layer == "task":
            from trifact.data.task_eda import task_section

            s, f = task_section(rep, FIG)
        else:
            s, f = layer_section(layer, rep)
        md += s
        findings += f
    md += comparison_section(reps)
    md += stage_section()
    md += ["## Automatic findings", ""] + [f"{i + 1}. {x}" for i, x in enumerate(findings)] + [""]
    out = EDA / "report.md"
    out.write_text("\n".join(md), encoding="utf-8", newline="\n")
    (EDA / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("report: %s, %d findings, %d figures", out, len(findings), len(list(FIG.glob("*.png"))))
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment("eda")
            with mlflow.start_run(run_name="eda-report"):
                mlflow.log_param("layers", args.layers)
                mlflow.log_metric("findings", len(findings))
                mlflow.log_artifacts(str(EDA), artifact_path="eda")
        except Exception as e:  # noqa: BLE001 - report files are the primary record
            log.warning("MLflow logging of the EDA report failed: %s", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
