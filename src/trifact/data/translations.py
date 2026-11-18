"""Machine translations from the Kaggle NLLB job -> `data/task/mt/<name>` with quality signals (ADR-0015, ADR-0020).

python -m trifact.data.translations import NAME --kaggle-out DIR --sentences FILE --sources a,b

Table `translations` (one row per sentence and target language): `id, src_lang, tgt_lang, text,
translation, back, chrf_back, len_ratio, mt_lang, mt_lang_score, copy`. Quality signals:
* `chrf_back` — chrF (character 1..6-grams, beta 2, whitespace ignored) of the back-translation against
  the source: round-trip adequacy, 0..100;
* `len_ratio` — characters of translation / source;
* `mt_lang`, `mt_lang_score` — lid.176 language of the translation;
* `copy` — translation equals the source (untranslated).
The model and its licence are part of the lineage: NLLB-200 is CC-BY-NC-4.0, so every derived set
inherits `release: false` (ADR-0015).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from trifact.core.params import ROOT, load_params
from trifact.data.curation import fingerprint, write_report
from trifact.data.task_common import DatasetWriter, finish, inherited, raw_input
from trifact.ingestion.langdetect import LangId, model_path
from trifact.lake.layout import LakeConfig, sha256_file
from trifact.lake.manifest import DatasetInput, InheritedLicense, LicenseUse
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.translations")
NLLB = {"eng_Latn": "en", "rus_Cyrl": "ru", "kaz_Cyrl": "kk"}
MT_LICENSE = InheritedLicense(
    source="nllb-200", license="CC-BY-NC-4.0", use=LicenseUse.research_only, release=False
)


def _ngrams(s: str, n: int) -> Counter[str]:
    s = "".join(s.split())
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015): mean character n-gram precision / recall, F-beta; 0..100."""
    if not hyp or not ref:
        return 0.0
    ps, rs = [], []
    for n in range(1, max_n + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        if not h or not r:
            continue
        m = sum((h & r).values())
        ps.append(m / sum(h.values()))
        rs.append(m / sum(r.values()))
    if not ps:
        return 0.0
    p, r = sum(ps) / len(ps), sum(rs) / len(rs)
    return 0.0 if p + r == 0 else round(100 * (1 + beta**2) * p * r / (beta**2 * p + r), 2)


def import_run(name: str, kaggle_out: Path, sentences: Path, sources: list[str]) -> int:
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    out = root / "task" / "mt" / name
    files = sorted((kaggle_out / "out").glob("*.parquet"))
    if not files:
        raise SystemExit(f"no translation parts in {kaggle_out / 'out'}")
    summary = json.loads((kaggle_out / "summary.json").read_text())
    cfg_path = next((kaggle_out.parent / "in").glob("config.json"))
    cfg = json.loads(cfg_path.read_text())
    stage_params = {"model": cfg["model"], "back_model": cfg.get("back_model"), "beams": cfg.get("beams"),
                    "max_new_tokens": cfg.get("max_new_tokens"), "kaggle": json.loads((kaggle_out / "run.json").read_text())}  # fmt: skip
    fp = fingerprint("translations", {k: v for k, v in stage_params.items() if k != "kaggle"}, [Path(__file__)],
                     [*files, sentences])  # fmt: skip
    t0 = time.perf_counter()
    lake = LakeConfig.from_params(params["lake"], root)
    spec = params["curation"]["langid"]
    lid = LangId(model_path("lid176", spec["models"]["lid176"], ROOT / spec["cache"]), "lid176")
    with ResourceMonitor() as mon:
        parts = []
        for f in files:
            stem = f.name.rsplit("-g", 1)[0]  # <file stem>-<tgt>
            src_file, tgt = stem.rsplit("-", 1)
            parts.append(pl.read_parquet(f).with_columns(tgt_code=pl.lit(tgt), file=pl.lit(src_file)))
        df = pl.concat(parts)
        src_code = {t["file"].removesuffix(".parquet"): t["src"] for t in cfg["tasks"]}
        df = df.with_columns(
            src_lang=pl.col("file").replace_strict({k: NLLB[v] for k, v in src_code.items()}),
            tgt_lang=pl.col("tgt_code").replace_strict(NLLB),
        )
        df = df.with_columns(
            chrf_back=pl.struct("back", "text").map_elements(
                lambda r: chrf(r["back"] or "", r["text"] or ""), return_dtype=pl.Float64
            ),
            len_ratio=(pl.col("translation").str.len_chars() / pl.col("text").str.len_chars().clip(1)).round(
                3
            ),
            copy=pl.col("translation").str.strip_chars() == pl.col("text").str.strip_chars(),
        )
        langs, scores = lid.predict([t.replace("\n", " ") for t in df["translation"].fill_null("").to_list()])
        df = df.with_columns(mt_lang=pl.Series(langs), mt_lang_score=pl.Series(scores, dtype=pl.Float32))
        df = df.select(
            "id",
            "src_lang",
            "tgt_lang",
            "text",
            "translation",
            "back",
            "chrf_back",
            "len_ratio",
            "mt_lang",
            "mt_lang_score",
            "copy",
        )
        if out.exists():
            shutil.rmtree(out)
        w = DatasetWriter(out, lake)
        w.write("translations", df.sort("src_lang", "tgt_lang", "id"), lang_col="tgt_lang")
        shards = w.close()
    q = df.group_by("src_lang", "tgt_lang").agg(
        n=pl.len(), chrf_back_mean=pl.col("chrf_back").mean().round(2),
        chrf_back_p10=pl.col("chrf_back").quantile(0.1).round(2), chrf_back_p50=pl.col("chrf_back").median().round(2),
        wrong_lang=(pl.col("mt_lang") != pl.col("tgt_lang")).mean().round(4), copies=pl.col("copy").sum(),
        len_ratio_p05=pl.col("len_ratio").quantile(0.05), len_ratio_p95=pl.col("len_ratio").quantile(0.95),
        empty=(pl.col("translation").str.strip_chars() == "").sum(),
    ).sort("src_lang", "tgt_lang")  # fmt: skip
    metrics: dict[str, Any] = {"by_direction": q.to_dicts(), "kaggle_summary": summary}
    inputs = [raw_input(root, s) for s in sources] + [
        DatasetInput(path=sentences.relative_to(root).as_posix(), sha256=sha256_file(sentences), rows={"multi": pl.scan_parquet(sentences).select(pl.len()).collect().item()})
    ]  # fmt: skip
    lic = inherited(root, sources) + [MT_LICENSE]
    finish(
        out, name=f"task/mt/{name}", layer="task", stage="translations", source="+".join(sources),
        description=f"NLLB-200 machine translations ({cfg['model']}, back-translation {cfg.get('back_model')}) with "
        "round-trip chrF, length ratio, language id and copy flags; produced on a Kaggle T4 x2 kernel.",
        params=stage_params, inputs=inputs, licenses=lic, shards=shards, rows_in={"multi": df.height},
        metrics=metrics, stats=mon.stats(df.height, 0, 1), fingerprint=fp,
    )  # fmt: skip
    write_report(ROOT / "reports" / f"mt_{name}.json", metrics)
    log_mlflow("task", f"mt-{name}", {k: str(v) for k, v in stage_params.items()},
               {"rows": df.height, "seconds": round(time.perf_counter() - t0, 1),
                **{f"chrf_{r['src_lang']}_{r['tgt_lang']}": r["chrf_back_mean"] for r in q.to_dicts()}})  # fmt: skip
    print(q)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("import")
    i.add_argument("name")
    i.add_argument("--kaggle-out", type=Path, required=True)
    i.add_argument("--sentences", type=Path, required=True, help="id/text table sent to the job")
    i.add_argument("--sources", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    return import_run(args.name, args.kaggle_out.resolve(), args.sentences.resolve(), args.sources.split(","))


if __name__ == "__main__":
    raise SystemExit(main())
