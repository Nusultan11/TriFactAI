"""Stage build_kb: Wikidata entities -> knowledge base for entity linking and silver NER (raw/wikidata -> task/kb).

python -m trifact.data.build_kb [--workers N]

1. Class closure: the P279 (subclass of) graph of all entities, walked downwards from the root classes
   of every core label in `configs/labels.yaml: wikidata_classes` for `wikidata_max_depth` levels, gives
   the class sets of PER / ORG / LOC (a class reachable from several labels keeps the first of
   PER > ORG > LOC).
2. NER type of an entity: the label of the first of its P31 (instance of) classes found in those sets.
3. Tables, written one raw shard at a time (memory bounded by a shard, ADR-0011):
   * `entities` — entities with a label or alias in at least `linking.kb_min_labels` of kk/ru/en
     (or the multilingual `mul` label): qid, ner_type, labels, descriptions, aliases, sitelinks,
     n_sitelinks, p31;
   * `titles` — (wiki, lang, title, qid, ner_type) for every kk/ru/en Wikipedia sitelink: the
     link resolver of silver NER (Wikipedia links point to titles);
   * `classes` — (class qid, ner_type, depth) of the closure, for audit.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from trifact.core.params import ROOT, load_params
from trifact.data.curation import fingerprint, shard_entry, write_report
from trifact.data.task_common import finish, inherited, raw_files, raw_input
from trifact.lake.layout import LakeConfig, ShardInfo, sha256_file
from trifact.lake.manifest import ShardEntry
from trifact.lake.stats import ResourceMonitor, log_mlflow

log = logging.getLogger("trifact.build_kb")
LABELS = ROOT / "configs" / "labels.yaml"
PRIORITY = ["PER", "ORG", "LOC"]
WIKIS = {"kkwiki": "kk", "ruwiki": "ru", "enwiki": "en"}


def class_closure(files: list[str], roots: dict[str, list[str]], depth: int) -> pl.DataFrame:
    """(class, ner_type, depth): descendants of the root classes through P279 edges."""
    edges = (
        pl.scan_parquet(files)
        .select(child="qid", parent="p279")
        .filter(pl.col("parent").list.len() > 0)
        .explode("parent")
        .collect(engine="streaming")
    )
    log.info("P279 graph: %d edges", edges.height)
    found = pl.DataFrame(
        [{"cls": q, "ner_type": lab, "depth": 0} for lab in PRIORITY for q in roots.get(lab, [])],
        schema={"cls": pl.String, "ner_type": pl.String, "depth": pl.Int32},
    )
    frontier = found
    for d in range(1, depth + 1):
        nxt = (
            frontier.join(edges, left_on="cls", right_on="parent")
            .select(cls="child", ner_type="ner_type", depth=pl.lit(d, dtype=pl.Int32))
            .join(found.select("cls", "ner_type"), on=["cls", "ner_type"], how="anti")
            .unique(["cls", "ner_type"])
        )
        if nxt.is_empty():
            break
        found = pl.concat([found, nxt])
        frontier = nxt
        log.info("depth %d: +%d classes", d, nxt.height)
    rank = {lab: i for i, lab in enumerate(PRIORITY)}
    return (
        found.with_columns(r=pl.col("ner_type").replace_strict(rank, return_dtype=pl.Int32))
        .sort("r", "depth")
        .unique("cls", keep="first")
        .drop("r")
        .sort("cls")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    params = load_params()
    root = (ROOT / params["lake"]["root"]).resolve()
    out = (args.out or root / "task" / "kb").resolve()
    labels = yaml.safe_load(LABELS.read_text(encoding="utf-8"))
    roots, depth = labels["wikidata_classes"], int(labels["wikidata_max_depth"])
    min_labels = int(params["linking"]["kb_min_labels"])
    stage_params = {"roots": roots, "depth": depth, "kb_min_labels": min_labels}
    files = raw_files(root, "wikidata", "entities")
    fp = fingerprint("build_kb", stage_params, [Path(__file__), LABELS], [root / "raw/wikidata/source.yaml"])
    lake = LakeConfig.from_params(params["lake"], root)
    if out.exists():
        shutil.rmtree(out)
    (out / "multi").mkdir(parents=True)
    t0 = time.perf_counter()
    with ResourceMonitor() as mon:
        classes = class_closure(files, roots, depth)
        cls_path = out / "multi" / "classes-00000.parquet"
        classes.write_parquet(cls_path, compression=lake.compression)
        langs = ["kk", "ru", "en", "mul"]
        ent_dir = out / "multi"
        n_ent = 0
        shards: list[ShardEntry] = []
        rank = {k: i for i, k in enumerate(PRIORITY)}
        cmap = classes.select(p31="cls", ner_type="ner_type")
        has = sum(
            ((pl.col(f"label_{lg}").is_not_null()) | (pl.col(f"aliases_{lg}").list.len() > 0)).cast(pl.Int32)
            for lg in langs[:3]
        )
        # one raw shard at a time: memory is bounded by a shard (~0.6 M entities), not by 122 M entities
        for k, f in enumerate(files):
            df = pl.read_parquet(f)
            typed = (
                df.select("qid", "p31").explode("p31").join(cmap, on="p31", how="inner")
                .with_columns(r=pl.col("ner_type").replace_strict(rank, return_dtype=pl.Int32))
                .group_by("qid").agg(pl.col("ner_type").sort_by("r").first())
            )  # fmt: skip
            df = df.join(typed, on="qid", how="left").with_columns(pl.col("ner_type").fill_null("O"))
            ent = df.filter((has >= min_labels) | pl.col("label_mul").is_not_null()).select(
                "qid", "ner_type", "type",
                *[f"label_{lg}" for lg in langs], *[f"description_{lg}" for lg in langs],
                *[f"aliases_{lg}" for lg in langs], *[f"sitelink_{w}" for w in WIKIS], "n_sitelinks",
                p31=pl.col("p31").list.head(5),
            )  # fmt: skip
            ent.write_parquet(ent_dir / f"entities-{k:05d}.parquet", compression=lake.compression)
            titles = pl.concat([
                df.filter(pl.col(f"sitelink_{w}").is_not_null()).select(
                    wiki=pl.lit(w), lang=pl.lit(lg), title=pl.col(f"sitelink_{w}"), qid="qid", ner_type="ner_type")
                for w, lg in WIKIS.items()
            ])  # fmt: skip
            titles.write_parquet(ent_dir / f"titles-{k:05d}.parquet", compression=lake.compression)
            if k % 20 == 0:
                log.info("shard %d/%d", k + 1, len(files))
        for p in sorted(ent_dir.glob("*.parquet")):
            n = pl.scan_parquet(p).select(pl.len()).collect().item()
            table = p.name.split("-")[0]
            n_ent += n if table == "entities" else 0
            shards.append(shard_entry(out, ShardInfo(p, n, p.stat().st_size, sha256_file(p)), "multi", table))
    kb = pl.scan_parquet([str(p) for p in ent_dir.glob("entities-*.parquet")])
    cover = (
        kb.select(
            entities=pl.len(),
            **{f"label_{lg}": pl.col(f"label_{lg}").is_not_null().sum() for lg in langs},
            **{f"description_{lg}": pl.col(f"description_{lg}").is_not_null().sum() for lg in langs},
            **{f"with_aliases_{lg}": (pl.col(f"aliases_{lg}").list.len() > 0).sum() for lg in langs},
            **{f"sitelink_{w}": pl.col(f"sitelink_{w}").is_not_null().sum() for w in WIKIS},
        )
        .collect(engine="streaming")
        .to_dicts()[0]
    )
    by_type = kb.group_by("ner_type").len().sort("ner_type").collect(engine="streaming")
    tit = pl.scan_parquet([str(p) for p in ent_dir.glob("titles-*.parquet")])
    titles_by = tit.group_by("lang", "ner_type").len().sort("lang", "ner_type").collect(engine="streaming")
    metrics: dict[str, Any] = {
        "classes": classes.group_by("ner_type").len().sort("ner_type").to_dicts(),
        "coverage": cover,
        "by_ner_type": by_type.to_dicts(),
        "titles_by_lang_type": titles_by.to_dicts(),
    }
    finish(
        out, name="task/kb", layer="task", stage="build_kb", source="wikidata",
        description="Knowledge base for entity linking and silver NER: Wikidata entities with kk/ru/en labels "
        "(table `entities`), Wikipedia title -> QID resolver (`titles`), P279 class closure of PER/ORG/LOC (`classes`).",
        params=stage_params, inputs=[raw_input(root, "wikidata")], licenses=inherited(root, ["wikidata"]),
        shards=shards, rows_in=raw_input(root, "wikidata").rows, metrics=metrics,
        stats=mon.stats(n_ent, 0, 1), fingerprint=fp,
    )  # fmt: skip
    write_report(ROOT / "reports" / "kb.json", metrics)
    log_mlflow("task", "build_kb", {**stage_params, "fingerprint": fp},
               {"entities": n_ent, "seconds": round(time.perf_counter() - t0, 1),
                **{f"type_{r['ner_type']}": r["len"] for r in by_type.to_dicts()}})  # fmt: skip
    print(metrics["by_ner_type"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
