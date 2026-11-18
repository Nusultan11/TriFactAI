"""fastText language identification on paragraph and document level (architecture §5.3).

Models (lid.176, GlotLID) are pinned in `params.yaml: curation.langid.models` by URL or HF
revision plus sha256 and cached in `data/.cache/models`; `model_path` downloads and checks them.
Labels of both families are mapped to one code space: ISO 639-1 for the languages of interest
(`kk ru en ky tt ba uz uk be ...`) and the model's own label otherwise; GlotLID labels carry the
script, and only `kaz_Cyrl`/`rus_Cyrl`/`eng_Latn` map to `kk`/`ru`/`en`.

Paragraph = non-empty line. A paragraph is classified when it has >= `min_para_chars` letters,
or when it is the longest line of a document that has no such line; prediction uses its first
`max_para_chars` characters. Document language = the language with the largest share of
classified characters weighted by score; `lang_score` is that share (agreement, 0..1) and
`lang_conf` the mean score of that language's paragraphs weighted by characters (confidence).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import polars as pl

log = logging.getLogger(__name__)

ISO3_TO_1 = {
    "kaz": "kk", "rus": "ru", "eng": "en", "kir": "ky", "tat": "tt", "bak": "ba", "uzb": "uz",
    "uzn": "uz", "ukr": "uk", "bel": "be", "bul": "bg", "srp": "sr", "mkd": "mk", "mon": "mn",
    "khk": "mn", "tuk": "tk", "aze": "az", "azj": "az", "tur": "tr", "chv": "cv", "sah": "sah",
    "kaa": "kaa", "crh": "crh", "kum": "kum", "nog": "nog", "krc": "krc", "alt": "alt", "tyv": "tyv",
}  # fmt: skip
EXPECTED_SCRIPT = {"kk": "Cyrl", "ru": "Cyrl", "en": "Latn", "ky": "Cyrl", "tt": "Cyrl", "ba": "Cyrl"}


def map_label(raw: str) -> str:
    """`__label__kk` / `__label__kaz_Cyrl` -> `kk`; script-mismatched GlotLID labels stay raw."""
    lab = raw.removeprefix("__label__")
    if "_" not in lab:
        return lab
    code, script = lab.split("_", 1)
    iso = ISO3_TO_1.get(code)
    if iso is None:
        return lab
    want = EXPECTED_SCRIPT.get(iso)
    return iso if want is None or want == script else lab


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def model_path(name: str, spec: dict[str, Any], cache_dir: Path) -> Path:
    """Local path of a pinned model; downloaded on first use, sha256 checked once per file."""
    cache_dir = Path(cache_dir)
    if "hf_id" in spec:
        path = cache_dir / name / spec["file"]
    else:
        path = cache_dir / name / Path(spec["url"]).name
    ok_marker = path.with_name(path.name + ".sha256")
    if path.is_file() and ok_marker.is_file() and ok_marker.read_text().strip() == spec["sha256"]:
        return path
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        if "hf_id" in spec:
            from huggingface_hub import hf_hub_download

            hf_hub_download(spec["hf_id"], spec["file"], revision=spec["revision"], local_dir=path.parent)
        else:
            tmp = path.with_name(path.name + ".part")
            with httpx.stream("GET", spec["url"], follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    f.writelines(r.iter_bytes(1 << 20))
            tmp.replace(path)
    digest = _sha256(path)
    if digest != spec["sha256"]:
        raise ValueError(f"{name}: sha256 {digest} != pinned {spec['sha256']}")
    ok_marker.write_text(digest)
    return path


class LangId:
    """One fastText language-id model with labels mapped by `map_label`."""

    def __init__(self, path: Path, name: str = ""):
        import fasttext

        self.name = name or Path(path).stem
        self._model = fasttext.load_model(str(path))
        # fasttext-predict's list API drops probabilities, so the per-line binding is used
        self._predict = self._model.f.predict
        self._labels: dict[str, str] = {}

    def predict(self, texts: list[str]) -> tuple[list[str], list[float]]:
        langs: list[str] = []
        scores: list[float] = []
        cache = self._labels
        pred = self._predict
        for t in texts:
            res = pred(t.replace("\n", " ") + "\n", 1, 0.0, "replace")
            if not res:
                langs.append("und")
                scores.append(0.0)
                continue
            p, raw = res[0]
            lab = cache.get(raw)
            if lab is None:
                lab = cache[raw] = map_label(raw)
            langs.append(lab)
            scores.append(min(float(p), 1.0))
        return langs, scores


@dataclass(frozen=True)
class LangConfig:
    min_para_chars: int = 20
    max_para_chars: int = 1000


def detect(texts: pl.Series, lid: LangId, cfg: LangConfig) -> pl.DataFrame:
    """Per document: `lang`, `lang_score`, `para_lang` (list aligned to non-empty lines, null = not classified)."""
    lines = (
        pl.DataFrame({"text": texts})
        .with_row_index("doc")
        .select("doc", pl.col("text").str.split("\n").alias("line"))
        .explode("line")
        .filter(pl.col("line").is_not_null() & (pl.col("line").str.len_bytes() > 0))
        .with_row_index("pos")
        .with_columns(
            n=pl.col("line").str.len_chars(),
            letters=pl.col("line").str.count_matches(r"\p{L}"),
        )
        .with_columns(
            longest=pl.col("pos") == pl.col("pos").sort_by("n", descending=True).first().over("doc"),
            any_long=(pl.col("letters") >= cfg.min_para_chars).any().over("doc"),
        )
        .with_columns(
            classify=(pl.col("letters") >= cfg.min_para_chars)
            | (~pl.col("any_long") & pl.col("longest") & (pl.col("letters") > 0))
        )
    )
    todo = lines.filter("classify")
    langs, scores = lid.predict(todo["line"].str.slice(0, cfg.max_para_chars).to_list())
    todo = todo.select("pos", "doc", "n").with_columns(
        plang=pl.Series(langs, dtype=pl.String), pscore=pl.Series(scores, dtype=pl.Float32)
    )
    para = lines.select("doc", "pos").join(todo.select("pos", "plang"), on="pos", how="left")
    para_lang = para.group_by("doc", maintain_order=True).agg(pl.col("plang").alias("para_lang"))
    mass = (
        todo.group_by("doc", "plang")
        .agg(m=(pl.col("n") * pl.col("pscore")).sum(), c=pl.col("n").sum())
        .with_columns(total=pl.col("m").sum().over("doc"))
        .sort(["doc", "m", "plang"], descending=[False, True, False])
        .group_by("doc", maintain_order=True)
        .agg(
            lang=pl.col("plang").first(),
            lang_score=(pl.col("m").first() / pl.col("total").first()),
            lang_conf=(pl.col("m").first() / pl.col("c").first()),
        )
    )
    n_docs = len(texts)
    return (
        pl.DataFrame({"doc": pl.arange(0, n_docs, eager=True, dtype=pl.UInt32)})
        .join(mass.with_columns(pl.col("doc").cast(pl.UInt32)), on="doc", how="left")
        .join(para_lang.with_columns(pl.col("doc").cast(pl.UInt32)), on="doc", how="left")
        .sort("doc")
        .select(
            pl.col("lang").fill_null("und"),
            pl.col("lang_score").fill_null(0.0).cast(pl.Float32),
            pl.col("lang_conf").fill_null(0.0).cast(pl.Float32),
            pl.col("para_lang").fill_null(pl.lit([], dtype=pl.List(pl.String))),
        )
    )
