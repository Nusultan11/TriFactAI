from __future__ import annotations

from pathlib import Path

import polars as pl

from trifact.data.corpus_stats import sample, volumes


def corpus(tmp_path: Path) -> list[str]:
    df = pl.DataFrame(
        {
            "doc_id": [f"d{i}" for i in range(6)],
            "source": ["wiki"] * 4 + ["web"] * 2,
            "lang": ["kk", "kk", "ru", "en", "kk", "kk"],
            "domain": ["wikipedia"] * 4 + ["web"] * 2,
            "n_chars": [50, 500, 5000, 150, 20000, 99],
            "n_words": [8, 80, 800, 25, 3000, 15],
            "para_lang": [["kk"], ["kk", "ru"], ["ru"], ["en", "en"], ["kk"], None],
            "text": ["x " * 25, "y " * 250, "z " * 2500, "w " * 75, "v " * 10000, "u " * 49],
        }
    )
    path = tmp_path / "part-00000.parquet"
    df.write_parquet(path)
    return [str(path)]


def test_volumes_groups_histogram_and_mixed_share(tmp_path: Path) -> None:
    v = volumes(corpus(tmp_path), [0, 100, 1000, 10000])
    kk = next(r for r in v["by_lang"] if r["lang"] == "kk")
    assert kk["docs"] == 4 and kk["chars"] == 50 + 500 + 20000 + 99
    assert kk["mixed_lang_share"] == 0.25  # one kk document has a ru paragraph
    assert v["length_histogram"]["kk"] == {"0-100": 2, "100-1000": 1, "1000-10000": 0, "10000+": 1}
    assert {(r["source"], r["lang"]) for r in v["by_source_lang_domain"]} == {
        ("web", "kk"), ("wiki", "en"), ("wiki", "kk"), ("wiki", "ru")
    }  # fmt: skip


def test_sample_is_capped_and_reproducible(tmp_path: Path) -> None:
    files = corpus(tmp_path)
    by = pl.DataFrame(volumes(files, [0, 100])["by_source_lang_domain"])
    a = sample(files, by, 1, 10, seed=42)
    assert a.equals(sample(files, by, 1, 10, seed=42))
    assert a["text"].str.len_chars().max() <= 10
    full = sample(files, by, 100, 10, seed=42)
    assert full.height == 6
