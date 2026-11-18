from __future__ import annotations

import polars as pl
import pytest

from trifact.ingestion.langdetect import LangConfig, detect, map_label


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("__label__kk", "kk"),
        ("__label__ru", "ru"),
        ("__label__kaz_Cyrl", "kk"),
        ("__label__rus_Cyrl", "ru"),
        ("__label__eng_Latn", "en"),
        ("__label__kir_Cyrl", "ky"),
        ("__label__kaz_Latn", "kaz_Latn"),  # Latin-script Kazakh is not the kk target
        ("__label__kaz_Arab", "kaz_Arab"),
        ("__label__fra_Latn", "fra_Latn"),
    ],
)
def test_map_label(raw: str, want: str) -> None:
    assert map_label(raw) == want


class FakeLid:
    """Cyrillic with a Kazakh letter -> kk, other Cyrillic -> ru, Latin -> en."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def predict(self, texts: list[str]) -> tuple[list[str], list[float]]:
        self.seen.extend(texts)
        out = []
        for t in texts:
            if any(c in "әғқңөұүһі" for c in t.lower()):
                out.append(("kk", 0.9))
            elif any("а" <= c <= "я" for c in t.lower()):
                out.append(("ru", 0.8))
            else:
                out.append(("en", 1.0))
        return [o[0] for o in out], [o[1] for o in out]


def test_detect_doc_and_paragraph() -> None:
    kk = "Қазақстан Республикасы Орталық Азияда орналасқан мемлекет."
    ru = "Это предложение написано на русском языке."
    texts = pl.Series(
        [
            f"{kk}\n\n{kk}\n{ru}",  # 2 kk paragraphs outweigh 1 ru
            "Hi\nline",  # no paragraph with >= 20 letters: longest line classified
            "",
            "12345 6789",  # no letters at all: nothing classified
        ]
    )
    lid = FakeLid()
    out = detect(texts, lid, LangConfig(min_para_chars=20, max_para_chars=30))  # type: ignore[arg-type]
    assert out["lang"].to_list() == ["kk", "en", "und", "und"]
    assert out["para_lang"].to_list() == [["kk", "kk", "ru"], [None, "en"], [], [None]]
    m_kk, m_ru = 2 * len(kk) * 0.9, len(ru) * 0.8
    assert out["lang_score"][0] == pytest.approx(m_kk / (m_kk + m_ru), rel=1e-5)
    assert out["lang_conf"][0] == pytest.approx(0.9, rel=1e-5)
    assert out["lang_score"][2] == 0.0
    assert all(len(t) <= 30 for t in lid.seen)  # max_para_chars truncation
