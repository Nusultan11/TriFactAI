from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from trifact.core.params import ROOT, load_params
from trifact.ingestion.engines import Parser, _ranges
from trifact.ingestion.parser import Block, ParseError, assemble, parse_html, parse_txt, sniff

pymupdf = pytest.importorskip("pymupdf")
docx = pytest.importorskip("docx")

CFG = load_params()["docproc"]
ARIAL = Path("C:/Windows/Fonts/arial.ttf")
FONT = str(ARIAL) if ARIAL.is_file() else None


def check_spans(doc) -> None:  # type: ignore[no-untyped-def]
    t = doc.text
    if doc.pages:
        assert doc.pages[0][1] == 0 and doc.pages[-1][2] == len(t)
        for (_, _, e), (_, s, _) in zip(doc.pages, doc.pages[1:], strict=False):
            assert e <= s
    for title, s, e in doc.sections:
        assert t[s : s + len(title)] == title and s <= e <= len(t)
    for tb in doc.tables:
        assert t[tb.char_start : tb.char_end].split("\n")[0].split(" | ")[0] == tb.rows[0][0]


def test_assemble_offsets_pages_sections_tables() -> None:
    blocks = [
        Block("heading", "Қазақстан", 1, 1),
        Block("text", "Бірінчи  абзац\u00a0мәтіні.", 1),
        Block("heading", "География", 2, 2),
        Block("table", page=2, rows=[["Атауы", "Жылы"], ["Алматы", "1854"]]),
        Block("text", "Соңғы абзац.", 3),
    ]
    doc = assemble(blocks, 4)
    check_spans(doc)
    assert [p[0] for p in doc.pages] == [1, 2, 3, 4]
    assert doc.pages[3][1] == doc.pages[3][2] == len(doc.text)  # empty last page
    assert doc.page_at(doc.text.index("Алматы")) == 2
    assert doc.text.startswith("Қазақстан\n\nБірінчи абзац мәтіні.")
    assert doc.sections[0][2] == len(doc.text)  # level-1 section spans the nested one
    assert doc.section_at(doc.text.index("Соңғы")) == "География"
    assert doc.tables[0].rows == [["Атауы", "Жылы"], ["Алматы", "1854"]]


def test_sniff_by_content_not_extension() -> None:
    assert sniff(b"%PDF-1.7 ...", "x.txt") == "pdf"
    assert sniff(b"<!doctype html><html><body>x</body></html>", "x.bin") == "html"
    assert sniff("Сәлем әлем".encode(), "notes") == "txt"
    for raw, name, reason in [
        (b"\xd0\xcf\x11\xe0" + b"\0" * 100, "a.doc", "unsupported_format"),
        (b"PK\x03\x04garbage", "a.docx", "corrupted"),
        (b"\x89PNG\r\n\x1a\n\0\0\0", "a.pdf", "corrupted"),
        (bytes(range(256)) * 20, "a.txt", "unsupported_format"),
    ]:
        with pytest.raises(ParseError) as e:
            sniff(raw, name)
        assert e.value.reason == reason


@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "utf-16", "cp1251"])
def test_txt_encodings(enc: str) -> None:
    text = "# Заголовок\n\nПервый абзац текста о Казахстане.\r\nВторая строка.\n\nЕщё абзац."
    doc = parse_txt(text.encode(enc))
    assert doc.sections and doc.sections[0][0] == "Заголовок"
    assert "Первый абзац текста о Казахстане.\nВторая строка." in doc.text
    check_spans(doc)


def test_html_drops_boilerplate_keeps_structure() -> None:
    html = """<html><head><title>T</title><script>var a=1;</script></head><body>
    <header><nav><a>Home</a><a>News</a></nav></header>
    <main><article><h1>Абай</h1><p>Абай — <b>ақын</b>, <a href="#">ағартушы</a>.</p>
    <h2>Өмірі</h2><ul><li>Бірінші</li><li>Екінші</li></ul>
    <table><tr><th>Жыл</th><th>Оқиға</th></tr><tr><td>1845</td><td>Туған</td></tr></table>
    </article></main><footer>© 2024</footer></body></html>"""
    doc = parse_html(html.encode())
    assert "Home" not in doc.text and "© 2024" not in doc.text and "var a" not in doc.text
    assert "Абай — ақын, ағартушы." in doc.text
    assert [s[0] for s in doc.sections] == ["Абай", "Өмірі"] and doc.levels == [1, 2]
    assert doc.tables[0].rows == [["Жыл", "Оқиға"], ["1845", "Туған"]]
    check_spans(doc)


def test_html_deep_nesting_does_not_crash() -> None:
    html = (
        "<html><body>" + "<div>" * 5000 + "<p>Deep paragraph text.</p>" + "</div>" * 5000 + "</body></html>"
    )
    assert "Deep paragraph text." in Parser(CFG, ROOT).parse(html.encode(), "x.html").text


def _docx(rendered_breaks: bool) -> bytes:
    from docx.oxml import OxmlElement

    d = docx.Document()
    d.add_heading("Title of doc", level=0)
    d.add_paragraph("First page paragraph.")
    if rendered_breaks:
        p = d.add_paragraph("Second page paragraph.")
        p.runs[0]._r.insert(0, OxmlElement("w:lastRenderedPageBreak"))
    else:
        d.add_page_break()
        d.add_paragraph("Second page paragraph.")
    d.add_heading("Section", level=1)
    t = d.add_table(rows=2, cols=2)
    for r, row in enumerate([["a", "b"], ["c", "d"]]):
        for c, v in enumerate(row):
            t.cell(r, c).text = v
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


@pytest.mark.parametrize("rendered", [False, True])
def test_docx_headings_tables_pages(rendered: bool) -> None:
    doc = Parser(CFG, ROOT).parse(_docx(rendered), "a.docx")
    assert [(s[0], lv) for s, lv in zip(doc.sections, doc.levels, strict=True)] == [
        ("Title of doc", 1),
        ("Section", 2),
    ]
    assert doc.tables[0].rows == [["a", "b"], ["c", "d"]]
    assert doc.page_at(doc.text.index("First")) == 1 and doc.page_at(doc.text.index("Second")) == 2
    check_spans(doc)


def _pdf(pages: list[list[tuple[float, float, str, float]]]) -> bytes:
    d = pymupdf.open()
    for items in pages:
        p = d.new_page(width=595, height=842)
        for x, y, text, size in items:
            kw = {"fontfile": FONT, "fontname": "F0"} if FONT else {}
            p.insert_text((x, y), text, fontsize=size, **kw)
    return d.tobytes()


def test_pdf_fast_path_two_columns_headers_footers() -> None:
    pages = []
    for n in range(1, 4):
        items = [
            (56, 36, "Running header of the report", 8),
            (290, 820, str(n), 8),
            (56, 80, f"Chapter {n}", 18),
        ]
        for i in range(12):
            items.append((56, 120 + i * 14, f"left column line {n}.{i} with words", 10))
            items.append((310, 120 + i * 14, f"right column line {n}.{i} with words", 10))
        pages.append(items)
    doc = Parser(CFG, ROOT, pdf="pymupdf", ocr=None).parse(_pdf(pages), "a.pdf")
    t = doc.text
    assert "Running header" not in t and "\n\n2\n\n" not in t
    assert t.index("left column line 1.11") < t.index("right column line 1.0")  # column by column
    assert t.index("right column line 1.11") < t.index("Chapter 2")
    assert [s[0] for s in doc.sections] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert doc.page_at(t.index("left column line 2.0")) == 2
    check_spans(doc)


def test_pdf_errors_have_reasons() -> None:
    p = Parser(CFG, ROOT, pdf="pymupdf", ocr=None)
    src = pymupdf.open()
    src.new_page().insert_text((50, 50), "secret text")
    enc = src.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    cases = [(b"", "empty"), (b"%PDF-1.4\n" + b"\0garbage" * 100, "corrupted"), (enc, "encrypted")]
    blank = pymupdf.open()
    blank.new_page()
    cases.append((blank.tobytes(), "no_text"))
    for raw, reason in cases:
        with pytest.raises(ParseError) as e:
            p.parse(raw, "x.pdf")
        assert e.value.reason == reason
    many = pymupdf.open()
    for _ in range(5):
        many.new_page()
    small = {**CFG, "limits": {**CFG["limits"], "max_pages": 4}}
    with pytest.raises(ParseError) as e:
        Parser(small, ROOT, pdf="pymupdf", ocr=None).parse(many.tobytes(), "x.pdf")
    assert e.value.reason == "too_many_pages"


def test_ranges() -> None:
    assert _ranges([1, 2, 3, 5, 7, 8]) == [(1, 3), (5, 5), (7, 8)]


def test_zip_without_word_document_is_rejected() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", "x")
    with pytest.raises(ParseError):
        Parser(CFG, ROOT).parse(buf.getvalue(), "a.docx")
