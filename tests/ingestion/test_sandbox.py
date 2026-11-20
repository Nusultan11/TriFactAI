from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from trifact.ingestion.parser import ParseError
from trifact.ingestion.sandbox import SafeParser

HELPERS = textwrap.dedent(
    """
    import os, time
    from trifact.ingestion.parser import ParseError, assemble, Block

    class Fake:
        def parse(self, raw, filename):
            if raw == b"hang":
                time.sleep(60)
            if raw == b"crash":
                os._exit(3)
            if raw == b"bad":
                raise ParseError("corrupted", "bad bytes")
            if raw == b"boom":
                raise ValueError("unexpected")
            if raw == b"eat":
                hog = [bytearray(50 << 20) for _ in range(40)]
                time.sleep(60)
            return assemble([Block("text", raw.decode())])

    def make():
        return Fake()
    """
)


@pytest.fixture
def safe(tmp_path: Path) -> SafeParser:
    (tmp_path / "sandbox_helpers_mod.py").write_text(HELPERS, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))  # the spawned child inherits sys.path
    import sandbox_helpers_mod

    sp = SafeParser(sandbox_helpers_mod.make, timeout_s=3, max_rss_gb=0.8, poll_s=0.05)
    yield sp
    sp.close()
    sys.path.remove(str(tmp_path))


def test_isolated_parser_survives_hang_crash_memory_and_errors(safe: SafeParser) -> None:
    assert safe.parse(b"hello world", "a.txt").text == "hello world"
    for raw, reason in [(b"hang", "timeout"), (b"crash", "crash"), (b"bad", "corrupted"), (b"boom", "corrupted"),
                        (b"eat", "memory")]:  # fmt: skip
        with pytest.raises(ParseError) as e:
            safe.parse(raw, "x.txt")
        assert e.value.reason == reason
        assert safe.parse(b"still alive", "a.txt").text == "still alive"
    assert safe.restarts >= 3
