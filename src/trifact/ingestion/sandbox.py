"""Isolated parsing: a `DocumentParser` that runs the real parser in a child process (architecture §6.1).

MuPDF, the Docling models and the OCR engines are native code: a hostile file can crash, hang or exhaust
memory. `SafeParser` keeps one long-lived child process (models stay loaded) and enforces, per document, the
`docproc.limits` timeout and resident-memory cap; when the child dies or is killed the document fails with
`ParseError("crash" | "timeout" | "memory")` and the next document starts a fresh child. The caller (worker)
never goes down with the parser.
"""

from __future__ import annotations

import functools
import multiprocessing as mp
import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Self

import psutil

from trifact.ingestion.parser import ParsedDoc, ParseError


def _serve(conn: Connection, factory: Callable[[], Any]) -> None:
    parser = factory()
    if hasattr(parser, "warmup"):
        parser.warmup()
    conn.send(("ready", None, None))
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if msg is None:
            return
        raw, filename = msg
        try:
            conn.send(("ok", parser.parse(raw, filename), None))
        except ParseError as e:
            conn.send(("err", e.reason, e.detail))
        except MemoryError:
            conn.send(("err", "memory", "MemoryError"))
        except RecursionError as e:
            conn.send(("err", "corrupted", f"RecursionError: {e}"))
        except Exception as e:  # noqa: BLE001 - any other parser failure is reported, the child keeps serving
            conn.send(("err", "corrupted", f"{type(e).__name__}: {e}"))


def parser_factory(cfg: dict[str, Any], root: Path, **kw: Any) -> Callable[[], Any]:
    from trifact.ingestion.engines import Parser

    return functools.partial(Parser, cfg, root, **kw)


class SafeParser:
    def __init__(
        self,
        factory: Callable[[], Any],
        timeout_s: float,
        max_rss_gb: float,
        start_timeout_s: float = 600.0,
        poll_s: float = 0.1,
    ):
        self.factory = factory
        self.timeout_s, self.max_rss = timeout_s, max_rss_gb * 2**30
        self.start_timeout_s, self.poll_s = start_timeout_s, poll_s
        self._proc: Any = None
        self._conn: Connection | None = None
        self.restarts = 0
        self._started = False

    @classmethod
    def from_params(cls, cfg: dict[str, Any], root: Path, **kw: Any) -> SafeParser:
        lim = cfg["limits"]
        return cls(parser_factory(cfg, root, **kw), lim["timeout_s"], lim["max_rss_gb"])

    def _start(self) -> None:
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, self.factory), daemon=True)
        self._proc.start()
        child.close()
        self._conn = parent
        if not parent.poll(self.start_timeout_s):
            self._kill()
            raise RuntimeError("parser process did not start")
        try:
            parent.recv()
        except EOFError as e:
            self._kill()
            raise RuntimeError("parser process died while starting") from e

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                for c in psutil.Process(self._proc.pid).children(recursive=True):
                    c.kill()
            except psutil.Error:
                pass
            self._proc.kill()
            self._proc.join(5)
        if self._conn is not None:
            self._conn.close()
        self._proc = self._conn = None

    def _rss(self) -> int:
        try:
            p = psutil.Process(self._proc.pid)
            return int(sum(x.memory_info().rss for x in [p, *p.children(recursive=True)]))
        except psutil.Error:
            return 0

    def parse(self, raw: bytes, filename: str) -> ParsedDoc:
        if self._proc is None or not self._proc.is_alive():
            self._kill()
            self.restarts += self._started
            self._start()
            self._started = True
        assert self._conn is not None
        try:
            self._conn.send((raw, filename))
        except (BrokenPipeError, OSError) as e:
            self._kill()
            raise ParseError("crash", f"parser process unavailable: {e}") from e
        t0 = time.monotonic()
        while True:
            try:
                ready = self._conn.poll(self.poll_s)
            except (EOFError, OSError):
                ready = True
            if ready:
                try:
                    kind, val, detail = self._conn.recv()
                except (EOFError, OSError) as e:
                    code = self._proc.exitcode if self._proc is not None else None
                    self._kill()
                    raise ParseError("crash", f"parser process exited ({code})") from e
                if kind == "ok":
                    return val  # type: ignore[no-any-return]
                raise ParseError(val, detail or "")
            if time.monotonic() - t0 > self.timeout_s:
                self._kill()
                raise ParseError("timeout", f"> {self.timeout_s} s")
            if self._rss() > self.max_rss:
                self._kill()
                raise ParseError("memory", f"> {self.max_rss / 2**30:.1f} GB")
            if not self._proc.is_alive():
                code = self._proc.exitcode
                self._kill()
                raise ParseError("crash", f"parser process exited ({code})")

    def close(self) -> None:
        if self._conn is not None and self._proc is not None and self._proc.is_alive():
            try:
                self._conn.send(None)
                self._proc.join(5)
            except OSError:
                pass
        self._kill()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
