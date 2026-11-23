"""Synchronize canonical `.mcp.json` into project-level Codex and Antigravity MCP configs.

Single source of truth:
  - `.mcp.json` (committed): canonical registry, natively loaded by Claude Code.
  - `.codex/config.toml` (gitignored): generated project-level Codex MCP config.
  - `.agents/mcp_config.json` (gitignored): generated workspace-level Antigravity MCP config.

Usage:
  uv run python scripts/sync_mcp.py sync [--targets codex,antigravity] [--dry-run]
  uv run python scripts/sync_mcp.py check [--targets codex,antigravity]
  uv run python scripts/sync_mcp.py audit [--probe]
  uv run python scripts/sync_mcp.py restore [--target codex|antigravity] [--list]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FILE = ".mcp.json"
CODEX_REL = Path(".codex") / "config.toml"
ANTIGRAVITY_REL = Path(".agents") / "mcp_config.json"
BACKUP_REL = Path(".agents") / ".mcp-backups"

SERVER_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
EXACT_ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
BEARER_ENV_RE = re.compile(r"^Bearer\s+\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
HARDCODED_HOME_RE = re.compile(r"(?:[A-Za-z]:[/\\]Users[/\\][^/\\\s]+|/(?:Users|home)/[^/\s]+)")
TOML_SECTION_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$")
TOML_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")

PROJECT_DIR_VARS = frozenset({"CLAUDE_PROJECT_DIR", "PROJECT_ROOT", "workspaceFolder"})
SECRET_KEY_HINTS = ("token", "secret", "key", "password", "passwd", "auth")
SERENA_CONTEXT_MAP = {
    "claude": "claude-code",
    "codex": "codex",
    "antigravity": "ide-assistant",
}
MANAGED_TOML_BANNER = "# BEGIN MANAGED MCP SERVERS (source: .mcp.json via scripts/sync_mcp.py)"
MANAGED_TOML_FOOTER = "# END MANAGED MCP SERVERS"


@dataclass
class Diagnostic:
    level: str  # "error" | "warning" | "info"
    server: str
    message: str
    target: str = "canonical"

    def format(self) -> str:
        prefix = f"[{self.level.upper()}]"
        scope = f"{self.target}:{self.server}" if self.server else self.target
        return f"{prefix} {scope}: {self.message}"


@dataclass
class RenderResult:
    content: str
    diagnostics: list[Diagnostic] = field(default_factory=list)


def detect_platform(override: str | None = None) -> str:
    if override in {"windows", "posix"}:
        return override
    return "windows" if os.name == "nt" else "posix"


def load_dotenv_file(path: Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines from `.env` without modifying `os.environ`."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_effective_env(repo_root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Combine `.env` defaults with active environment (shell env takes precedence)."""
    dotenv_vals = load_dotenv_file(repo_root / ".env")
    active = dict(os.environ) if base_env is None else dict(base_env)
    merged = {k: v for k, v in dotenv_vals.items() if v}
    merged.update({k: v for k, v in active.items() if v})
    return merged


def load_canonical_registry(mcp_path: Path) -> tuple[dict[str, dict[str, Any]], list[Diagnostic]]:
    """Load and validate `.mcp.json`."""
    diags: list[Diagnostic] = []
    if not mcp_path.is_file():
        diags.append(Diagnostic("error", "", f"Canonical registry not found: {mcp_path}"))
        return {}, diags
    try:
        raw = json.loads(mcp_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        diags.append(Diagnostic("error", "", f"Invalid JSON in {mcp_path.name}: {exc}"))
        return {}, diags
    if not isinstance(raw, dict) or not isinstance(raw.get("mcpServers"), dict):
        diags.append(Diagnostic("error", "", "Root object must contain an 'mcpServers' object"))
        return {}, diags

    servers: dict[str, dict[str, Any]] = {}
    seen_urls: dict[str, str] = {}
    seen_commands: dict[tuple[str, tuple[str, ...]], str] = {}

    for name, cfg in raw["mcpServers"].items():
        if not isinstance(name, str) or not SERVER_NAME_RE.fullmatch(name):
            diags.append(Diagnostic("error", str(name), "Server name must match ^[A-Za-z0-9._-]+$"))
            continue
        if not isinstance(cfg, dict):
            diags.append(Diagnostic("error", name, "Server configuration must be a JSON object"))
            continue

        transport = _infer_transport(name, cfg, diags)
        if transport is None:
            continue

        if transport == "http":
            _validate_http_server(name, cfg, seen_urls, diags)
        elif transport == "stdio":
            _validate_stdio_server(name, cfg, seen_commands, diags)

        if "timeout" in cfg and (not isinstance(cfg["timeout"], int) or cfg["timeout"] <= 0):
            diags.append(Diagnostic("error", name, "'timeout' must be a positive integer (ms)"))
        if "disabled" in cfg and not isinstance(cfg["disabled"], bool):
            diags.append(Diagnostic("error", name, "'disabled' must be a boolean"))

        servers[name] = cfg

    return servers, diags


def _infer_transport(name: str, cfg: dict[str, Any], diags: list[Diagnostic]) -> str | None:
    raw_type = cfg.get("type")
    has_url = "url" in cfg
    has_cmd = "command" in cfg

    if has_url and has_cmd:
        diags.append(Diagnostic("error", name, "Server cannot define both 'url' and 'command'"))
        return None
    if raw_type in {"sse", "ws"}:
        diags.append(
            Diagnostic(
                "error",
                name,
                f"Transport '{raw_type}' is not supported by Antigravity/Codex; "
                "use Streamable HTTP ('http') or a stdio bridge",
            )
        )
        return None
    if raw_type in {"http", "streamable-http"}:
        if not has_url:
            diags.append(Diagnostic("error", name, "HTTP server requires a 'url' field"))
            return None
        return "http"
    if raw_type == "stdio":
        if not has_cmd:
            diags.append(Diagnostic("error", name, "Stdio server requires a 'command' field"))
            return None
        return "stdio"
    if raw_type is not None:
        diags.append(Diagnostic("error", name, f"Unsupported transport type: {raw_type!r}"))
        return None
    if has_url:
        diags.append(
            Diagnostic(
                "error",
                name,
                "Remote server with 'url' must declare '\"type\": \"http\"' for Claude Code compatibility",
            )
        )
        return None
    if has_cmd:
        return "stdio"
    diags.append(Diagnostic("error", name, "Server must define either 'url' (http) or 'command' (stdio)"))
    return None


def _validate_http_server(
    name: str,
    cfg: dict[str, Any],
    seen_urls: dict[str, str],
    diags: list[Diagnostic],
) -> None:
    url = cfg.get("url")
    if not isinstance(url, str) or not url.strip():
        diags.append(Diagnostic("error", name, "'url' must be a non-empty string"))
        return
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        diags.append(Diagnostic("error", name, f"Invalid HTTP(S) URL: {url!r}"))
    norm_url = url.rstrip("/").lower()
    if norm_url in seen_urls:
        diags.append(
            Diagnostic("error", name, f"Duplicate remote URL also defined by '{seen_urls[norm_url]}'")
        )
    else:
        seen_urls[norm_url] = name

    headers = cfg.get("headers")
    if headers is not None:
        if not isinstance(headers, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
        ):
            diags.append(Diagnostic("error", name, "'headers' must be an object of string key/value pairs"))
        else:
            for hk, hv in headers.items():
                _check_secret_leak(name, f"headers.{hk}", hk, hv, diags)


def _validate_stdio_server(
    name: str,
    cfg: dict[str, Any],
    seen_commands: dict[tuple[str, tuple[str, ...]], str],
    diags: list[Diagnostic],
) -> None:
    cmd = cfg.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        diags.append(Diagnostic("error", name, "'command' must be a non-empty string"))
        return
    args = cfg.get("args", [])
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        diags.append(Diagnostic("error", name, "'args' must be an array of strings"))
        args = []

    sig = (cmd, tuple(args))
    if sig in seen_commands:
        diags.append(
            Diagnostic("error", name, f"Duplicate stdio command/args also defined by '{seen_commands[sig]}'")
        )
    else:
        seen_commands[sig] = name

    for idx, arg in enumerate(args):
        if HARDCODED_HOME_RE.search(arg):
            diags.append(
                Diagnostic(
                    "error",
                    name,
                    f"Hardcoded user directory in args[{idx}]; use '${{CLAUDE_PROJECT_DIR:-.}}' instead",
                )
            )
        _check_project_dir_fallback(name, f"args[{idx}]", arg, diags)

    cwd = cfg.get("cwd")
    if cwd is not None:
        if not isinstance(cwd, str) or not cwd.strip():
            diags.append(Diagnostic("error", name, "'cwd' must be a non-empty string"))
        elif HARDCODED_HOME_RE.search(cwd):
            diags.append(
                Diagnostic(
                    "error",
                    name,
                    "Hardcoded user directory in 'cwd'; use '${CLAUDE_PROJECT_DIR:-.}' instead",
                )
            )
        else:
            _check_project_dir_fallback(name, "cwd", cwd, diags)

    env = cfg.get("env")
    if env is not None:
        if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()
        ):
            diags.append(Diagnostic("error", name, "'env' must be an object of string key/value pairs"))
        else:
            for ek, ev in env.items():
                if HARDCODED_HOME_RE.search(ev):
                    diags.append(
                        Diagnostic(
                            "error",
                            name,
                            f"Hardcoded user directory in env.{ek}; use '${{CLAUDE_PROJECT_DIR:-.}}' instead",
                        )
                    )
                _check_project_dir_fallback(name, f"env.{ek}", ev, diags)
                _check_secret_leak(name, f"env.{ek}", ek, ev, diags)


def _check_project_dir_fallback(server: str, field_path: str, value: str, diags: list[Diagnostic]) -> None:
    """Require `${CLAUDE_PROJECT_DIR:-.}` fallback because Claude Code injects it into spawned processes,
    not always into `.mcp.json` interpolation."""
    for match in ENV_REF_RE.finditer(value):
        if match.group(1) == "CLAUDE_PROJECT_DIR" and match.group(2) is None:
            diags.append(
                Diagnostic(
                    "error",
                    server,
                    f"Bare '${{CLAUDE_PROJECT_DIR}}' in {field_path} is not guaranteed during Claude Code "
                    "config interpolation; use '${CLAUDE_PROJECT_DIR:-.}' instead",
                )
            )


def _check_secret_leak(server: str, field_path: str, key: str, value: str, diags: list[Diagnostic]) -> None:
    if ENV_REF_RE.search(value):
        return
    lowered = key.lower()
    if any(hint in lowered for hint in SECRET_KEY_HINTS) and value.strip():
        diags.append(
            Diagnostic(
                "error",
                server,
                f"Potential plaintext credential in {field_path}; reference an env var via '${{VAR}}' instead",
            )
        )


def _uses_project_dir_placeholder(cfg: dict[str, Any]) -> bool:
    texts: list[str] = []
    if isinstance(cfg.get("cwd"), str):
        texts.append(cfg["cwd"])
    texts.extend(a for a in cfg.get("args", []) if isinstance(a, str))
    if isinstance(cfg.get("env"), dict):
        texts.extend(v for v in cfg["env"].values() if isinstance(v, str))
    return any(m.group(1) in PROJECT_DIR_VARS for text in texts for m in ENV_REF_RE.finditer(text))


def _unwrap_cmd_env_wrapper(command: str, args: list[str]) -> tuple[str, list[str]]:
    """Strip Claude Code Windows `cmd /c ... && npx ...` DATA_DIR shell wrapper when syncing to
    clients (Codex/Antigravity) that receive explicit resolved `env` and `cwd`."""
    if (
        command.lower() in {"cmd", "cmd.exe"}
        and len(args) == 2
        and args[0].lower() == "/c"
        and "&&" in args[1]
        and "CLAUDE_PROJECT_DIR" in args[1]
    ):
        tail = args[1].rsplit("&&", 1)[1].strip()
        parts = shlex.split(tail, posix=False)
        if parts:
            return "cmd", ["/c", *parts]
    return command, list(args)


def _normalize_command_for_platform(
    command: str, args: list[str], target_platform: str
) -> tuple[str, list[str]]:
    command, args = _unwrap_cmd_env_wrapper(command, args)
    if target_platform == "windows":
        if command.lower() == "npx":
            return "cmd", ["/c", "npx", *args]
        return command, list(args)
    # posix
    if command.lower() in {"cmd", "cmd.exe"} and len(args) >= 2 and args[0].lower() == "/c":
        return args[1], list(args[2:])
    return command, list(args)


def _adapt_serena_args(server: str, args: list[str], client: str) -> list[str]:
    is_serena = server.lower() == "serena" or any("serena" in a.lower() for a in args)
    if not is_serena:
        return list(args)
    target_ctx = SERENA_CONTEXT_MAP[client]
    adapted: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--context" and i + 1 < len(args):
            adapted.extend(["--context", target_ctx])
            i += 2
            continue
        if arg.startswith("--context="):
            adapted.append(f"--context={target_ctx}")
            i += 1
            continue
        adapted.append(arg)
        i += 1
    return adapted


def _resolve_template(
    value: str,
    repo_root: Path,
    env: dict[str, str],
    *,
    allow_env_values: bool = True,
) -> tuple[str | None, list[str]]:
    """Expand project-dir vars and `${VAR}` / `${VAR:-default}` references.

    Returns `(resolved_value_or_None_if_missing, list_of_missing_var_names)`.
    """
    missing: list[str] = []
    root_posix = repo_root.resolve().as_posix()

    def repl(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_val = match.group(2)
        if var_name in PROJECT_DIR_VARS:
            return root_posix
        if allow_env_values and var_name in env and env[var_name]:
            return env[var_name]
        if default_val is not None:
            return default_val
        missing.append(var_name)
        return match.group(0)

    resolved = ENV_REF_RE.sub(repl, value)
    if missing:
        return None, missing
    return resolved, []


# ---------------------------------------------------------------------------
# TOML formatting & merge-safe preservation for Codex (.codex/config.toml)
# ---------------------------------------------------------------------------


def _toml_key(key: str) -> str:
    if TOML_BARE_KEY_RE.fullmatch(key):
        return key
    return json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        if not value:
            return "[]"
        if len(value) <= 3 and sum(len(str(v)) for v in value) < 60:
            return "[" + ", ".join(_toml_value(v) for v in value) + "]"
        inner = "\n".join(f"    {_toml_value(v)}," for v in value)
        return f"[\n{inner}\n]"
    raise TypeError(f"Unsupported TOML value type: {type(value).__name__}")


def _strip_existing_mcp_toml_sections(existing: str) -> str:
    """Preserve all non-`mcp_servers` content in `.codex/config.toml` verbatim."""
    kept: list[str] = []
    skip_section = False
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped in {MANAGED_TOML_BANNER, MANAGED_TOML_FOOTER}:
            continue
        match = TOML_SECTION_RE.match(line)
        if match:
            header = match.group(1).strip()
            parts = [p.strip().strip('"').strip("'") for p in header.split(".")]
            if parts and parts[0] == "mcp_servers":
                skip_section = True
                continue
            skip_section = False
        if not skip_section:
            kept.append(line)
    # Trim trailing blank lines cleanly
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


def render_codex_config(
    servers: dict[str, dict[str, Any]],
    repo_root: Path,
    env: dict[str, str],
    *,
    existing_text: str = "",
    target_platform: str | None = None,
) -> RenderResult:
    plat = detect_platform(target_platform)
    diags: list[Diagnostic] = []
    blocks: list[str] = [MANAGED_TOML_BANNER]

    for name in sorted(servers):
        cfg = servers[name]
        table_key = f"mcp_servers.{_toml_key(name)}"
        lines: list[str] = [f"[{table_key}]"]

        if "url" in cfg:
            url_val, missing = _resolve_template(cfg["url"], repo_root, env)
            if url_val is None:
                diags.append(
                    Diagnostic(
                        "error",
                        name,
                        f"Missing env var(s) {', '.join(missing)} required in 'url'",
                        target="codex",
                    )
                )
                continue
            lines.append(f"url = {_toml_value(url_val)}")
            static_headers: dict[str, str] = {}
            env_headers: dict[str, str] = {}
            for hk, hv in sorted(cfg.get("headers", {}).items()):
                bearer_m = BEARER_ENV_RE.fullmatch(hv.strip())
                if hk.lower() == "authorization" and bearer_m:
                    var_name = bearer_m.group(1)
                    lines.append(f"bearer_token_env_var = {_toml_value(var_name)}")
                    if var_name not in env:
                        diags.append(
                            Diagnostic(
                                "warning",
                                name,
                                f"Env var '{var_name}' is not set; Codex will require it at runtime "
                                "or OAuth login",
                                target="codex",
                            )
                        )
                    continue
                exact_m = EXACT_ENV_REF_RE.fullmatch(hv.strip())
                if exact_m:
                    var_name = exact_m.group(1)
                    env_headers[hk] = var_name
                    if var_name not in env:
                        diags.append(
                            Diagnostic(
                                "warning",
                                name,
                                f"Env var '{var_name}' for header '{hk}' is not set",
                                target="codex",
                            )
                        )
                    continue
                if ENV_REF_RE.search(hv):
                    diags.append(
                        Diagnostic(
                            "warning",
                            name,
                            f"Header '{hk}' uses a composite env template not supported by Codex "
                            "config.toml; skipped",
                            target="codex",
                        )
                    )
                    continue
                static_headers[hk] = hv

            if cfg.get("disabled") is True:
                lines.append("enabled = false")
            if "timeout" in cfg:
                lines.append(f"tool_timeout_sec = {max(1, round(cfg['timeout'] / 1000))}")
            if static_headers:
                lines.append("")
                lines.append(f"[{table_key}.http_headers]")
                for hk, hv in static_headers.items():
                    lines.append(f"{_toml_key(hk)} = {_toml_value(hv)}")
            if env_headers:
                lines.append("")
                lines.append(f"[{table_key}.env_http_headers]")
                for hk, var_name in env_headers.items():
                    lines.append(f"{_toml_key(hk)} = {_toml_value(var_name)}")
        else:
            cmd, args = _normalize_command_for_platform(cfg["command"], cfg.get("args", []), plat)
            args = _adapt_serena_args(name, args, "codex")
            resolved_args: list[str] = []
            arg_error = False
            for arg in args:
                val, missing = _resolve_template(arg, repo_root, env)
                if val is None:
                    diags.append(
                        Diagnostic(
                            "warning",
                            name,
                            f"Unset env var(s) {', '.join(missing)} in args; skipping server for Codex",
                            target="codex",
                        )
                    )
                    arg_error = True
                    break
                resolved_args.append(val)
            if arg_error:
                continue

            lines.append(f"command = {_toml_value(cmd)}")
            if resolved_args:
                lines.append(f"args = {_toml_value(resolved_args)}")

            cwd_raw = cfg.get("cwd")
            if isinstance(cwd_raw, str):
                cwd_val, _ = _resolve_template(cwd_raw, repo_root, env)
                if cwd_val:
                    lines.append(f"cwd = {_toml_value(cwd_val)}")
            elif _uses_project_dir_placeholder(cfg):
                lines.append(f"cwd = {_toml_value(repo_root.resolve().as_posix())}")

            if cfg.get("disabled") is True:
                lines.append("enabled = false")
            if "timeout" in cfg:
                lines.append(f"tool_timeout_sec = {max(1, round(cfg['timeout'] / 1000))}")

            static_env: dict[str, str] = {}
            forwarded_env_vars: list[str] = []
            for ek, ev in sorted(cfg.get("env", {}).items()):
                exact_m = EXACT_ENV_REF_RE.fullmatch(ev.strip())
                if exact_m and exact_m.group(1) not in PROJECT_DIR_VARS:
                    var_name = exact_m.group(1)
                    if var_name == ek:
                        forwarded_env_vars.append(var_name)
                        if var_name not in env:
                            diags.append(
                                Diagnostic(
                                    "warning",
                                    name,
                                    f"Env var '{var_name}' is not set; Codex will forward it when present",
                                    target="codex",
                                )
                            )
                    elif var_name in env:
                        static_env[ek] = env[var_name]
                    else:
                        diags.append(
                            Diagnostic(
                                "warning",
                                name,
                                f"Env var '{var_name}' for '{ek}' is not set; omitted for Codex",
                                target="codex",
                            )
                        )
                    continue

                resolved_ev, missing = _resolve_template(ev, repo_root, env)
                if resolved_ev is None:
                    diags.append(
                        Diagnostic(
                            "warning",
                            name,
                            f"Env var(s) {', '.join(missing)} for '{ek}' not set; omitted for Codex",
                            target="codex",
                        )
                    )
                    continue
                static_env[ek] = resolved_ev

            if forwarded_env_vars:
                lines.append(f"env_vars = {_toml_value(forwarded_env_vars)}")
            if static_env:
                lines.append("")
                lines.append(f"[{table_key}.env]")
                for ek, ev in static_env.items():
                    lines.append(f"{_toml_key(ek)} = {_toml_value(ev)}")

        blocks.append("\n".join(lines))

    blocks.append(MANAGED_TOML_FOOTER)
    managed_text = "\n\n".join(blocks) + "\n"

    preserved = _strip_existing_mcp_toml_sections(existing_text)
    full_text = f"{preserved}\n\n{managed_text}" if preserved else managed_text

    # Guarantee generated TOML is syntactically valid.
    tomllib.loads(full_text)
    return RenderResult(content=full_text, diagnostics=diags)


# ---------------------------------------------------------------------------
# JSON renderer for Google Antigravity (.agents/mcp_config.json)
# ---------------------------------------------------------------------------


def render_antigravity_config(
    servers: dict[str, dict[str, Any]],
    repo_root: Path,
    env: dict[str, str],
    *,
    existing_text: str = "",
    target_platform: str | None = None,
    resolve_secrets: bool = True,
) -> RenderResult:
    plat = detect_platform(target_platform)
    diags: list[Diagnostic] = []

    base_obj: dict[str, Any] = {}
    if existing_text.strip():
        try:
            parsed = json.loads(existing_text)
            if isinstance(parsed, dict):
                base_obj = {k: v for k, v in parsed.items() if k != "mcpServers"}
        except json.JSONDecodeError:
            diags.append(
                Diagnostic(
                    "warning",
                    "",
                    "Existing .agents/mcp_config.json was invalid JSON and will be replaced (backed up first)",
                    target="antigravity",
                )
            )

    rendered_servers: dict[str, dict[str, Any]] = {}
    for name in sorted(servers):
        cfg = servers[name]
        entry: dict[str, Any] = {}

        if "url" in cfg:
            url_val, missing = _resolve_template(cfg["url"], repo_root, env)
            if url_val is None:
                diags.append(
                    Diagnostic(
                        "error",
                        name,
                        f"Missing env var(s) {', '.join(missing)} required in 'url'",
                        target="antigravity",
                    )
                )
                continue
            entry["serverUrl"] = url_val

            headers_out: dict[str, str] = {}
            for hk, hv in sorted(cfg.get("headers", {}).items()):
                if ENV_REF_RE.search(hv):
                    resolved_hv, missing = _resolve_template(
                        hv, repo_root, env, allow_env_values=resolve_secrets
                    )
                    if resolved_hv is None:
                        diags.append(
                            Diagnostic(
                                "warning",
                                name,
                                f"Env var(s) {', '.join(missing)} not set for header '{hk}'; "
                                "header omitted for Antigravity (Antigravity does not expand ${VAR}; "
                                "authenticate via OAuth or set the env var and re-run sync)",
                                target="antigravity",
                            )
                        )
                        continue
                    headers_out[hk] = resolved_hv
                else:
                    headers_out[hk] = hv
            if headers_out:
                entry["headers"] = headers_out
        else:
            cmd, args = _normalize_command_for_platform(cfg["command"], cfg.get("args", []), plat)
            args = _adapt_serena_args(name, args, "antigravity")
            resolved_args: list[str] = []
            arg_error = False
            for arg in args:
                val, missing = _resolve_template(arg, repo_root, env)
                if val is None:
                    diags.append(
                        Diagnostic(
                            "warning",
                            name,
                            f"Unset env var(s) {', '.join(missing)} in args; skipping server for Antigravity",
                            target="antigravity",
                        )
                    )
                    arg_error = True
                    break
                resolved_args.append(val)
            if arg_error:
                continue

            entry["command"] = cmd
            if resolved_args:
                entry["args"] = resolved_args

            cwd_raw = cfg.get("cwd")
            if isinstance(cwd_raw, str):
                cwd_val, _ = _resolve_template(cwd_raw, repo_root, env)
                if cwd_val:
                    entry["cwd"] = cwd_val
            elif _uses_project_dir_placeholder(cfg):
                entry["cwd"] = repo_root.resolve().as_posix()

            env_out: dict[str, str] = {}
            for ek, ev in sorted(cfg.get("env", {}).items()):
                resolved_ev, missing = _resolve_template(ev, repo_root, env, allow_env_values=resolve_secrets)
                if resolved_ev is None:
                    diags.append(
                        Diagnostic(
                            "warning",
                            name,
                            f"Env var(s) {', '.join(missing)} not set for env.{ek}; omitted for Antigravity",
                            target="antigravity",
                        )
                    )
                    continue
                env_out[ek] = resolved_ev
            if env_out:
                entry["env"] = env_out

        if cfg.get("disabled") is True:
            entry["disabled"] = True
        if "timeout" in cfg:
            entry["timeoutSeconds"] = max(1, round(cfg["timeout"] / 1000))

        rendered_servers[name] = entry

    out_obj = {**base_obj, "mcpServers": rendered_servers}
    content = json.dumps(out_obj, indent=2) + "\n"
    return RenderResult(content=content, diagnostics=diags)


# ---------------------------------------------------------------------------
# Sync, Check, Audit, Backup & Restore operations
# ---------------------------------------------------------------------------


def _target_path(repo_root: Path, target: str) -> Path:
    if target == "codex":
        return repo_root / CODEX_REL
    if target == "antigravity":
        return repo_root / ANTIGRAVITY_REL
    raise ValueError(f"Unknown target: {target}")


def _backup_file(repo_root: Path, target: str, path: Path) -> Path:
    backup_dir = repo_root / BACKUP_REL
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = backup_dir / f"{stamp}_{target}_{path.name}"
    shutil.copy2(path, dest)
    return dest


def sync_all(
    repo_root: Path,
    *,
    targets: list[str] | None = None,
    dry_run: bool = False,
    check_only: bool = False,
    backup: bool = True,
    target_platform: str | None = None,
    base_env: dict[str, str] | None = None,
) -> tuple[int, list[str], list[Diagnostic]]:
    selected = targets or ["codex", "antigravity"]
    mcp_path = repo_root / CANONICAL_FILE
    servers, diags = load_canonical_registry(mcp_path)
    errors = [d for d in diags if d.level == "error"]
    if errors:
        return 1, [], diags

    env = load_effective_env(repo_root, base_env=base_env)
    messages: list[str] = []
    drifted = False

    for target in selected:
        dest = _target_path(repo_root, target)
        existing = dest.read_text(encoding="utf-8") if dest.is_file() else ""
        if target == "codex":
            res = render_codex_config(
                servers, repo_root, env, existing_text=existing, target_platform=target_platform
            )
        elif target == "antigravity":
            res = render_antigravity_config(
                servers, repo_root, env, existing_text=existing, target_platform=target_platform
            )
        else:
            diags.append(Diagnostic("error", "", f"Unsupported sync target: {target}"))
            return 1, messages, diags

        diags.extend(res.diagnostics)
        if any(d.level == "error" for d in res.diagnostics):
            return 1, messages, diags

        if existing == res.content:
            messages.append(f"{target}: up to date ({dest.relative_to(repo_root).as_posix()})")
            continue

        drifted = True
        rel = dest.relative_to(repo_root).as_posix()
        if check_only:
            messages.append(f"{target}: OUT OF SYNC ({rel}) — run `uv run python scripts/sync_mcp.py sync`")
            continue
        if dry_run:
            messages.append(f"{target}: would update {rel} ({len(servers)} servers)")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        backup_note = ""
        if backup and dest.is_file():
            bpath = _backup_file(repo_root, target, dest)
            backup_note = f" (backup: {bpath.relative_to(repo_root).as_posix()})"
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(res.content, encoding="utf-8", newline="\n")
        tmp.replace(dest)
        messages.append(f"{target}: synced {len(servers)} servers -> {rel}{backup_note}")

    if check_only and drifted:
        return 1, messages, diags
    return 0, messages, diags


def _parse_mcp_http_jsonrpc(body: str, content_type: str) -> dict[str, Any] | None:
    """Parse a JSON-RPC message from either `application/json` or `text/event-stream`."""
    if "text/event-stream" in content_type.lower():
        for line in body.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if not payload:
                    continue
                try:
                    obj = json.loads(payload)
                    if isinstance(obj, dict) and ("result" in obj or "error" in obj):
                        return obj
                except json.JSONDecodeError:
                    continue
        return None
    try:
        obj = json.loads(body)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _probe_http_mcp(url: str, headers: dict[str, str], timeout: int = 12) -> str:
    """Run MCP initialize + notifications/initialized + tools/list over Streamable HTTP."""
    base_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "trifact-sync-mcp/0.1",
        **headers,
    }
    init_body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "trifact-sync-mcp", "version": "0.1.0"},
            },
        }
    ).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=init_body, method="POST", headers=base_headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            session_id = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            raw_body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            www_auth = exc.headers.get("WWW-Authenticate", "")
            oauth_hint = " (OAuth/Bearer)" if "resource_metadata" in www_auth else ""
            return f"auth_required: HTTP {exc.code}{oauth_hint}"
        return f"http_error: HTTP {exc.code}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"external_service_unavailable: {type(exc).__name__}"

    init_msg = _parse_mcp_http_jsonrpc(raw_body, ctype)
    if not init_msg or "result" not in init_msg:
        return "http_200_non_mcp: initialize response was not valid MCP JSON-RPC"

    server_info = init_msg["result"].get("serverInfo", {})
    sname = server_info.get("name", "unknown")
    sver = server_info.get("version", "")
    init_label = f"mcp_initialized({sname} {sver}".strip() + ")"

    session_headers = dict(base_headers)
    if session_id:
        session_headers["Mcp-Session-Id"] = session_id

    # Send notifications/initialized (best-effort)
    try:
        notif_body = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode("utf-8")
        urllib.request.urlopen(
            urllib.request.Request(url, data=notif_body, method="POST", headers=session_headers),
            timeout=timeout,
        ).close()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass

    # Query tools/list to verify real tool discovery
    tools_body = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}).encode("utf-8")
    try:
        t_req = urllib.request.Request(url, data=tools_body, method="POST", headers=session_headers)
        with urllib.request.urlopen(t_req, timeout=timeout) as t_resp:
            t_ctype = t_resp.headers.get("Content-Type", "")
            t_raw = t_resp.read().decode("utf-8", errors="replace")
        t_msg = _parse_mcp_http_jsonrpc(t_raw, t_ctype)
        if t_msg and isinstance(t_msg.get("result", {}).get("tools"), list):
            n_tools = len(t_msg["result"]["tools"])
            return f"{init_label}, tools_discovered({n_tools})"
        return f"{init_label}, tools_list_failed"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"{init_label}, tools_list_error({type(exc).__name__})"


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        proc.kill()


def _probe_stdio_mcp(cmd: list[str], cwd: Path, extra_env: dict[str, str], timeout: float = 20.0) -> str:
    """Run MCP initialize + notifications/initialized + tools/list over stdio."""
    env = os.environ.copy()
    env.update(extra_env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdin and proc.stdout
    incoming: list[dict[str, Any]] = []
    lock = threading.Lock()

    def reader() -> None:
        assert proc.stdout
        for line in proc.stdout:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    with lock:
                        incoming.append(obj)
            except json.JSONDecodeError:
                continue

    threading.Thread(target=reader, daemon=True).start()
    root_uri = cwd.resolve().as_uri()
    try:
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"roots": {"listChanged": False}},
                        "clientInfo": {"name": "trifact-sync-mcp", "version": "0.1.0"},
                    },
                }
            )
            + "\n"
        )
        proc.stdin.flush()

        deadline = time.monotonic() + timeout
        init_label: str | None = None
        answered_ids: set[Any] = set()
        while time.monotonic() < deadline:
            with lock:
                batch = list(incoming)
                incoming.clear()
            for msg in batch:
                # Handle server->client roots/list requests (e.g. Shrimp)
                if msg.get("method") == "roots/list" and "id" in msg and msg["id"] not in answered_ids:
                    answered_ids.add(msg["id"])
                    proc.stdin.write(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": msg["id"],
                                "result": {"roots": [{"uri": root_uri, "name": cwd.name}]},
                            }
                        )
                        + "\n"
                    )
                    proc.stdin.flush()
                elif msg.get("id") == 1 and "result" in msg and init_label is None:
                    info = msg["result"].get("serverInfo", {})
                    init_label = (
                        f"mcp_initialized({info.get('name', 'unknown')} {info.get('version', '')}".strip()
                        + ")"
                    )
                    proc.stdin.write(
                        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
                    )
                    proc.stdin.write(
                        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n"
                    )
                    proc.stdin.flush()
                elif msg.get("id") == 2 and isinstance(msg.get("result", {}).get("tools"), list):
                    n_tools = len(msg["result"]["tools"])
                    return f"{init_label or 'mcp_initialized'}, tools_discovered({n_tools})"
            if proc.poll() is not None:
                return f"stdio_exited(rc={proc.returncode})"
            time.sleep(0.1)
        if init_label:
            return f"{init_label}, tools_list_timeout"
        return "stdio_initialize_timeout"
    finally:
        _kill_process_tree(proc)


def _probe_local_http_dep(url: str, timeout: int = 3) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def audit_registry(
    repo_root: Path, *, probe: bool = False, probe_stdio: bool = False
) -> tuple[int, list[str]]:
    mcp_path = repo_root / CANONICAL_FILE
    servers, diags = load_canonical_registry(mcp_path)
    lines = [d.format() for d in diags]
    if any(d.level == "error" for d in diags):
        return 1, lines

    env = load_effective_env(repo_root)
    plat = detect_platform()

    for name in sorted(servers):
        cfg = servers[name]
        if "url" in cfg:
            url = cfg["url"]
            missing_vars: list[str] = []
            resolved_headers: dict[str, str] = {}
            for hk, hv in cfg.get("headers", {}).items():
                val, missing = _resolve_template(hv, repo_root, env)
                if val is None:
                    missing_vars.extend(missing)
                else:
                    resolved_headers[hk] = val
            tags = ["config_valid", "remote-http"]
            if missing_vars:
                tags.append(f"missing_env({', '.join(missing_vars)})")
            if probe:
                tags.append(_probe_http_mcp(url, resolved_headers))
            lines.append(f"  {name:14s} {' | '.join(tags)} -> {url}")
        else:
            cmd_norm, args_norm = _normalize_command_for_platform(cfg["command"], cfg.get("args", []), plat)
            found = shutil.which(cmd_norm)
            tags = ["config_valid", f"stdio({cmd_norm}={'OK' if found else 'MISSING'})"]
            resolved_env: dict[str, str] = {}
            for ek, ev in cfg.get("env", {}).items():
                val, _ = _resolve_template(ev, repo_root, env)
                if val is not None:
                    resolved_env[ek] = val
                    if ek.endswith("_URI") and val.startswith(("http://", "https://")) and probe:
                        up = _probe_local_http_dep(val)
                        tags.append(f"service({ek}={'UP' if up else 'external_service_unavailable'})")
            if probe_stdio and found:
                resolved_args = [_resolve_template(a, repo_root, env)[0] or a for a in args_norm]
                tags.append(_probe_stdio_mcp([cmd_norm, *resolved_args], repo_root, resolved_env))
            lines.append(f"  {name:14s} {' | '.join(tags)}")
    return 0, lines


def restore_backup(repo_root: Path, target: str | None, list_only: bool) -> tuple[int, list[str]]:
    backup_dir = repo_root / BACKUP_REL
    if not backup_dir.is_dir():
        return 0, ["No MCP backups found."]
    pattern = f"*_{target}_*" if target else "*"
    files = sorted(backup_dir.glob(pattern))
    if not files:
        return 0, [f"No MCP backups found for target={target or 'all'}."]
    if list_only:
        return 0, [f.relative_to(repo_root).as_posix() for f in files]
    if target is None:
        return 1, ["Specify --target codex|antigravity to restore the latest backup."]
    latest = files[-1]
    dest = _target_path(repo_root, target)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, dest)
    return 0, [f"Restored {dest.relative_to(repo_root).as_posix()} from {latest.name}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified MCP configuration sync, check, audit, and restore tool."
    )
    sub = parser.add_subparsers(dest="action")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--targets",
            default="codex,antigravity",
            help="Comma-separated targets to sync/check (codex,antigravity)",
        )
        p.add_argument("--platform", choices=["windows", "posix"], default=None)
        p.add_argument("--root", type=Path, default=ROOT)

    p_sync = sub.add_parser("sync", help="Validate .mcp.json and sync generated client configs")
    add_common(p_sync)
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.add_argument("--no-backup", action="store_true")

    p_check = sub.add_parser("check", help="Validate .mcp.json and verify generated configs are in sync")
    add_common(p_check)

    p_audit = sub.add_parser("audit", help="Audit server definitions, local runtimes, and env vars")
    p_audit.add_argument(
        "--probe",
        action="store_true",
        help="Run MCP initialize + tools/list on remote HTTP servers and check local service URIs",
    )
    p_audit.add_argument(
        "--probe-stdio",
        action="store_true",
        help="Also run MCP initialize + tools/list on local stdio servers",
    )
    p_audit.add_argument("--root", type=Path, default=ROOT)

    p_restore = sub.add_parser("restore", help="List or restore backed up generated MCP configs")
    p_restore.add_argument("--target", choices=["codex", "antigravity"], default=None)
    p_restore.add_argument("--list", action="store_true", dest="list_only")
    p_restore.add_argument("--root", type=Path, default=ROOT)

    args = parser.parse_args(argv)
    action = args.action or "sync"

    if action == "audit":
        code, lines = audit_registry(args.root, probe=args.probe, probe_stdio=args.probe_stdio)
        for line in lines:
            print(line)
        return code

    if action == "restore":
        code, lines = restore_backup(args.root, args.target, args.list_only)
        for line in lines:
            print(line)
        return code

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    code, messages, diags = sync_all(
        args.root,
        targets=targets,
        dry_run=getattr(args, "dry_run", False),
        check_only=(action == "check"),
        backup=not getattr(args, "no_backup", False),
        target_platform=args.platform,
    )
    for msg in messages:
        print(msg)
    for diag in diags:
        print(diag.format(), file=sys.stderr if diag.level == "error" else sys.stdout)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
