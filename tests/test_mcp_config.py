"""Tests for canonical `.mcp.json` validation and multi-client MCP synchronization."""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sync_mcp


def _write_mcp(root: Path, servers: dict[str, object]) -> Path:
    path = root / ".mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n", encoding="utf-8")
    return path


def test_repo_canonical_mcp_json_is_valid_and_portable() -> None:
    mcp_path = ROOT / ".mcp.json"
    assert b"\r\n" not in mcp_path.read_bytes(), ".mcp.json must use LF line endings"
    servers, diags = sync_mcp.load_canonical_registry(mcp_path)
    errors = [d.format() for d in diags if d.level == "error"]
    assert errors == [], f"Canonical .mcp.json validation errors: {errors}"
    expected = {
        "github",
        "shrimp",
        "structurizr",
        "context7",
        "exa",
        "huggingface",
        "mlflow",
        "playwright",
        "serena",
    }
    assert set(servers) == expected


def test_validation_rejects_unsupported_transports_and_missing_http_type(tmp_path: Path) -> None:
    mcp_path = _write_mcp(
        tmp_path,
        {
            "legacy_sse": {"type": "sse", "url": "https://example.com/sse"},
            "missing_type": {"url": "https://example.com/mcp"},
            "both": {"type": "http", "url": "https://a.example/mcp", "command": "node"},
        },
    )
    _, diags = sync_mcp.load_canonical_registry(mcp_path)
    errors = {(d.server, d.message) for d in diags if d.level == "error"}
    assert any(s == "legacy_sse" and "Streamable HTTP" in m for s, m in errors)
    assert any(s == "missing_type" and '"type": "http"' in m for s, m in errors)
    assert any(s == "both" and "both" in m for s, m in errors)


def test_validation_detects_duplicates_hardcoded_paths_and_plaintext_secrets(tmp_path: Path) -> None:
    mcp_path = _write_mcp(
        tmp_path,
        {
            "a": {"type": "http", "url": "https://mcp.example.com/mcp/"},
            "b": {"type": "http", "url": "https://mcp.example.com/mcp"},
            "c": {
                "type": "stdio",
                "command": "uvx",
                "args": ["tool", "--project", "C:/Users/alice/repo"],
                "env": {"API_KEY": "sk-plaintext-leak"},
            },
            "d": {
                "type": "stdio",
                "command": "uvx",
                "args": ["tool", "--project", "C:/Users/alice/repo"],
            },
            "e": {
                "type": "http",
                "url": "https://auth.example.com/mcp",
                "headers": {"Authorization": "Bearer ghp_plaintext123"},
            },
            "f": {
                "type": "stdio",
                "command": "uvx",
                "args": ["tool", "--project", "${CLAUDE_PROJECT_DIR}"],
            },
        },
    )
    _, diags = sync_mcp.load_canonical_registry(mcp_path)
    errors = [(d.server, d.message) for d in diags if d.level == "error"]
    assert any(s == "b" and "Duplicate remote URL" in m for s, m in errors)
    assert any(s == "d" and "Duplicate stdio" in m for s, m in errors)
    assert any(s == "c" and "Hardcoded user directory" in m for s, m in errors)
    assert any(s == "c" and "plaintext credential" in m for s, m in errors)
    assert any(s == "e" and "plaintext credential" in m for s, m in errors)
    assert any(s == "f" and "CLAUDE_PROJECT_DIR:-." in m for s, m in errors)


def test_codex_translation_auth_serena_platform_and_preserve_unrelated(tmp_path: Path) -> None:
    servers = {
        "github": {
            "type": "http",
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {
                "Authorization": "Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}",
                "X-Org-Key": "${ORG_KEY}",
                "X-Static": "v1",
            },
            "timeout": 120000,
        },
        "serena": {
            "type": "stdio",
            "command": "uvx",
            "args": [
                "--from",
                "git+https://github.com/oraios/serena",
                "serena",
                "start-mcp-server",
                "--context",
                "claude-code",
                "--project",
                "${CLAUDE_PROJECT_DIR}",
            ],
        },
        "playwright": {
            "type": "stdio",
            "command": "cmd",
            "args": ["/c", "npx", "-y", "@playwright/mcp@latest"],
            "disabled": True,
        },
        "shrimp": {
            "type": "stdio",
            "command": "cmd",
            "args": [
                "/c",
                (
                    'if defined CLAUDE_PROJECT_DIR (set "DATA_DIR=%CLAUDE_PROJECT_DIR%\\.shrimp") '
                    'else (set "DATA_DIR=%CD%\\.shrimp") && npx -y mcp-shrimp-task-manager'
                ),
            ],
            "env": {"DATA_DIR": "${CLAUDE_PROJECT_DIR:-.}/.shrimp"},
        },
        "local": {
            "type": "stdio",
            "command": "npx",
            "args": ["-y", "example-mcp"],
            "env": {
                "DATA_DIR": "${CLAUDE_PROJECT_DIR:-.}/.data",
                "MY_TOKEN": "${MY_TOKEN}",
                "URI": "${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}",
            },
        },
    }
    existing = (
        'model = "gpt-5.4"\n'
        "\n"
        "[profiles.fast]\n"
        'approval_policy = "on-request"\n'
        "\n"
        "[mcp_servers.stale_old]\n"
        'url = "https://old.example/mcp"\n'
    )
    res_win = sync_mcp.render_codex_config(
        servers, tmp_path, env={}, existing_text=existing, target_platform="windows"
    )
    parsed = tomllib.loads(res_win.content)
    assert parsed["model"] == "gpt-5.4"
    assert parsed["profiles"]["fast"]["approval_policy"] == "on-request"
    assert "stale_old" not in parsed["mcp_servers"]

    gh = parsed["mcp_servers"]["github"]
    assert gh["url"] == "https://api.githubcopilot.com/mcp/"
    assert gh["bearer_token_env_var"] == "GITHUB_PERSONAL_ACCESS_TOKEN"
    assert gh["env_http_headers"] == {"X-Org-Key": "ORG_KEY"}
    assert gh["http_headers"] == {"X-Static": "v1"}
    assert gh["tool_timeout_sec"] == 120

    serena = parsed["mcp_servers"]["serena"]
    assert "--context" in serena["args"]
    assert serena["args"][serena["args"].index("--context") + 1] == "codex"
    assert serena["args"][serena["args"].index("--project") + 1] == tmp_path.resolve().as_posix()
    assert serena["cwd"] == tmp_path.resolve().as_posix()

    pw = parsed["mcp_servers"]["playwright"]
    assert pw["command"] == "cmd"
    assert pw["args"][:2] == ["/c", "npx"]
    assert pw["enabled"] is False

    local = parsed["mcp_servers"]["local"]
    assert local["command"] == "cmd"
    assert local["args"] == ["/c", "npx", "-y", "example-mcp"]
    assert local["env_vars"] == ["MY_TOKEN"]
    assert local["env"]["DATA_DIR"] == f"{tmp_path.resolve().as_posix()}/.data"
    assert local["env"]["URI"] == "http://127.0.0.1:5000"

    # POSIX unwraps cmd /c npx -> npx (including shrimp wrapper)
    res_posix = sync_mcp.render_codex_config(
        servers, tmp_path, env={}, existing_text="", target_platform="posix"
    )
    parsed_posix = tomllib.loads(res_posix.content)
    assert parsed_posix["mcp_servers"]["playwright"]["command"] == "npx"
    assert parsed_posix["mcp_servers"]["playwright"]["args"] == ["-y", "@playwright/mcp@latest"]
    assert parsed_posix["mcp_servers"]["shrimp"]["command"] == "npx"
    assert parsed_posix["mcp_servers"]["shrimp"]["args"] == ["-y", "mcp-shrimp-task-manager"]
    assert parsed["mcp_servers"]["shrimp"]["command"] == "cmd"
    assert parsed["mcp_servers"]["shrimp"]["args"] == ["/c", "npx", "-y", "mcp-shrimp-task-manager"]


def test_parse_mcp_http_jsonrpc_distinguishes_json_sse_and_non_mcp() -> None:
    json_ok = sync_mcp._parse_mcp_http_jsonrpc(
        '{"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"demo"}}}', "application/json"
    )
    assert json_ok is not None and json_ok["result"]["serverInfo"]["name"] == "demo"

    sse_ok = sync_mcp._parse_mcp_http_jsonrpc(
        'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"t1"}]}}\n',
        "text/event-stream",
    )
    assert sse_ok is not None and len(sse_ok["result"]["tools"]) == 1

    assert sync_mcp._parse_mcp_http_jsonrpc("<html>OK</html>", "text/html") is None


def test_antigravity_translation_omits_unset_secret_and_resolves_when_set(tmp_path: Path) -> None:
    servers = {
        "github": {
            "type": "http",
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"Authorization": "Bearer ${GITHUB_PERSONAL_ACCESS_TOKEN}"},
            "timeout": 90000,
        },
        "serena": {
            "type": "stdio",
            "command": "uvx",
            "args": [
                "serena",
                "start-mcp-server",
                "--context",
                "claude-code",
                "--project",
                "${CLAUDE_PROJECT_DIR}",
            ],
        },
    }
    # When token is unset: must NOT write literal ${GITHUB_PERSONAL_ACCESS_TOKEN}
    res_unset = sync_mcp.render_antigravity_config(
        servers, tmp_path, env={}, existing_text='{"customKey": true}\n', target_platform="windows"
    )
    data_unset = json.loads(res_unset.content)
    assert data_unset["customKey"] is True
    assert data_unset["mcpServers"]["github"] == {
        "serverUrl": "https://api.githubcopilot.com/mcp/",
        "timeoutSeconds": 90,
    }
    assert "${GITHUB_PERSONAL_ACCESS_TOKEN}" not in res_unset.content
    assert any("GITHUB_PERSONAL_ACCESS_TOKEN" in d.message for d in res_unset.diagnostics)
    assert data_unset["mcpServers"]["serena"]["args"][3] == "ide-assistant"
    assert data_unset["mcpServers"]["serena"]["cwd"] == tmp_path.resolve().as_posix()

    # When token is set: resolves into gitignored mcp_config.json without exposing token in diagnostics
    secret = "ghp_super_secret_test_value"
    res_set = sync_mcp.render_antigravity_config(
        servers,
        tmp_path,
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": secret},
        target_platform="windows",
    )
    data_set = json.loads(res_set.content)
    assert data_set["mcpServers"]["github"]["headers"]["Authorization"] == f"Bearer {secret}"
    assert all(secret not in d.message for d in res_set.diagnostics)


def test_sync_idempotency_check_backup_and_restore(tmp_path: Path) -> None:
    _write_mcp(
        tmp_path,
        {
            "context7": {"type": "http", "url": "https://mcp.context7.com/mcp"},
            "shrimp": {
                "type": "stdio",
                "command": "cmd",
                "args": ["/c", "npx", "-y", "mcp-shrimp-task-manager"],
                "env": {"DATA_DIR": "${CLAUDE_PROJECT_DIR:-.}/.shrimp"},
            },
        },
    )
    # Seed an older codex config to verify backup + restore
    codex_path = tmp_path / ".codex" / "config.toml"
    codex_path.parent.mkdir(parents=True)
    original_codex = '[mcp_servers.old]\nurl = "https://old.example/mcp"\n'
    codex_path.write_text(original_codex, encoding="utf-8")

    # Initial check reports drift
    code_chk, msgs_chk, _ = sync_mcp.sync_all(tmp_path, check_only=True, base_env={})
    assert code_chk == 1
    assert any("OUT OF SYNC" in m for m in msgs_chk)

    # First sync updates both targets and creates a backup of the old codex file
    code1, msgs1, _ = sync_mcp.sync_all(tmp_path, base_env={})
    assert code1 == 0
    assert any("backup:" in m for m in msgs1)
    backups = list((tmp_path / ".agents" / ".mcp-backups").glob("*_codex_config.toml"))
    assert len(backups) == 1

    # Second sync is strictly idempotent: no rewrites, no extra backups
    code2, msgs2, _ = sync_mcp.sync_all(tmp_path, base_env={})
    assert code2 == 0
    assert all("up to date" in m for m in msgs2)
    assert len(list((tmp_path / ".agents" / ".mcp-backups").glob("*_codex_config.toml"))) == 1

    # Check now passes
    code_chk2, _, _ = sync_mcp.sync_all(tmp_path, check_only=True, base_env={})
    assert code_chk2 == 0

    # Restore brings back original codex config
    rc_restore, _ = sync_mcp.restore_backup(tmp_path, target="codex", list_only=False)
    assert rc_restore == 0
    assert codex_path.read_text(encoding="utf-8") == original_codex
