#!/usr/bin/env python3
"""Unit tests for the macOS daemon's opt-in `auto_renew` (claude CLI renewal).

On a 401, with `auto_renew = on`, the daemon runs one minimal `claude -p` so
Claude Code renews its own token, then re-polls. The daemon itself still never
touches the OAuth refresh endpoint (see test_freeride.py).

Run: python -m pytest daemon/tests/test_macos_autorenew.py -x -q
"""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import daemon.claude_usage_daemon as mod

A = Path("/cfg/a")
PAYLOAD = {"s": 5, "sr": 100, "w": 1, "wr": 900, "st": "allowed", "acct": "pro", "ok": True}


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Never read the real config, never reuse cooldown state between tests."""
    monkeypatch.setattr(mod, "CONFIG_FILE", tmp_path / "config")
    monkeypatch.setattr(mod, "_renew_attempts", {})
    monkeypatch.setattr(mod, "_SELECTOR", mod.PlanSelector())
    return tmp_path


def _enable(tmp_path, value="on"):
    (tmp_path / "config").write_text(f"clock = off\nauto_renew = {value}  # comment\n")


def _fake_proc(returncode=0, out=b"ok\n", err=b""):
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(out, err))
    proc.wait = AsyncMock()
    return proc


# --- setting -----------------------------------------------------------------

def test_auto_renew_defaults_off(tmp_path):
    assert mod.read_auto_renew_setting() == "off"


def test_auto_renew_parses_on(tmp_path):
    _enable(tmp_path)
    assert mod.read_auto_renew_setting() == "on"


def test_auto_renew_ignores_garbage(tmp_path):
    _enable(tmp_path, "maybe")
    assert mod.read_auto_renew_setting() == "off"


# --- renew_token_via_cli -----------------------------------------------------

def test_renew_off_never_spawns_cli(tmp_path):
    spawn = AsyncMock()
    with patch.object(mod.asyncio, "create_subprocess_exec", new=spawn):
        assert asyncio.run(mod.renew_token_via_cli(A)) is False
    spawn.assert_not_called()


def test_renew_runs_minimal_cli_and_reports_success(tmp_path, monkeypatch):
    _enable(tmp_path)
    monkeypatch.setattr(mod, "find_claude_cli", lambda: "/bin/claude")
    spawn = AsyncMock(return_value=_fake_proc())
    with patch.object(mod.asyncio, "create_subprocess_exec", new=spawn):
        assert asyncio.run(mod.renew_token_via_cli(A)) is True
    args = spawn.call_args.args
    assert args[0] == "/bin/claude" and "-p" in args
    assert "--bare" not in args  # --bare skips Keychain reads -> can't renew
    assert args[args.index("--model") + 1] == "haiku"
    # Non-default dir: point the CLI at that dir's credentials.
    assert spawn.call_args.kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(A)


def test_renew_default_dir_does_not_set_config_dir(tmp_path, monkeypatch):
    _enable(tmp_path)
    monkeypatch.setattr(mod, "find_claude_cli", lambda: "/bin/claude")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/somewhere/else")
    spawn = AsyncMock(return_value=_fake_proc())
    with patch.object(mod.asyncio, "create_subprocess_exec", new=spawn):
        assert asyncio.run(mod.renew_token_via_cli(mod.DEFAULT_CONFIG_DIR)) is True
    assert "CLAUDE_CONFIG_DIR" not in spawn.call_args.kwargs["env"]


def test_renew_cli_failure_returns_false(tmp_path, monkeypatch):
    _enable(tmp_path)
    monkeypatch.setattr(mod, "find_claude_cli", lambda: "/bin/claude")
    spawn = AsyncMock(return_value=_fake_proc(1, b"", b"Invalid API key \xc2\xb7 Please run /login\n"))
    with patch.object(mod.asyncio, "create_subprocess_exec", new=spawn):
        assert asyncio.run(mod.renew_token_via_cli(A)) is False


def test_renew_cli_missing_returns_false(tmp_path, monkeypatch):
    _enable(tmp_path)
    monkeypatch.setattr(mod, "find_claude_cli", lambda: None)
    spawn = AsyncMock()
    with patch.object(mod.asyncio, "create_subprocess_exec", new=spawn):
        assert asyncio.run(mod.renew_token_via_cli(A)) is False
    spawn.assert_not_called()


def test_renew_cooldown_limits_attempts_per_dir(tmp_path, monkeypatch):
    """A dead login must not spawn a CLI run every poll."""
    _enable(tmp_path)
    monkeypatch.setattr(mod, "find_claude_cli", lambda: "/bin/claude")
    spawn = AsyncMock(return_value=_fake_proc(1, b"", b"login expired\n"))

    async def go():
        await mod.renew_token_via_cli(A)
        await mod.renew_token_via_cli(A)  # within cooldown -> skipped

    with patch.object(mod.asyncio, "create_subprocess_exec", new=spawn):
        asyncio.run(go())
    assert spawn.call_count == 1


def test_find_claude_cli_falls_back_to_local_bin(tmp_path, monkeypatch):
    """launchd's PATH lacks ~/.local/bin, where the native installer puts claude."""
    monkeypatch.setattr(mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr(mod.Path, "home", classmethod(lambda cls: tmp_path))
    cli = tmp_path / ".local" / "bin" / "claude"
    cli.parent.mkdir(parents=True)
    cli.write_text("#!/bin/sh\n")
    cli.chmod(0o755)
    assert mod.find_claude_cli() == str(cli)


# --- poll path ---------------------------------------------------------------

def test_poll_active_renews_and_repolls_on_401(tmp_path, monkeypatch):
    """Expired token -> CLI renews -> re-read token -> payload; no no-data beat."""
    _enable(tmp_path)
    monkeypatch.setattr(mod, "read_config_dirs", lambda: [A])
    tokens = iter(["OLD", "NEW"])
    monkeypatch.setattr(mod, "read_token_for", lambda d: next(tokens))
    seen = []

    async def fake_poll(token):
        seen.append(token)
        if token == "OLD":
            raise mod.TokenExpired()
        return dict(PAYLOAD)

    with patch.object(mod, "poll_api", new=fake_poll), \
         patch.object(mod, "renew_token_via_cli", new=AsyncMock(return_value=True)) as renew:
        payload, dead = asyncio.run(mod.poll_active())
    assert payload == PAYLOAD and dead is False
    assert seen == ["OLD", "NEW"]
    renew.assert_awaited_once_with(A)


def test_poll_active_still_dead_when_renew_fails(monkeypatch):
    monkeypatch.setattr(mod, "read_config_dirs", lambda: [A])
    monkeypatch.setattr(mod, "read_token_for", lambda d: "OLD")
    poll = AsyncMock(side_effect=mod.TokenExpired())
    with patch.object(mod, "poll_api", new=poll), \
         patch.object(mod, "renew_token_via_cli", new=AsyncMock(return_value=False)):
        payload, dead = asyncio.run(mod.poll_active())
    assert payload is None and dead is True
    assert poll.await_count == 1  # no re-poll without a renewal


def test_poll_active_dead_when_renewed_token_still_rejected(monkeypatch):
    monkeypatch.setattr(mod, "read_config_dirs", lambda: [A])
    monkeypatch.setattr(mod, "read_token_for", lambda d: "OLD")
    poll = AsyncMock(side_effect=mod.TokenExpired())
    with patch.object(mod, "poll_api", new=poll), \
         patch.object(mod, "renew_token_via_cli", new=AsyncMock(return_value=True)):
        payload, dead = asyncio.run(mod.poll_active())
    assert payload is None and dead is True
    assert poll.await_count == 2  # one re-poll, then give up
