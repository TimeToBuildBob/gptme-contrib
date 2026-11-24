"""Bug-hunt repros for credential_slots (concurrency / data loss / security).

All fixtures are synthetic and live under tmp_path.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest
from credential_slots import SlotManager


def _ms(offset_s: float) -> int:
    return int((datetime.now(timezone.utc).timestamp() + offset_s) * 1000)


def _cred(rt: str, expires_ms: int) -> str:
    return json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": f"at-{rt}",
                "refreshToken": rt,
                "expiresAt": expires_ms,
            }
        }
    )


@pytest.fixture
def mgr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SlotManager:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    d = tmp_path / "creds"
    d.mkdir()
    return SlotManager(creds_dir=d, subscriptions=["bob", "alice"])


def _all_files_text(d: Path) -> str:
    out = []
    for p in d.iterdir():
        if p.is_file() and not p.is_symlink():
            out.append(p.read_text())
    return "\n".join(out)


# 1. switch_to irrecoverably destroys a drifted live credential ------------


def test_switch_to_destroys_drifted_live_credential(mgr: SlotManager) -> None:
    """Live is a regular file holding a fresh /login token that matches no
    slot (the drift state the package itself documents). switch_to(force=False)
    os.replace()s it with a symlink — the only on-disk copy of the fresh
    refresh token is gone."""
    mgr.slot_path("bob").write_text(_cred("rt-bob-old", _ms(3600)))
    mgr.slot_path("alice").write_text(_cred("rt-alice", _ms(3600)))
    mgr.live_path.write_text(_cred("rt-fresh-login", _ms(3600)))
    drift = mgr.detect_live_slot_drift()
    assert drift is not None and drift["drift"] is True

    result = mgr.switch_to("alice", "rebalance")

    survived = "rt-fresh-login" in _all_files_text(mgr.creds_dir)
    assert (not result.ok) or survived, (
        f"switch_to returned ok={result.ok} and the drifted live credential "
        "was destroyed (no file in creds_dir still holds it)"
    )


# 2. heal_drift_to writes credential content to a group/world-readable tmp --


def test_heal_slot_tmp_readable_before_chmod(
    mgr: SlotManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr.slot_path("bob").write_text(_cred("rt-old", _ms(-3600)))
    mgr.live_path.write_text(_cred("rt-new", _ms(3600)))
    os.chmod(mgr.live_path, 0o600)
    old_umask = os.umask(0o022)
    seen: list[tuple[str, int, int]] = []
    real_copyfile = shutil.copyfile

    def spy_copyfile(src, dst, *a, **kw):  # type: ignore[no-untyped-def]
        r = real_copyfile(src, dst, *a, **kw)
        st = os.stat(dst)
        seen.append((str(dst), stat.S_IMODE(st.st_mode), st.st_size))
        return r

    monkeypatch.setattr(shutil, "copyfile", spy_copyfile)
    try:
        result = mgr.heal_drift_to("bob")
    finally:
        os.umask(old_umask)
    assert result.ok, result.reason
    assert seen, "copyfile not called"
    path, mode, size = seen[0]
    assert size > 0
    assert mode & 0o077 == 0, (
        f"{Path(path).name} held {size} bytes of credential at mode {oct(mode)} "
        "before chmod(0o600)"
    )


# 3. heal validates one read of live, but copies a second read --------------


def test_heal_copies_unvalidated_live_bytes(
    mgr: SlotManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer that truncates-then-writes live (non-atomic) between heal's
    validation read and its copy: heal reports ok and installs the partial
    file as the slot."""
    mgr.slot_path("bob").write_text(_cred("rt-old", _ms(-3600)))
    good = _cred("rt-new", _ms(3600))
    mgr.live_path.write_text(good)
    real_copy2 = shutil.copy2

    def racing_copy2(src, dst, *a, **kw):  # type: ignore[no-untyped-def]
        Path(src).write_text(good[:20])  # writer mid-write
        return real_copy2(src, dst, *a, **kw)

    monkeypatch.setattr(shutil, "copy2", racing_copy2)
    result = mgr.heal_drift_to("bob")
    slot_text = mgr.slot_path("bob").read_text()
    try:
        json.loads(slot_text)
        slot_valid = True
    except json.JSONDecodeError:
        slot_valid = False
    assert (not result.ok) or slot_valid, (
        f"heal ok={result.ok} but slot now holds unvalidated partial JSON "
        f"({len(slot_text)} bytes)"
    )


# 4. heal overwrites live without re-checking -> concurrent refresh lost ----


def test_heal_drops_token_written_to_live_during_heal(
    mgr: SlotManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    mgr.slot_path("bob").write_text(_cred("rt-old", _ms(-3600)))
    mgr.live_path.write_text(_cred("rt-new", _ms(3600)))
    live = str(mgr.live_path)
    real_replace = os.replace

    def racing_replace(src, dst, *a, **kw):  # type: ignore[no-untyped-def]
        if str(dst) == live:
            # CC rotates the refresh token again, writing live, right before
            # heal's symlink swap.
            Path(live).write_text(_cred("rt-newest", _ms(7200)))
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", racing_replace)
    result = mgr.heal_drift_to("bob")
    survived = "rt-newest" in _all_files_text(mgr.creds_dir)
    assert (not result.ok) or survived, (
        "heal returned ok and the refresh token written to live during the "
        "heal exists nowhere on disk"
    )


# 5. Fixed per-pid tmp name: concurrent same-process switches clobber -------


def test_concurrent_switch_same_process_tmp_collision(
    mgr: SlotManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thread B entering switch_to while thread A is between symlink_to and
    os.replace: B's symlink_to hits FileExistsError, B's cleanup unlinks A's
    tmp, A's os.replace then fails. Neither switch completes; both raise."""
    mgr.slot_path("bob").write_text(_cred("rt-bob", _ms(3600)))
    mgr.slot_path("alice").write_text(_cred("rt-alice", _ms(3600)))
    mgr.live_path.symlink_to(".credentials.json.bob")
    real_replace = os.replace
    nested: dict[str, object] = {}

    def interleave(src, dst, *a, **kw):  # type: ignore[no-untyped-def]
        if not nested:
            nested["started"] = True
            try:
                nested["result"] = mgr.switch_to("bob", "thread B")
            except BaseException as e:  # noqa: BLE001
                nested["exc"] = e
        return real_replace(src, dst, *a, **kw)

    monkeypatch.setattr(os, "replace", interleave)
    outer_exc: BaseException | None = None
    try:
        outer = mgr.switch_to("alice", "thread A")
    except BaseException as e:  # noqa: BLE001
        outer_exc = e
        outer = None
    assert outer_exc is None and "exc" not in nested, (
        f"outer: {outer_exc!r}; nested: {nested.get('exc')!r}"
    )
    assert outer is not None and outer.ok
