"""Missed-standup inbound callback routing."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gptme_voice.realtime.server import VoiceServer
from gptme_voice.realtime.standup_callback import (
    CALLBACK_GUIDANCE_MARK,
    load_missed_standup_callback_brief,
)

ERIK_NUMBER = "+46700000001"
PHILIP_NUMBER = "+46700000002"
OUTBOUND_SID = "CAmissedstandup00000000000000000001"


def _write_person(tmpdir: Path, filename: str, body: str) -> None:
    people_dir = tmpdir / "people"
    people_dir.mkdir(exist_ok=True)
    (people_dir / filename).write_text(body)


def _write_brief(
    tmpdir: Path,
    *,
    generated_at: datetime,
    text: str = "Good morning Erik, here is the standup.",
    extra: dict[str, object] | None = None,
) -> None:
    state_dir = tmpdir / "state"
    state_dir.mkdir(exist_ok=True)
    payload: dict[str, object] = {
        "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "text": text,
        "conversation_goals": ["Confirm the callback still has the plan."],
    }
    if extra:
        payload.update(extra)
    (state_dir / "standup-brief.json").write_text(json.dumps(payload) + "\n")


def _write_sid_stamp(
    tmpdir: Path, *, placed_at: datetime, sid: str = OUTBOUND_SID
) -> None:
    voice_dir = tmpdir / "state" / "voice-calls"
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "last-standup-call-sid.txt").write_text(
        json.dumps(
            {
                "sid": sid,
                "date": placed_at.strftime("%Y-%m-%d"),
                "placed_at": placed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
        + "\n"
    )


def _write_answered_archive(tmpdir: Path, sid: str = OUTBOUND_SID) -> None:
    archive_dir = tmpdir / "state" / "voice-calls" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / f"20260914T080000Z-000-twilio-{sid}.json").write_text(
        json.dumps(
            {
                "caller_id": ERIK_NUMBER,
                "metadata": {"call_sid": sid},
                "transcript": [{"role": "assistant", "text": "Good morning Erik"}],
            }
        )
        + "\n"
    )


def _fresh_artifacts(
    tmpdir: Path,
    *,
    now: datetime,
    placed_delta: timedelta = timedelta(minutes=2),
    generated_delta: timedelta = timedelta(hours=2),
) -> None:
    _write_brief(tmpdir, generated_at=now - generated_delta)
    _write_sid_stamp(tmpdir, placed_at=now - placed_delta)


def test_load_returns_fresh_callback_brief(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now)

    result = load_missed_standup_callback_brief(
        tmp_path, trusted=True, caller_is_operator=True, now=now
    )

    assert result is not None
    assert "Good morning Erik, here is the standup." in result
    assert "Structured standup plan" in result
    assert "Confirm the callback still has the plan." in result


def test_load_rejects_stale_brief(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now, generated_delta=timedelta(hours=25))

    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=True, caller_is_operator=True, now=now
        )
        is None
    )


def test_load_rejects_malformed_brief(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _write_sid_stamp(tmp_path, placed_at=now - timedelta(minutes=2))
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    (state_dir / "standup-brief.json").write_text("{not json")

    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=True, caller_is_operator=True, now=now
        )
        is None
    )

    (state_dir / "standup-brief.json").write_text(
        json.dumps({"generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "text": "  "})
    )
    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=True, caller_is_operator=True, now=now
        )
        is None
    )


def test_load_allows_date_boundary_within_window(tmp_path: Path) -> None:
    now = datetime(2026, 9, 15, 0, 10, tzinfo=timezone.utc)
    _write_brief(
        tmp_path, generated_at=datetime(2026, 9, 14, 5, 30, tzinfo=timezone.utc)
    )
    _write_sid_stamp(
        tmp_path, placed_at=datetime(2026, 9, 14, 23, 50, tzinfo=timezone.utc)
    )

    result = load_missed_standup_callback_brief(
        tmp_path, trusted=True, caller_is_operator=True, now=now
    )

    assert result is not None
    assert "Good morning Erik, here is the standup." in result


def test_load_rejects_still_ringing_outbound(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 0, 15, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now, placed_delta=timedelta(seconds=10))

    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=True, caller_is_operator=True, now=now
        )
        is None
    )


def test_load_rejects_unrelated_timing(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now, placed_delta=timedelta(hours=2))

    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=True, caller_is_operator=True, now=now
        )
        is None
    )


def test_load_rejects_answered_outbound(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now)
    _write_answered_archive(tmp_path)

    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=True, caller_is_operator=True, now=now
        )
        is None
    )


def test_load_rejects_answered_stamp_without_archive(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now)
    (tmp_path / "state" / "voice-calls" / "last-standup-answered.txt").write_text(
        json.dumps(
            {"sid": OUTBOUND_SID, "answered_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
        )
        + "\n"
    )

    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=True, caller_is_operator=True, now=now
        )
        is None
    )


def test_load_rejects_answered_archive_in_voice_state_dir(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now)
    state_dir = tmp_path / "voice-state"
    archive_dir = state_dir / "archive"
    archive_dir.mkdir(parents=True)
    (archive_dir / f"20260914T080000Z-000-twilio-{OUTBOUND_SID}.json").write_text(
        json.dumps({"metadata": {"call_sid": OUTBOUND_SID}}) + "\n"
    )

    assert (
        load_missed_standup_callback_brief(
            tmp_path,
            trusted=True,
            caller_is_operator=True,
            now=now,
            state_dir=state_dir,
        )
        is None
    )


def test_load_rejects_answered_recent_record(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now)
    recent_dir = tmp_path / "state" / "voice-calls" / "recent"
    recent_dir.mkdir(parents=True, exist_ok=True)
    (recent_dir / "abc123def456.json").write_text(
        json.dumps({"metadata": {"call_sid": OUTBOUND_SID}, "transcript": []}) + "\n"
    )

    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=True, caller_is_operator=True, now=now
        )
        is None
    )


def test_load_rejects_untrusted_caller(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now)

    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=False, caller_is_operator=True, now=now
        )
        is None
    )


def test_load_rejects_non_operator(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _fresh_artifacts(tmp_path, now=now)

    assert (
        load_missed_standup_callback_brief(
            tmp_path, trusted=True, caller_is_operator=False, now=now
        )
        is None
    )


def _operator_workspace(tmp_path: Path) -> None:
    _write_person(
        tmp_path,
        "erik.md",
        "# Erik Bjäreholt\n\n- Call role: operator\nPhone: +46700000001\n",
    )
    _write_person(
        tmp_path,
        "philip.md",
        "# Philip Johansson\n\nPhone: +46700000002\n",
    )


def test_bootstrap_injects_fresh_trusted_callback(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _operator_workspace(tmp_path)
    _fresh_artifacts(tmp_path, now=now)
    server = VoiceServer(workspace=str(tmp_path))
    server._instructions = "You are Bob."

    bootstrap = asyncio.run(
        server._build_session_bootstrap(
            caller_id=ERIK_NUMBER,
            from_number=ERIK_NUMBER,
            inbound_trusted=True,
            now=now,
        )
    )

    assert CALLBACK_GUIDANCE_MARK in bootstrap.instructions
    assert "Good morning Erik, here is the standup." in bootstrap.instructions
    assert "Do NOT search journals" in bootstrap.instructions
    assert bootstrap.should_greet_first is True
    assert "inbound callback" in bootstrap.initial_response_instructions
    assert "just tried them for standup" in bootstrap.initial_response_instructions
    assert "Do NOT read the full standup until they ask" in (
        bootstrap.initial_response_instructions
    )
    assert "This is an outbound daily standup call" not in (
        bootstrap.initial_response_instructions
    )


def test_bootstrap_stale_brief_keeps_normal_inbound(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _operator_workspace(tmp_path)
    _fresh_artifacts(tmp_path, now=now, generated_delta=timedelta(hours=25))
    server = VoiceServer(workspace=str(tmp_path))
    server._instructions = "You are Bob."

    bootstrap = asyncio.run(
        server._build_session_bootstrap(
            caller_id=ERIK_NUMBER,
            from_number=ERIK_NUMBER,
            inbound_trusted=True,
            now=now,
        )
    )

    assert CALLBACK_GUIDANCE_MARK not in bootstrap.instructions
    assert "Good morning Erik, here is the standup." not in bootstrap.instructions
    assert "The caller is Erik" in bootstrap.initial_response_instructions


def test_bootstrap_unrelated_timing_keeps_normal_inbound(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _operator_workspace(tmp_path)
    _fresh_artifacts(tmp_path, now=now, placed_delta=timedelta(hours=2))
    server = VoiceServer(workspace=str(tmp_path))
    server._instructions = "You are Bob."

    bootstrap = asyncio.run(
        server._build_session_bootstrap(
            caller_id=ERIK_NUMBER,
            from_number=ERIK_NUMBER,
            inbound_trusted=True,
            now=now,
        )
    )

    assert CALLBACK_GUIDANCE_MARK not in bootstrap.instructions
    assert "inbound callback" not in bootstrap.initial_response_instructions


def test_bootstrap_untrusted_caller_does_not_get_brief(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _operator_workspace(tmp_path)
    _fresh_artifacts(tmp_path, now=now)
    server = VoiceServer(workspace=str(tmp_path))
    server._instructions = "You are Bob."

    spoofed = asyncio.run(
        server._build_session_bootstrap(
            caller_id=ERIK_NUMBER,
            from_number=ERIK_NUMBER,
            inbound_trusted=False,
            now=now,
        )
    )
    non_operator = asyncio.run(
        server._build_session_bootstrap(
            caller_id=PHILIP_NUMBER,
            from_number=PHILIP_NUMBER,
            inbound_trusted=True,
            now=now,
        )
    )

    assert CALLBACK_GUIDANCE_MARK not in spoofed.instructions
    assert "Good morning Erik, here is the standup." not in spoofed.instructions
    assert CALLBACK_GUIDANCE_MARK not in non_operator.instructions
    assert "Good morning Erik, here is the standup." not in non_operator.instructions


def test_bootstrap_explicit_outbound_brief_still_wins(tmp_path: Path) -> None:
    now = datetime(2026, 9, 14, 8, 2, tzinfo=timezone.utc)
    _operator_workspace(tmp_path)
    _fresh_artifacts(tmp_path, now=now)
    server = VoiceServer(workspace=str(tmp_path))
    server._instructions = "You are Bob."

    bootstrap = asyncio.run(
        server._build_session_bootstrap(
            caller_id=ERIK_NUMBER,
            from_number=ERIK_NUMBER,
            standup_brief="OUTBOUND BRIEF PAYLOAD",
            inbound_trusted=True,
            now=now,
        )
    )

    assert "OUTBOUND BRIEF PAYLOAD" in bootstrap.instructions
    assert CALLBACK_GUIDANCE_MARK not in bootstrap.instructions
    assert "outbound daily standup call" in bootstrap.initial_response_instructions
