"""Load a missed-standup callback brief for trusted inbound operator calls.

Outbound standup calls pass the generated plan as a Twilio ``standup_brief``
custom parameter. Inbound callbacks do not get that parameter, so a trusted
operator calling back shortly after a missed scheduled standup would otherwise
receive only the generic voice digest.

This module reconstructs that continuity from workspace artifacts:

- ``state/standup-brief.json`` — the same generated plan the outbound call used
- ``state/voice-calls/last-standup-call-sid.txt`` — evidence the standup was placed
- ``state/voice-calls/archive/*{sid}*`` — presence means the outbound was answered

Fail closed on untrusted callers, non-operators, stale/malformed briefs,
unrelated timing, and answered outbound calls.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CALLBACK_WINDOW = timedelta(minutes=30)
MAX_BRIEF_AGE = timedelta(hours=24)
FUTURE_SKEW = timedelta(minutes=5)

BRIEF_RELATIVE_PATH = Path("state") / "standup-brief.json"
SID_STAMP_RELATIVE_PATH = Path("state") / "voice-calls" / "last-standup-call-sid.txt"
ARCHIVE_RELATIVE_DIR = Path("state") / "voice-calls" / "archive"

CALLBACK_GUIDANCE_MARK = "MISSED STANDUP CALLBACK GUIDANCE"


def _parse_dt(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_brief(path: Path, now: datetime) -> dict[str, object] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    text = parsed.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    generated_at = _parse_dt(parsed.get("generated_at"))
    if generated_at is None:
        return None
    age = now - generated_at
    if age < -FUTURE_SKEW or age > MAX_BRIEF_AGE:
        return None
    return parsed


def _load_sid_stamp(path: Path) -> tuple[str, datetime] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    sid = raw.get("sid")
    if not isinstance(sid, str) or not sid.strip():
        return None
    placed_at = _parse_dt(raw.get("placed_at"))
    if placed_at is None:
        return None
    return sid.strip(), placed_at


def _outbound_was_answered(archive_dir: Path, call_sid: str) -> bool:
    """An archive file for the outbound SID means the media stream started."""
    if not call_sid or not archive_dir.is_dir():
        return False
    try:
        return any(path.is_file() for path in archive_dir.glob(f"*{call_sid}*"))
    except OSError:
        return False


def format_callback_brief(brief: dict[str, object]) -> str:
    """Render the generated plan for the realtime session instructions."""
    text = brief.get("text")
    narrative = text.strip() if isinstance(text, str) else ""
    generated_at = brief.get("generated_at")
    header = (
        f"generated_at: {generated_at}"
        if isinstance(generated_at, str) and generated_at.strip()
        else "generated_at: unknown"
    )
    structured = json.dumps(brief, ensure_ascii=False, indent=2)
    return f"{header}\n\n{narrative}\n\n--- Structured standup plan ---\n{structured}"


def build_missed_standup_callback_guidance() -> str:
    """Permanent in-session guidance for a missed-standup inbound callback."""
    return (
        f"{CALLBACK_GUIDANCE_MARK}:\n"
        "- The caller is phoning back shortly after a scheduled outbound standup "
        "that was not answered. The generated standup plan is loaded below — "
        "the same plan that would have been delivered on the outbound call.\n"
        "- When they ask for the standup, what the call was about, whether you "
        "still have the briefing, or say 'give me the standup', deliver it from "
        "this plan immediately. Do NOT search journals, spawn a subagent, or say "
        "you cannot find a recap. The plan is already in context.\n"
        "- Do NOT treat this as an outbound call you initiated: they called you. "
        "Greet them, mention you just tried them for standup, and offer to run it.\n"
        "- Follow-up questions about items in the plan are answered from the plan "
        "first. The subagent is only for genuinely novel questions the plan is "
        "silent on.\n"
        "- Queue state in the plan may have changed since generation. Frame pending "
        "items as 'as of the briefing' rather than definitely still pending.\n"
    )


def build_missed_standup_callback_greeting(
    *, spoken_name: str, agent_name: str = "bob"
) -> str:
    """First-turn instructions for a missed-standup inbound callback."""
    return (
        f"You are {agent_name.capitalize()}. "
        f"This is an inbound callback from {spoken_name} shortly after a scheduled "
        "standup call that was not answered.\n"
        f"Greet them by name in one short sentence, for example 'Hi {spoken_name}'. "
        "Then say you just tried them for standup and can run it now if they want it. "
        "Do NOT read the full standup until they ask for it or say yes. "
        "Do NOT say you need to look it up, search journals, or spawn a subagent. "
        "Do NOT say 'thanks for calling'. Then stop and wait."
    )


def load_missed_standup_callback_brief(
    workspace: str | Path | None,
    *,
    trusted: bool,
    caller_is_operator: bool,
    now: datetime | None = None,
) -> str | None:
    """Return the formatted standup plan when this inbound is a missed callback.

    Requires a trusted inbound (signature-validated /incoming or a matching
    call-scoped grant — never spoofable WebSocket custom parameters alone),
    an operator caller, a fresh generated brief, and a recent un-answered
    outbound standup stamp inside ``CALLBACK_WINDOW``.
    """
    if not trusted or not caller_is_operator or not workspace:
        return None

    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    root = Path(workspace)
    brief = _load_brief(root / BRIEF_RELATIVE_PATH, current)
    if brief is None:
        return None

    stamp = _load_sid_stamp(root / SID_STAMP_RELATIVE_PATH)
    if stamp is None:
        return None
    call_sid, placed_at = stamp
    delta = current - placed_at
    if delta < -FUTURE_SKEW or delta > CALLBACK_WINDOW:
        return None
    if _outbound_was_answered(root / ARCHIVE_RELATIVE_DIR, call_sid):
        logger.info(
            "Skipping missed-standup callback brief; outbound %s was answered",
            call_sid,
        )
        return None

    logger.info(
        "Loading missed-standup callback brief for outbound %s (placed %s)",
        call_sid,
        placed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return format_callback_brief(brief)
