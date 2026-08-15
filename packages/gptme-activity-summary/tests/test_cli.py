"""Tests for the activity-summary CLI."""

from pathlib import Path

from click.testing import CliRunner

from gptme_activity_summary.cli import cli


def test_smart_exits_nonzero_when_daily_generation_fails(monkeypatch):
    """The systemd wrapper must see a failed required daily summary as failed."""

    monkeypatch.setattr(
        "gptme_activity_summary.cli.get_journal_entries_for_date",
        lambda _target_date: [Path("journal.md")],
    )

    def fail_daily(_target_date, verbose=False):  # noqa: ANN001, ANN202
        raise RuntimeError("Claude weekly limit")

    monkeypatch.setattr("gptme_activity_summary.cli.generate_daily_with_cc", fail_daily)

    result = CliRunner().invoke(cli, ["smart", "--date", "2026-08-14"])

    assert result.exit_code == 1
    assert "daily: FAILED" in result.output
