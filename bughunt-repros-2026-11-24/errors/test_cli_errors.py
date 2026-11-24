"""Error-handling / fail-closed / data-loss probes for the redact CLI."""

import io
import json
import subprocess
from unittest.mock import Mock

import pytest

from redact.cli import main

TOKEN = "ghp_" + "a" * 36


def test_utf16le_file_with_token_is_not_reported_clean(tmp_path, capsys):
    # BOM-less UTF-16LE of ASCII is valid UTF-8 (NUL-interleaved), so the
    # strict utf-8 read succeeds and every pattern misses.
    f = tmp_path / "leak.txt"
    f.write_bytes(f"token {TOKEN}\n".encode("utf-16-le"))
    code = main(["--json", "--no-env-secrets", str(f)])
    payload = json.loads(capsys.readouterr().out)
    assert code != 0, payload
    assert payload["blocked"] is True


def test_out_write_failure_does_not_destroy_existing_file(tmp_path, monkeypatch):
    # Under a C/POSIX locale Python's stdin uses surrogateescape, so a stray
    # non-UTF-8 byte arrives as a lone surrogate. `redact - --out f < f` then
    # truncates f via open("w") before the strict utf-8 encode fails.
    out = tmp_path / "doc.md"
    out.write_text("original precious content\n", encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO("plain text \udcff more\n"))
    try:
        main(["--no-env-secrets", "--out", str(out), "-"])
    except UnicodeEncodeError:
        pass
    assert out.read_text(encoding="utf-8") == "original precious content\n"


@pytest.mark.parametrize("mode", ["--json", "--report"])
def test_out_is_honoured_with_report_modes(tmp_path, capsys, mode):
    src = tmp_path / "in.txt"
    src.write_text("plain text", encoding="utf-8")
    out = tmp_path / "out.txt"
    code = main([mode, "--no-env-secrets", "--out", str(out), str(src)])
    assert code == 0
    assert out.exists(), f"{mode} --out silently dropped the redacted artifact"


def test_bad_cli_args_exit_code_differs_from_blocked(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("plain text", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["--no-such-flag", str(src)])
    # main() documents 2 == blocked; a usage error must not masquerade as it.
    assert exc.value.code != 2


def test_json_mode_trufflehog_failure_still_emits_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("redact.shutil.which", lambda _: "/fake/trufflehog")
    monkeypatch.setattr(
        "redact.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], 1, "", "")),
    )
    src = tmp_path / "in.txt"
    src.write_text("plain text", encoding="utf-8")
    code = main(["--json", "--trufflehog", "--no-env-secrets", str(src)])
    out = capsys.readouterr().out
    assert code == 2
    payload = json.loads(out)  # --json contract: machine-readable report
    assert payload["blocked"] is True


def test_trufflehog_encode_error_is_clean_refusal(tmp_path, monkeypatch, capsys):
    # Lone surrogate from surrogateescape stdin: write_text(utf-8) raises
    # UnicodeEncodeError, which trufflehog_scan does not convert to
    # RuntimeError, so the CLI's "refusing artifact" path is bypassed.
    monkeypatch.setattr("redact.shutil.which", lambda _: "/fake/trufflehog")
    monkeypatch.setattr(
        "redact.subprocess.run",
        Mock(return_value=subprocess.CompletedProcess([], 0, "", "")),
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("plain \udcff text\n"))
    code = main(["--trufflehog", "--no-env-secrets", "-"])
    assert code == 2
    assert "refusing artifact" in capsys.readouterr().err


def test_trufflehog_undecodable_stdout_is_clean_refusal(tmp_path, monkeypatch, capsys):
    # Real fake binary on PATH whose stdout is not UTF-8: subprocess.run(text=True)
    # raises UnicodeDecodeError, which is not caught -> raw traceback, exit 1.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "trufflehog"
    fake.write_text("#!/bin/sh\nprintf '\\377\\n'\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:/usr/bin:/bin")
    src = tmp_path / "in.txt"
    src.write_text("plain text", encoding="utf-8")
    code = main(["--trufflehog", "--no-env-secrets", str(src)])
    assert code == 2
    assert "refusing artifact" in capsys.readouterr().err
