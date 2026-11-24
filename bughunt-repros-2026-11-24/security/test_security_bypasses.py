"""Security bughunt 2026-11-24: secret-detection misses / severity downgrades.

All fixtures are synthetic (repeated / made-up characters). redact() is called
with literals=[] so the env layer never participates.
"""
from __future__ import annotations

from redact import Severity, redact, _mask_preview

PW = "S3cr3t!Passw0rd"  # synthetic; '!' is a legal URL userinfo sub-delim
SECRET = "Zq9xVb7mN3kL2pR8"


def _assert_contained(text: str, secret: str) -> None:
    r = redact(text, literals=[])
    assert secret not in r.redacted, r.redacted
    assert r.blocked, [(f.category, f.severity.name) for f in r.findings]


# --- P0: supported credential types escaping in their common formats ---------

def test_telegram_token_in_bot_api_url_is_detected():
    # Canonical form in logs: https://api.telegram.org/bot<id>:<hash>/method
    token = "123456789:AA" + "b" * 33
    _assert_contained(f"curl https://api.telegram.org/bot{token}/sendMessage", token)


def test_postgres_sqlalchemy_driver_url_password_is_detected():
    _assert_contained(
        f"create_engine('postgresql+psycopg2://app:{PW}@db.internal:5432/prod')", PW
    )


def test_secret_assignment_in_json_escaped_json_is_detected():
    # Nested JSON as stored in a JSONL transcript (tool output inside a string).
    _assert_contained('{"output": "{\\"password\\": \\"' + SECRET + '\\"}"}', SECRET)


def test_yaml_block_scalar_secret_value_is_detected():
    _assert_contained("db:\n  password: |\n    " + SECRET + "\n", SECRET)


# --- P1: assignment downgraded to HIGH while the real value survives ---------

def test_php_ruby_hash_rocket_password_is_detected():
    _assert_contained(f"'password' => '{SECRET}',", SECRET)


def test_go_short_var_decl_token_is_detected():
    _assert_contained(f'token := "{SECRET}"', SECRET)


def test_unquoted_value_with_embedded_quote_is_fully_masked():
    _assert_contained(f'password: ab"{SECRET}', SECRET)


# --- P1: whole credential classes the docs/categories imply but miss ---------

def test_aws_sts_temporary_credentials_json_is_detected():
    # `aws sts assume-role` / `get-session-token` output (PascalCase keys,
    # ASIA-prefixed temporary access key id).
    secret = "w" * 40
    text = (
        '{"Credentials": {"AccessKeyId": "ASIA' + "Q" * 16 + '", '
        '"SecretAccessKey": "' + secret + '", "SessionToken": "' + "F" * 300 + '"}}'
    )
    _assert_contained(text, secret)


def test_http_basic_authorization_header_is_detected():
    b64 = "dXNlcjpaczl4VmI3bU4za0wycFI4"  # base64("user:" + synthetic pw)
    _assert_contained(f"Authorization: Basic {b64}", b64)


def test_truncated_private_key_body_is_detected():
    body = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gtZW"
    text = "-----BEGIN OPENSSH PRIVATE KEY-----\n" + (body + "\n") * 3 + "... [truncated]\n"
    _assert_contained(text, body)


# --- P2: partial redaction leaves token tail / preview leaks bytes -----------

def test_long_sk_ant_session_key_not_partially_redacted():
    tail = "Zz" * 30
    key = "sk-ant-sid01-" + "Ab_-" * 16 + tail
    r = redact(key, literals=[])
    assert r.blocked
    assert tail not in r.redacted, r.redacted


def test_high_severity_webhook_preview_does_not_echo_secret_suffix():
    url = "https://discord.com/api/webhooks/123456789012345678/" + "S" * 56 + "WXYZ"
    r = redact(url, literals=[])
    f = next(f for f in r.findings if f.category == "discord_webhook_url")
    assert f.severity == Severity.HIGH
    assert "WXYZ" not in f.preview, f.preview
