"""MailAPI + BaseTool acceptance tests — Phase 2."""
from __future__ import annotations

from models import DriftEvent
from tools.mail import MailAPI


def _fresh_mail() -> MailAPI:
    seed = {
        "messages": [
            {"id": "msg_1", "from": "a@x.com", "to": "u@org.com",
             "subject": "hi", "body": "hello", "folder": "inbox"},
            {"id": "msg_2", "from": "b@x.com", "to": "u@org.com",
             "subject": "welcome", "body": "hi there", "folder": "inbox"},
            {"id": "msg_3", "from": "c@x.com", "to": "u@org.com",
             "subject": "fyi", "body": "fyi", "folder": "inbox"},
        ]
    }
    return MailAPI(seed_data=seed)


def test_baseline_list_messages() -> None:
    mail = _fresh_mail()
    resp = mail.call("list_messages", {"folder": "inbox"})
    assert resp.ok is True
    assert resp.status == 200
    assert resp.body is not None
    assert "messages" in resp.body
    assert "next_page_token" in resp.body
    assert len(resp.body["messages"]) == 3


def test_baseline_send_message() -> None:
    mail = _fresh_mail()
    resp = mail.call(
        "send_message",
        {"to": "x@y.com", "subject": "Hello", "body": "Hi!"},
    )
    assert resp.ok is True
    assert resp.status == 200
    assert resp.body is not None
    assert "message_id" in resp.body
    assert "sent_at" in resp.body


def test_missing_required_param() -> None:
    mail = _fresh_mail()
    resp = mail.call(
        "send_message",
        {"to": "x@y.com", "subject": "Hello"},  # body missing
    )
    assert resp.ok is False
    assert resp.status == 400
    assert resp.error is not None
    assert "missing" in resp.error.lower()
    assert "body" in resp.error


def test_drift_field_rename_list_messages() -> None:
    mail = _fresh_mail()
    event = DriftEvent(
        tool="mail",
        endpoint="list_messages",
        kind="field_rename",
        fires_at_step=1,
        details={"from": "messages", "to": "items"},
    )
    mail.apply_drift(event)
    resp = mail.call("list_messages", {"folder": "inbox"})
    assert resp.ok is True
    assert resp.body is not None
    assert "items" in resp.body
    assert "next_cursor" in resp.body
    assert "messages" not in resp.body
    assert "next_page_token" not in resp.body


def test_drift_endpoint_deprecation() -> None:
    mail = _fresh_mail()
    event = DriftEvent(
        tool="mail",
        endpoint="send_message",
        kind="endpoint_deprecation",
        fires_at_step=1,
        details={"replacement": "messages.send"},
    )
    mail.apply_drift(event)

    old = mail.call(
        "send_message",
        {"to": "x@y.com", "subject": "Hi", "body": "Hi"},
    )
    assert old.ok is False
    assert old.status == 410

    new = mail.call(
        "messages.send",
        {"to": "x@y.com", "subject": "Hi", "body": "Hi"},
    )
    assert new.ok is True
    assert new.status == 200
    assert new.body is not None
    assert "message_id" in new.body


def test_drift_new_required_param() -> None:
    mail = _fresh_mail()
    event = DriftEvent(
        tool="mail",
        endpoint="send_message",
        kind="new_required_param",
        fires_at_step=1,
        details={"param": "idempotency_key"},
    )
    mail.apply_drift(event)

    without_key = mail.call(
        "send_message",
        {"to": "x@y.com", "subject": "Hi", "body": "Hi"},
    )
    assert without_key.ok is False
    assert without_key.status == 400
    assert without_key.error is not None
    assert "idempotency_key" in without_key.error

    with_key = mail.call(
        "send_message",
        {"to": "x@y.com", "subject": "Hi", "body": "Hi",
         "idempotency_key": "k1"},
    )
    assert with_key.ok is True
    assert with_key.status == 200


def test_get_message_roundtrip() -> None:
    mail = _fresh_mail()
    sent = mail.call(
        "send_message",
        {"to": "x@y.com", "subject": "Hi", "body": "Hi there"},
    )
    assert sent.ok is True
    assert sent.body is not None
    mid = sent.body["message_id"]

    got = mail.call("get_message", {"message_id": mid})
    assert got.ok is True
    assert got.status == 200
    assert got.body is not None
    assert got.body["id"] == mid
