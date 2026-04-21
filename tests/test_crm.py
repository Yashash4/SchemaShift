"""CRMAPI acceptance tests — Phase 4."""
from __future__ import annotations

import pytest

from models import DriftEvent
from tools.crm import CRMAPI


def _fresh_crm() -> CRMAPI:
    return CRMAPI(seed_data={
        "contacts": [
            {"contact_id": "c_1", "customer_email": "alice@customer.com",
             "name": "Alice Nguyen", "company": "Acme Corp", "status": "active"},
            {"contact_id": "c_2", "customer_email": "bob@customer.com",
             "name": "Bob Taylor", "company": "Globex Industries", "status": "active"},
            {"contact_id": "c_3", "customer_email": "carol@customer.com",
             "name": "Carol Davis", "company": "Initech", "status": "inactive"},
        ]
    })


def test_baseline_search_contacts() -> None:
    crm = _fresh_crm()
    by_email = crm.call("search_contacts", {"customer_email": "bob@customer.com"})
    assert by_email.ok is True
    assert by_email.status == 200
    assert by_email.body is not None
    assert by_email.body["total"] == 1
    assert by_email.body["contacts"][0]["name"] == "Bob Taylor"
    assert by_email.body["contacts"][0]["customer_email"] == "bob@customer.com"

    by_name = crm.call("search_contacts", {"name": "Alice"})
    assert by_name.ok is True
    assert by_name.body["total"] == 1
    assert by_name.body["contacts"][0]["contact_id"] == "c_1"

    all_contacts = crm.call("search_contacts", {})
    assert all_contacts.ok is True
    assert all_contacts.body["total"] == 3


def test_baseline_create_and_get() -> None:
    crm = _fresh_crm()
    created = crm.call("create_contact", {
        "customer_email": "dan@customer.com",
        "name": "Dan Ellis",
        "company": "DEMO Corp",
    })
    assert created.ok is True
    assert created.status == 200
    cid = created.body["contact_id"]
    assert cid.startswith("c_")

    got = crm.call("get_contact", {"contact_id": cid})
    assert got.ok is True
    assert got.status == 200
    assert got.body is not None
    assert got.body["name"] == "Dan Ellis"
    assert got.body["customer_email"] == "dan@customer.com"
    assert got.body["company"] == "DEMO Corp"


def test_drift_field_rename_crm_tool_wide() -> None:
    crm = _fresh_crm()
    event = DriftEvent(
        tool="crm", endpoint=None, kind="field_rename",
        fires_at_step=1, details={},
    )
    crm.apply_drift(event)

    for ep in ("search_contacts", "get_contact", "create_contact", "update_contact"):
        schema = crm.get_schema(ep)
        assert "customer_email" not in schema["params"]
        assert "customer_email" not in schema["required"]
        assert "customer_email" not in schema["response_shape"]

    assert "email_address" in crm.get_schema("search_contacts")["params"]
    assert "email_address" in crm.get_schema("create_contact")["required"]
    assert "email_address" in crm.get_schema("get_contact")["response_shape"]

    bad = crm.call("search_contacts", {"customer_email": "bob@customer.com"})
    assert bad.ok is False
    assert bad.status == 400

    ok = crm.call("search_contacts", {"email_address": "bob@customer.com"})
    assert ok.ok is True
    assert ok.status == 200
    assert ok.body is not None
    assert ok.body["total"] == 1
    contact = ok.body["contacts"][0]
    assert "email_address" in contact
    assert "customer_email" not in contact
    assert contact["email_address"] == "bob@customer.com"


def test_drift_rate_limit_tightening() -> None:
    crm = _fresh_crm()
    event = DriftEvent(
        tool="crm", endpoint=None, kind="rate_limit_tightening",
        fires_at_step=1, details={},
    )
    crm.apply_drift(event)

    r1 = crm.call("search_contacts", {})
    assert r1.ok is True

    r2 = crm.call("search_contacts", {})
    assert r2.ok is True

    r3 = crm.call("search_contacts", {})
    assert r3.ok is False
    assert r3.status == 429


def test_drift_endpoint_deprecation_update_contact() -> None:
    crm = _fresh_crm()
    event = DriftEvent(
        tool="crm", endpoint="update_contact", kind="endpoint_deprecation",
        fires_at_step=1, details={"replacement": "contacts.patch"},
    )
    crm.apply_drift(event)

    old = crm.call("update_contact", {"contact_id": "c_1", "status": "updated"})
    assert old.ok is False
    assert old.status == 410

    new = crm.call("contacts.patch", {"contact_id": "c_1", "status": "updated"})
    assert new.ok is True
    assert new.status == 200
    assert new.body is not None
    assert new.body["status"] == "updated"


def test_crm_unknown_drift_raises() -> None:
    crm = _fresh_crm()
    event = DriftEvent(
        tool="crm", endpoint="search_contacts", kind="tool_removal",
        fires_at_step=1, details={},
    )
    with pytest.raises(ValueError):
        crm.apply_drift(event)
