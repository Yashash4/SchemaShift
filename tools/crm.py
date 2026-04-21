"""CRMAPI — search_contacts, get_contact, create_contact, update_contact + 3 drift handlers."""
from __future__ import annotations

import copy

from models import DriftEvent, ToolResponse
from tools.base import BaseTool, EndpointSchema


_SEARCH_CONTACTS_BASE = EndpointSchema(
    name="search_contacts",
    params={"customer_email": "str", "name": "str", "limit": "int"},
    required=[],
    response_shape={"contacts": "list", "total": "int"},
    error_codes={200: "ok", 400: "bad_request", 429: "rate_limit"},
)

_GET_CONTACT_BASE = EndpointSchema(
    name="get_contact",
    params={"contact_id": "str"},
    required=["contact_id"],
    response_shape={
        "contact_id": "str", "name": "str", "customer_email": "str",
        "company": "str", "status": "str",
    },
    error_codes={200: "ok", 404: "not_found", 429: "rate_limit"},
)

_CREATE_CONTACT_BASE = EndpointSchema(
    name="create_contact",
    params={"customer_email": "str", "name": "str", "company": "str"},
    required=["customer_email", "name"],
    response_shape={
        "contact_id": "str", "name": "str",
        "customer_email": "str", "status": "str",
    },
    error_codes={200: "ok", 400: "bad_request", 422: "validation", 429: "rate_limit"},
)

_UPDATE_CONTACT_BASE = EndpointSchema(
    name="update_contact",
    params={
        "contact_id": "str", "customer_email": "str", "name": "str",
        "company": "str", "status": "str",
    },
    required=["contact_id"],
    response_shape={"contact_id": "str", "status": "str"},
    error_codes={200: "ok", 404: "not_found", 422: "validation", 429: "rate_limit"},
)


class CRMAPI(BaseTool):
    name = "crm"
    baseline_schemas = {
        "search_contacts": _SEARCH_CONTACTS_BASE,
        "get_contact": _GET_CONTACT_BASE,
        "create_contact": _CREATE_CONTACT_BASE,
        "update_contact": _UPDATE_CONTACT_BASE,
    }

    def __init__(self, seed_data: dict) -> None:
        super().__init__()
        self.contacts: list[dict] = list(seed_data.get("contacts", []))
        self._next_id: int = len(self.contacts) + 1
        self._call_count: int = 0
        self._rate_limit_active: bool = False
        self.handlers = {
            "search_contacts": self._search_contacts,
            "get_contact": self._get_contact,
            "create_contact": self._create_contact,
            "update_contact": self._update_contact,
        }

    def call(self, endpoint: str, params: dict) -> ToolResponse:
        rl = self._check_rate_limit()
        if rl is not None:
            return rl
        return super().call(endpoint, params)

    def _check_rate_limit(self) -> ToolResponse | None:
        self._call_count += 1
        if self._rate_limit_active and self._call_count > 2:
            return ToolResponse(ok=False, status=429, error="Rate limit exceeded")
        return None

    def _email_field_active(self) -> str:
        any_schema = next(iter(self.active_schemas.values()))
        if "email_address" in any_schema.response_shape or "email_address" in any_schema.params:
            return "email_address"
        for s in self.active_schemas.values():
            if "email_address" in s.response_shape or "email_address" in s.params:
                return "email_address"
        return "customer_email"

    def _normalize_email_in(self, params: dict) -> dict:
        out = dict(params)
        if "email_address" in out and "customer_email" not in out:
            out["customer_email"] = out.pop("email_address")
        return out

    def _project_contact(self, contact: dict) -> dict:
        out = dict(contact)
        drifted = self._email_field_active() == "email_address"
        if drifted and "customer_email" in out:
            out["email_address"] = out.pop("customer_email")
        elif not drifted and "email_address" in out:
            out["customer_email"] = out.pop("email_address")
        return out

    def _search_contacts(self, params: dict) -> ToolResponse:
        email = params.get("email_address") or params.get("customer_email")
        name = params.get("name")

        if email is None and name is None:
            matched = list(self.contacts)
        else:
            matched = []
            for c in self.contacts:
                email_match = email is not None and c.get("customer_email", "").lower() == email.lower()
                name_match = name is not None and name.lower() in c.get("name", "").lower()
                if email_match or name_match:
                    matched.append(c)

        projected = [self._project_contact(c) for c in matched]
        return ToolResponse(
            ok=True, status=200,
            body={"contacts": projected, "total": len(projected)},
        )

    def _get_contact(self, params: dict) -> ToolResponse:
        cid = params["contact_id"]
        for c in self.contacts:
            if c.get("contact_id") == cid:
                return ToolResponse(ok=True, status=200, body=self._project_contact(c))
        return ToolResponse(ok=False, status=404, error=f"contact {cid} not found")

    def _create_contact(self, params: dict) -> ToolResponse:
        normalized = self._normalize_email_in(params)
        cid = f"c_{self._next_id}"
        self._next_id += 1
        contact = {
            "contact_id": cid,
            "customer_email": normalized.get("customer_email", ""),
            "name": normalized.get("name", ""),
            "company": normalized.get("company", ""),
            "status": normalized.get("status", "active"),
        }
        self.contacts.append(contact)
        return ToolResponse(ok=True, status=200, body=self._project_contact(contact))

    def _update_contact(self, params: dict) -> ToolResponse:
        cid = params["contact_id"]
        normalized = self._normalize_email_in(params)
        for c in self.contacts:
            if c.get("contact_id") == cid:
                for field in ("customer_email", "name", "company", "status"):
                    if field in normalized:
                        c[field] = normalized[field]
                return ToolResponse(ok=True, status=200, body=self._project_contact(c))
        return ToolResponse(ok=False, status=404, error=f"contact {cid} not found")

    def apply_drift(self, event: DriftEvent) -> None:
        if event.kind == "field_rename":
            targets = (
                "search_contacts", "get_contact", "create_contact",
                "update_contact", "contacts.patch",
            )
            for ep in targets:
                if ep not in self.active_schemas:
                    continue
                schema = self.active_schemas[ep]
                if "customer_email" in schema.params:
                    schema.params.pop("customer_email")
                    schema.params["email_address"] = "str"
                if "customer_email" in schema.required:
                    schema.required = [
                        "email_address" if r == "customer_email" else r
                        for r in schema.required
                    ]
                if "customer_email" in schema.response_shape:
                    schema.response_shape.pop("customer_email")
                    schema.response_shape["email_address"] = "str"
            return

        if event.kind == "rate_limit_tightening":
            self._rate_limit_active = True
            self._call_count = 0
            return

        if event.kind == "endpoint_deprecation" and event.endpoint == "update_contact":
            drifted_rename = any(
                "email_address" in s.params or "email_address" in s.response_shape
                for s in self.active_schemas.values()
            )
            self.active_schemas.pop("update_contact", None)
            self.handlers.pop("update_contact", None)
            new_schema = copy.deepcopy(self.baseline_schemas["update_contact"])
            new_schema.name = "contacts.patch"
            if drifted_rename:
                if "customer_email" in new_schema.params:
                    new_schema.params.pop("customer_email")
                    new_schema.params["email_address"] = "str"
                if "customer_email" in new_schema.response_shape:
                    new_schema.response_shape.pop("customer_email")
                    new_schema.response_shape["email_address"] = "str"
            self.active_schemas["contacts.patch"] = new_schema
            self.handlers["contacts.patch"] = self._update_contact
            return

        raise ValueError(
            f"CRMAPI.apply_drift: unhandled drift "
            f"kind={event.kind!r} endpoint={event.endpoint!r}"
        )
