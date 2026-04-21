"""MailAPI — list_messages, send_message, get_message + 3 drift handlers."""
from __future__ import annotations

import copy

from models import DriftEvent, ToolResponse
from tools.base import BaseTool, EndpointSchema


_LIST_MESSAGES_BASE = EndpointSchema(
    name="list_messages",
    params={"folder": "str", "limit": "int", "page_token": "str"},
    required=["folder"],
    response_shape={"messages": "list", "next_page_token": "str"},
    error_codes={200: "ok", 400: "bad_request", 429: "rate_limit"},
)

_SEND_MESSAGE_BASE = EndpointSchema(
    name="send_message",
    params={"to": "str", "subject": "str", "body": "str"},
    required=["to", "subject", "body"],
    response_shape={"message_id": "str", "sent_at": "str"},
    error_codes={200: "ok", 400: "bad_request", 422: "validation"},
)

_GET_MESSAGE_BASE = EndpointSchema(
    name="get_message",
    params={"message_id": "str"},
    required=["message_id"],
    response_shape={"id": "str", "from": "str", "subject": "str", "body": "str"},
    error_codes={200: "ok", 404: "not_found"},
)


class MailAPI(BaseTool):
    name = "mail"
    baseline_schemas = {
        "list_messages": _LIST_MESSAGES_BASE,
        "send_message": _SEND_MESSAGE_BASE,
        "get_message": _GET_MESSAGE_BASE,
    }

    def __init__(self, seed_data: dict) -> None:
        super().__init__()
        self.mailbox: list[dict] = list(seed_data.get("messages", []))
        self.sent_messages: list[dict] = []
        self.handlers = {
            "list_messages": self._list_messages,
            "send_message": self._send_message,
            "get_message": self._get_message,
        }

    def _list_messages(self, params: dict) -> ToolResponse:
        folder = params["folder"]
        items = [m for m in self.mailbox if m.get("folder") == folder]
        shape_keys = set(self.active_schemas["list_messages"].response_shape.keys())
        if "items" in shape_keys and "next_cursor" in shape_keys:
            return ToolResponse(
                ok=True,
                status=200,
                body={
                    "items": items[:10],
                    "next_cursor": "cur_abc123" if len(items) > 10 else None,
                },
            )
        return ToolResponse(
            ok=True,
            status=200,
            body={
                "messages": items[:10],
                "next_page_token": "tok_xyz" if len(items) > 10 else None,
            },
        )

    def _send_message(self, params: dict) -> ToolResponse:
        mid = f"msg_{len(self.sent_messages) + len(self.mailbox) + 1}"
        self.sent_messages.append({"id": mid, **params})
        return ToolResponse(
            ok=True,
            status=200,
            body={"message_id": mid, "sent_at": "2026-04-25T10:00:00Z"},
        )

    def _get_message(self, params: dict) -> ToolResponse:
        mid = params["message_id"]
        for m in self.mailbox:
            if m.get("id") == mid:
                return ToolResponse(ok=True, status=200, body=dict(m))
        for m in self.sent_messages:
            if m.get("id") == mid:
                return ToolResponse(ok=True, status=200, body=dict(m))
        return ToolResponse(ok=False, status=404, error="message not found")

    def apply_drift(self, event: DriftEvent) -> None:
        if event.kind == "field_rename" and event.endpoint == "list_messages":
            self.active_schemas["list_messages"].response_shape = {
                "items": "list",
                "next_cursor": "str",
            }
            return

        if event.kind == "endpoint_deprecation" and event.endpoint == "send_message":
            self.active_schemas.pop("send_message", None)
            self.handlers.pop("send_message", None)
            new_schema = copy.deepcopy(self.baseline_schemas["send_message"])
            new_schema.name = "messages.send"
            self.active_schemas["messages.send"] = new_schema
            self.handlers["messages.send"] = self._send_message
            return

        if event.kind == "new_required_param" and event.endpoint == "send_message":
            schema = self.active_schemas["send_message"]
            if "idempotency_key" not in schema.required:
                schema.required.append("idempotency_key")
            schema.params["idempotency_key"] = "str"
            return

        raise ValueError(
            f"MailAPI.apply_drift: unhandled drift "
            f"kind={event.kind!r} endpoint={event.endpoint!r}"
        )
