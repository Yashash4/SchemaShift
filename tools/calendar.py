"""CalendarAPI — list_events, create_event, update_event, delete_event + 2 drift handlers."""
from __future__ import annotations

from models import DriftEvent, ToolResponse
from tools.base import BaseTool, EndpointSchema


_LIST_EVENTS_BASE = EndpointSchema(
    name="list_events",
    params={"date_from": "str", "date_to": "str", "limit": "int"},
    required=["date_from", "date_to"],
    response_shape={"events": "list"},
    error_codes={200: "ok", 400: "bad_request"},
)

_CREATE_EVENT_BASE = EndpointSchema(
    name="create_event",
    params={
        "title": "str", "start": "str", "end": "str",
        "attendees": "list", "location": "str",
    },
    required=["title", "start", "end", "attendees"],
    response_shape={
        "event_id": "str", "title": "str", "start": "str", "end": "str",
        "attendees": "list", "status": "str",
    },
    error_codes={200: "ok", 400: "bad_request", 422: "validation"},
)

_UPDATE_EVENT_BASE = EndpointSchema(
    name="update_event",
    params={
        "event_id": "str", "title": "str", "start": "str", "end": "str",
        "attendees": "list", "location": "str", "status": "str",
    },
    required=["event_id"],
    response_shape={"event_id": "str", "status": "str", "title": "str"},
    error_codes={200: "ok", 404: "not_found"},
)

_DELETE_EVENT_BASE = EndpointSchema(
    name="delete_event",
    params={"event_id": "str"},
    required=["event_id"],
    response_shape={"deleted": "bool", "event_id": "str"},
    error_codes={200: "ok", 404: "not_found"},
)


class CalendarAPI(BaseTool):
    name = "calendar"
    baseline_schemas = {
        "list_events": _LIST_EVENTS_BASE,
        "create_event": _CREATE_EVENT_BASE,
        "update_event": _UPDATE_EVENT_BASE,
        "delete_event": _DELETE_EVENT_BASE,
    }

    def __init__(self, seed_data: dict) -> None:
        super().__init__()
        self.events: list[dict] = list(seed_data.get("events", []))
        self._next_id: int = len(self.events) + 1
        self.handlers = {
            "list_events": self._list_events,
            "create_event": self._create_event,
            "update_event": self._update_event,
            "delete_event": self._delete_event,
        }

    def _list_events(self, params: dict) -> ToolResponse:
        date_from = params["date_from"]
        date_to = params["date_to"]
        # Overlap: event starts on/before date_to AND ends on/after date_from.
        filtered = [
            e for e in self.events
            if e.get("start", "") <= date_to and e.get("end", "") >= date_from
        ]
        return ToolResponse(ok=True, status=200, body={"events": filtered[:10]})

    def _create_event(self, params: dict) -> ToolResponse:
        event_id = f"evt_{self._next_id}"
        self._next_id += 1

        title = params.get("title", "")
        start = params.get("start", "")
        end = params.get("end", "")
        location = params.get("location", "")
        status = params.get("status", "confirmed")

        shape = self.active_schemas["create_event"].response_shape
        drifted = "participants" in shape

        if drifted:
            if "participants" in params:
                participants = list(params["participants"])
            elif "attendees" in params:
                participants = [
                    {"email": e, "role": "required"} for e in params["attendees"]
                ]
            else:
                participants = []
            event = {
                "event_id": event_id, "title": title, "start": start, "end": end,
                "participants": participants, "location": location, "status": status,
            }
        else:
            attendees = list(params.get("attendees", []))
            event = {
                "event_id": event_id, "title": title, "start": start, "end": end,
                "attendees": attendees, "location": location, "status": status,
            }

        self.events.append(event)
        return ToolResponse(ok=True, status=200, body=dict(event))

    def _update_event(self, params: dict) -> ToolResponse:
        event_id = params["event_id"]
        for event in self.events:
            if event.get("event_id") == event_id:
                for field in (
                    "title", "start", "end", "attendees", "participants",
                    "location", "status",
                ):
                    if field in params:
                        event[field] = params[field]
                return ToolResponse(ok=True, status=200, body=dict(event))
        return ToolResponse(
            ok=False, status=404, error=f"event {event_id} not found"
        )

    def _delete_event(self, params: dict) -> ToolResponse:
        event_id = params["event_id"]
        for i, event in enumerate(self.events):
            if event.get("event_id") == event_id:
                self.events.pop(i)
                return ToolResponse(
                    ok=True, status=200,
                    body={"deleted": True, "event_id": event_id},
                )
        return ToolResponse(
            ok=False, status=404, error=f"event {event_id} not found"
        )

    def apply_drift(self, event: DriftEvent) -> None:
        if event.kind == "field_rename" and event.endpoint == "create_event":
            schema = self.active_schemas["create_event"]
            schema.params.pop("attendees", None)
            schema.params["participants"] = "list"
            schema.required = [
                "participants" if r == "attendees" else r for r in schema.required
            ]
            schema.response_shape.pop("attendees", None)
            schema.response_shape["participants"] = "list"
            return

        if event.kind == "tool_removal" and event.endpoint == "delete_event":
            self.active_schemas.pop("delete_event", None)
            self.handlers.pop("delete_event", None)
            return

        raise ValueError(
            f"CalendarAPI.apply_drift: unhandled drift "
            f"kind={event.kind!r} endpoint={event.endpoint!r}"
        )
