"""BaseTool + EndpointSchema — abstract contract for all SaaS tool APIs."""
from __future__ import annotations

import copy
from typing import Callable, Optional

from pydantic import BaseModel

from models import DriftEvent, ToolResponse


class EndpointSchema(BaseModel):
    name: str
    params: dict[str, str]
    required: list[str]
    response_shape: dict[str, str]
    error_codes: dict[int, str]


class BaseTool:
    name: str = ""
    baseline_schemas: dict[str, EndpointSchema] = {}

    def __init__(self) -> None:
        self.active_schemas: dict[str, EndpointSchema] = copy.deepcopy(self.baseline_schemas)
        self.handlers: dict[str, Callable[[dict], ToolResponse]] = {}

    def get_schema(self, endpoint: Optional[str] = None) -> dict:
        if endpoint is None:
            return {k: v.model_dump() for k, v in self.active_schemas.items()}
        if endpoint not in self.active_schemas:
            return {}
        return self.active_schemas[endpoint].model_dump()

    def call(self, endpoint: str, params: dict) -> ToolResponse:
        if endpoint not in self.active_schemas:
            return ToolResponse(
                ok=False,
                status=410,
                error=f"Endpoint '{endpoint}' is no longer available on {self.name}.",
            )
        schema = self.active_schemas[endpoint]
        unknown = [p for p in params if p not in schema.params]
        if unknown:
            return ToolResponse(
                ok=False,
                status=400,
                error=f"Unknown params on {endpoint}: {unknown}",
            )
        missing = [p for p in schema.required if p not in params]
        if missing:
            return ToolResponse(
                ok=False,
                status=400,
                error=f"Missing required params: {missing}",
            )
        handler = self.handlers.get(endpoint)
        if handler is None:
            return ToolResponse(
                ok=False,
                status=501,
                error=f"No handler registered for endpoint '{endpoint}' on {self.name}.",
            )
        return handler(params)

    def apply_drift(self, event: DriftEvent) -> None:
        raise NotImplementedError
