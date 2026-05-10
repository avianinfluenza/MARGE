"""Adapter: wrap our orchestrator-local Python callables as BeeAI Tools.

Each tool module exposes:
- `TOOL_NAME`        — the BeeAI tool name (must match what the orchestrator
                        uses when consulting the protocol enforcer)
- `TOOL_DESCRIPTION` — human-readable, also seen by the LLM
- `ToolInput`        — Pydantic schema for input validation

This adapter reads those three off each module and wraps the bundle's
factory-built callables as `beeai_framework.tools.Tool` instances.

BeeAI's `@tool` decorator inspects the wrapped function's signature: each
parameter name must match a field in `input_schema`. Our factory closures
already have the right signatures (e.g., `get_patient_history(handle)`),
so we just wrap the return value as `JSONToolOutput`.
"""

import functools
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from apps.orchestrator.tools import (
    abstain as _ab,
    clinical_report as _cr,
    consult_expert as _ce,
    request_more_info as _rmi,
)

if TYPE_CHECKING:
    from beeai_framework.tools import Tool

    from apps.orchestrator.agent import OrchestratorBundle


# Order is the order the LLM will see them in tool listings.
# Patient tools (list_patients, get_patient, update_patient) come from the
# patient-data MCP server and are not listed here.
# Casual chat is plain natural-language content (no tool) — see system_prompt.md.
LOCAL_TOOL_MODULES: tuple[ModuleType, ...] = (_ce, _rmi, _cr, _ab)


def _to_tool_output(result: Any) -> Any:
    """Coerce a Python return value into a BeeAI ToolOutput."""
    from beeai_framework.tools import JSONToolOutput

    if isinstance(result, BaseModel):
        return JSONToolOutput(result.model_dump(mode="json"))
    return JSONToolOutput(result)


def to_beeai_tool(
    fn: Callable[..., Any],
    *,
    name: str,
    description: str,
    input_schema: type[BaseModel],
) -> "Tool":
    """Wrap a Python callable + Pydantic input schema as a BeeAI Tool.

    Sync and async callables are both supported. Returns are coerced to
    `JSONToolOutput`.
    """
    import inspect as _inspect

    from beeai_framework.tools import tool

    if _inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def output_wrapped(*args: Any, **kwargs: Any) -> Any:
            result = await fn(*args, **kwargs)
            return _to_tool_output(result)
    else:
        @functools.wraps(fn)
        def output_wrapped(*args: Any, **kwargs: Any) -> Any:
            return _to_tool_output(fn(*args, **kwargs))

    decorator = tool(name=name, description=description, input_schema=input_schema)
    return decorator(output_wrapped)


def local_tools_as_beeai(bundle: "OrchestratorBundle") -> list["Tool"]:
    """Convert local tools in `bundle` into BeeAI Tools."""
    tools: list[Tool] = []
    for mod in LOCAL_TOOL_MODULES:
        impl = bundle.local_tools[mod.TOOL_NAME]
        bt = to_beeai_tool(
            impl,
            name=mod.TOOL_NAME,
            description=mod.TOOL_DESCRIPTION,
            input_schema=mod.ToolInput,
        )
        tools.append(bt)
    return tools
