"""Wrap expert-only Python callables as BeeAI tools."""

from __future__ import annotations

import functools
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from services.medical_expert_agent.tools import search_web as _search_web

if TYPE_CHECKING:
    from beeai_framework.tools import Tool


EXPERT_TOOL_MODULES: tuple[ModuleType, ...] = (_search_web,)


def _to_tool_output(result: Any) -> Any:
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
    import inspect as _inspect

    from beeai_framework.tools import tool

    if _inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def output_wrapped(*args: Any, **kwargs: Any) -> Any:
            return _to_tool_output(await fn(*args, **kwargs))

    else:

        @functools.wraps(fn)
        def output_wrapped(*args: Any, **kwargs: Any) -> Any:
            return _to_tool_output(fn(*args, **kwargs))

    decorator = tool(name=name, description=description, input_schema=input_schema)
    return decorator(output_wrapped)


def expert_tools_as_beeai(
    overrides: dict[str, Callable[..., Any]] | None = None,
) -> list["Tool"]:
    tools: list[Tool] = []
    for mod in EXPERT_TOOL_MODULES:
        impl = overrides.get(mod.TOOL_NAME) if overrides else getattr(mod, mod.TOOL_NAME)
        tools.append(
            to_beeai_tool(
                impl,
                name=mod.TOOL_NAME,
                description=mod.TOOL_DESCRIPTION,
                input_schema=mod.ToolInput,
            )
        )
    return tools
