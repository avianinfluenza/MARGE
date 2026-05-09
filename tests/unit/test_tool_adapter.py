"""Tests for the BeeAI tool adapter.

The adapter wraps Python callables as BeeAI Tools.
`local_tools_as_beeai(bundle)` returns the two orchestrator-local tools
(consult_medical_expert, final_report) as BeeAI Tools with the enforcer wired in.
"""

import asyncio

from pydantic import BaseModel, Field

from apps.orchestrator.agent import build_bundle
from apps.orchestrator.tools._adapter import local_tools_as_beeai, to_beeai_tool


class _SampleInput(BaseModel):
    x: int = Field(description="a number")


class TestToBeeaiTool:
    def test_preserves_name_and_description(self):
        def fn(input_obj: _SampleInput) -> dict:
            return {"x": input_obj.x}

        bt = to_beeai_tool(fn, name="sample", description="sample tool", input_schema=_SampleInput)
        assert bt.name == "sample"
        assert bt.description == "sample tool"

    def test_returns_a_beeai_tool(self):
        from beeai_framework.tools import Tool

        def fn(input_obj: _SampleInput) -> dict:
            return {"x": input_obj.x}

        bt = to_beeai_tool(fn, name="sample", description="d", input_schema=_SampleInput)
        assert isinstance(bt, Tool)

    def test_supports_async_callables(self):
        async def fn(x: int) -> dict:
            return {"x": x}

        bt = to_beeai_tool(
            fn, name="sample_async", description="d", input_schema=_SampleInput
        )
        result = asyncio.get_event_loop().run_until_complete(bt.run({"x": 7}))

        assert result.result == {"x": 7}


class TestLocalToolsAsBeeai:
    def test_returns_two_tools(self):
        bundle = build_bundle()
        tools = local_tools_as_beeai(bundle)
        assert len(tools) == 2

    def test_tool_names_match_expected_set(self):
        bundle = build_bundle()
        tools = local_tools_as_beeai(bundle)
        names = {t.name for t in tools}
        assert names == {"consult_medical_expert", "final_report"}
