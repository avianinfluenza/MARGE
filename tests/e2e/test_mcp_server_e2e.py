"""E2E: FastMCP server — boot in-process, discover tools, invoke via MCP transport.

Mirrors what the orchestrator does at startup: builds the FastMCP server,
opens a Client session against it, lists tools, and calls each ML tool with
sample inputs. This validates the full MCP wire contract independently of
BeeAI MCPTool wrapping.

Requires trained artifacts in services/ml_mcp_server/artifacts/.
"""

import pytest
from fastmcp import Client

from services.ml_mcp_server.server import build_server


@pytest.fixture()
def server():
    return build_server()


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

class TestMCPToolDiscovery:
    @pytest.mark.asyncio
    async def test_lists_both_ml_tools(self, server):
        async with Client(server) as client:
            tools = await client.list_tools()
        names = {t.name for t in tools}
        assert "predict_breast_cancer_malignancy" in names
        assert "predict_diabetes_risk" in names

    @pytest.mark.asyncio
    async def test_exactly_two_tools_registered(self, server):
        async with Client(server) as client:
            tools = await client.list_tools()
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self, server):
        async with Client(server) as client:
            tools = await client.list_tools()
        for tool in tools:
            assert tool.description, f"Tool '{tool.name}' has no description"

    @pytest.mark.asyncio
    async def test_tools_have_input_schemas(self, server):
        async with Client(server) as client:
            tools = await client.list_tools()
        for tool in tools:
            assert tool.inputSchema is not None, f"Tool '{tool.name}' has no input schema"


# ---------------------------------------------------------------------------
# Tool invocation — diabetes
# ---------------------------------------------------------------------------

def _text(result) -> str:
    """Extract text payload from a FastMCP CallToolResult."""
    return result.content[0].text if result.content else ""


class TestDiabetesToolInvocation:
    @pytest.mark.asyncio
    async def test_call_returns_predicted_class(self, server):
        sample = {
            "preg": 6.0, "plas": 148.0, "pres": 72.0, "skin": 35.0,
            "insu": 0.0, "mass": 33.6, "pedi": 0.627, "age": 50.0,
        }
        async with Client(server) as client:
            result = await client.call_tool("predict_diabetes_risk", {"inputs": sample})
        content = _text(result)
        assert "diabetic_risk" in content or "low_risk" in content

    @pytest.mark.asyncio
    async def test_call_result_is_valid_json(self, server):
        import json

        sample = {
            "preg": 6.0, "plas": 148.0, "pres": 72.0, "skin": 35.0,
            "insu": 0.0, "mass": 33.6, "pedi": 0.627, "age": 50.0,
        }
        async with Client(server) as client:
            result = await client.call_tool("predict_diabetes_risk", {"inputs": sample})
        parsed = json.loads(_text(result))
        assert "predicted_class" in parsed
        assert "confidence" in parsed
        assert "xai_scores" in parsed

    @pytest.mark.asyncio
    async def test_call_json_has_no_nan(self, server):
        sample = {
            "preg": 6.0, "plas": 148.0, "pres": 72.0, "skin": 35.0,
            "insu": 0.0, "mass": 33.6, "pedi": 0.627, "age": 50.0,
        }
        async with Client(server) as client:
            result = await client.call_tool("predict_diabetes_risk", {"inputs": sample})
        assert "NaN" not in _text(result), "NaN in MCP response breaks JSON transport"

    @pytest.mark.asyncio
    async def test_partial_inputs_do_not_crash(self, server):
        """Missing features become NaN inside DynamicMLAgent — must not raise."""
        partial = {"plas": 148.0, "age": 50.0}
        async with Client(server) as client:
            result = await client.call_tool("predict_diabetes_risk", {"inputs": partial})
        assert not result.is_error


# ---------------------------------------------------------------------------
# Tool invocation — breast cancer
# ---------------------------------------------------------------------------

class TestBreastCancerToolInvocation:
    @pytest.mark.asyncio
    async def test_call_returns_predicted_class(self, server):
        from sklearn.datasets import load_breast_cancer

        dataset = load_breast_cancer()
        feature_names = [n.replace(" ", "_") for n in dataset.feature_names]
        sample = {name: float(val) for name, val in zip(feature_names, dataset.data[0])}

        async with Client(server) as client:
            result = await client.call_tool(
                "predict_breast_cancer_malignancy", {"inputs": sample}
            )
        content = _text(result)
        assert "malignant" in content or "benign" in content

    @pytest.mark.asyncio
    async def test_call_result_has_xai_scores(self, server):
        import json

        from sklearn.datasets import load_breast_cancer

        dataset = load_breast_cancer()
        feature_names = [n.replace(" ", "_") for n in dataset.feature_names]
        sample = {name: float(val) for name, val in zip(feature_names, dataset.data[0])}

        async with Client(server) as client:
            result = await client.call_tool(
                "predict_breast_cancer_malignancy", {"inputs": sample}
            )
        parsed = json.loads(_text(result))
        assert len(parsed["xai_scores"]) > 0
