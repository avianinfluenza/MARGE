"""FastMCP server that exposes every registered MLModel as a tool.

Run as a stdio MCP server: `python -m services.ml_mcp_server.server`

The orchestrator (BeeAI) connects to this over MCP and discovers the tools
dynamically — adding a new model never requires orchestrator changes.
"""

from typing import Any

from fastmcp import FastMCP

from packages.schemas.prediction import Prediction
from services.ml_mcp_server.models._base import MLModel
from services.ml_mcp_server.registry import discover_models


def _register(mcp: FastMCP, model: MLModel) -> None:
    """Register one MLModel as an MCP tool with proper input/output schemas."""
    input_cls = model.input_schema

    def tool_fn(inputs: input_cls) -> Any:  # type: ignore[valid-type]
        from pydantic import TypeAdapter
        result = model.predict(inputs)
        # Ensure NaNs are converted to nulls for valid MCP JSON output
        return TypeAdapter(Prediction).dump_python(result, mode="json")

    tool_fn.__name__ = model.name
    tool_fn.__doc__ = model.metadata.description
    mcp.tool(tool_fn)


def build_server() -> FastMCP:
    mcp = FastMCP("ml-models")
    for model in discover_models():
        _register(mcp, model)
    return mcp


def main() -> None:
    server = build_server()
    server.run()  # stdio transport


if __name__ == "__main__":
    main()
