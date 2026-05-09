"""Auto-discovers MLModel subclasses in models/ and instantiates them.

Adding a new clinical model = drop a new file under models/ that defines a
single MLModel subclass. The registry imports the module, finds the subclass,
instantiates it, and surfaces it to the MCP server.

Files starting with `_` (e.g., `_base.py`) are skipped.
"""

import importlib
import inspect
import pkgutil

from services.ml_mcp_server.models._base import MLModel


def discover_models() -> list[MLModel]:
    """Import every module in services.ml_mcp_server.models and instantiate every MLModel subclass."""
    instances: list[MLModel] = []
    package = importlib.import_module("services.ml_mcp_server.models")

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"services.ml_mcp_server.models.{module_info.name}")
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, MLModel) or obj is MLModel:
                continue
            if obj.__module__ != module.__name__:
                continue  # skip re-exports
            try:
                instances.append(obj())
            except FileNotFoundError as e:
                print(f"[ml_mcp_server.registry] skipping {obj.__name__}: {e}")
    return instances
