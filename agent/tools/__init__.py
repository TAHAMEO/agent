from . import actuators, sensors
from .base import Policy, Tool, ToolError, Toolset


def default_toolset(policy: Policy | None = None) -> Toolset:
    ts = Toolset(policy)
    sensors.register(ts)
    actuators.register(ts)
    return ts


__all__ = ["Policy", "Tool", "ToolError", "Toolset", "default_toolset"]
