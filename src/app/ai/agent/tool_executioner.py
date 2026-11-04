from app.ai.llm.models import Tool, ToolCall, ToolResult


import inspect
import json


class ToolExecutor:
    """Dispatches tool calls safely with error recovery."""

    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    async def execute(self, call: ToolCall) -> ToolResult:
        # print(call,"\n")

        tool = self._tools.get(call.name)
        if not tool:
            return ToolResult(
                name=call.name,
                tool_call_id=call.tool_call_id,
                output=f"Unknown tool: '{call.name}'",
                is_error=True,
            )
        try:
            res = tool.handler(**call.arguments)
            if inspect.isawaitable(res):
                res = await res
            output = res if isinstance(res, str) else json.dumps(res)
            return ToolResult(
                name=call.name, tool_call_id=call.tool_call_id, output=output
            )
        except Exception as exc:
            return ToolResult(
                name=call.name,
                tool_call_id=call.tool_call_id,
                output=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )
