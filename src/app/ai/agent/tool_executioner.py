from app.ai.llm.models import ToolCall, ToolResult
from app.ai.llm.tools import FunctionTool as Tool
from app.ai.llm.exceptions import ToolCallException
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

        tool = self._tools.get(call.name)
        err = None
        output = None

        if not tool:
            return ToolResult(
                name=call.name,
                tool_call_id=call.tool_call_id,
                output=f"Unknown tool: '{call.name}'",
                is_error=True,
            )
        try:
            print(tool)
            res = await tool(**call.arguments)
            print("res", res)
            output = res if isinstance(res, str) else json.dumps(res)
        except ToolCallException as e:
            err = e

        except Exception as e:
            err = e

        finally:
            if output:
                return ToolResult(
                    name=call.name, tool_call_id=call.tool_call_id, output=output
                )

            elif err:
                return ToolResult(
                    name=call.name,
                    tool_call_id=call.tool_call_id,
                    output=f"{type(err).__name__}: {err}",
                    is_error=True,
                )

            else:
                return ToolResult(
                    name=call.name,
                    tool_call_id=call.tool_call_id,
                    output=f"{type(err).__name__}: {err}",
                    is_error=True,
                )
