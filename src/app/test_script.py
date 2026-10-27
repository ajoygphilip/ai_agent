import asyncio
import json
import sys
from pprint import pprint

from ai.llm.models import (
    ChatMessage,
    Request,
    ChatRole,
    GenerationConfig,
    ToolResult,
    Tool,
    ToolCall,
    Response,
)
import inspect
from ai.llm.providers import provider_factory
from ai.llm.tools import weather_tool


async def execute_tool(tools: dict[str, Tool], call: ToolCall) -> ToolResult:
    tool = tools.get(call.name)
    if tool is None:
        return ToolResult(
            name=call.name,
            tool_call_id=call.tool_call_id,
            output=f"Unknown tool: {call.name}",
            is_error=True,
        )
    try:
        result = tool.handler(**call.arguments)
        if inspect.isawaitable(result):
            result = await result
        output = result if isinstance(result, str) else json.dumps(result)
        return ToolResult(name=call.name, tool_call_id=call.tool_call_id, output=output)
    except Exception as exc:
        # Feed the error back so the model can recover or explain
        return ToolResult(
            name=call.name,
            tool_call_id=call.tool_call_id,
            output=f"{type(exc).__name__}: {exc}",
            is_error=True,
        )


async def run_with_tools(client, request: Request, max_turns: int = 5) -> Response:
    tools = {t.name: t for t in request.tools}

    for turn in range(max_turns):
        response = await client.generate(request)
        print(response, "\n\n")
        print(
            f"[turn {turn}] finish={response.finish_reason} tool_calls={len(response.tool_calls)}"
        )

        if not response.tool_calls:
            return response

        # 1. echo the assistant turn (with toolUse blocks)
        request.messages.append(response.to_message())

        # 2. run all tools, then send ALL results in one message
        results = await asyncio.gather(
            *(execute_tool(tools, call) for call in response.tool_calls)
        )
        print(results)
        request.messages.append(
            ChatMessage(role=ChatRole.TOOL, tool_results=list(results))
        )

    raise RuntimeError(f"Exceeded {max_turns} tool-use turns")


async def main():
    client = provider_factory("bedrock")

    request = Request(
        model="amazon.nova-lite-v1:0",
        messages=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                content="You are a concise, helpful assistant.",
            ),
            ChatMessage(
                role=ChatRole.USER,
                content="What is the weather at trivandrum in celcius",
            ),
        ],
        tools=[weather_tool],
        config=GenerationConfig(
            temperature=0.2,
            max_tokens=300,
        ),
    )

    print(f"Model: {request.model}")
    print("Sending request...\n")

    try:
        response = await run_with_tools(client, request)

        print("Response")
        print("-" * 60)
        print("Provider:", response.provider)
        print("Model:", response.model)
        print("Finish reason:", response.finish_reason)
        print("Usage:")
        pprint(response.usage.model_dump() if response.usage else None)
        print("\nText:")
        print(response.text)

        for call in response.tool_calls:
            print(call)

    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
