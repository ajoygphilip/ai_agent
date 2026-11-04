from app.ai.agent.models import AgentResult, Budget
from app.ai.agent.tool_executioner import ToolExecutor
from app.ai.llm.models import (
    ChatMessage,
    ChatRole,
    GenerationConfig,
    LLMClient,
    Request,
    Response,
    Tool,
    Usage,
)


import asyncio
import time


class AIAgent:
    def __init__(self, tools: list[Tool] | None = None):
        self.executor = ToolExecutor(tools or [])

    async def run(
        self,
        client: LLMClient,
        messages: list[ChatMessage] | None,
        budget: Budget,
        config: GenerationConfig | None,
        max_turns: int = 5,
    ) -> AgentResult:
        start_time = time.monotonic()
        messages = messages or []
        total_usage = Usage()

        for step in range(1, max_turns + 1):
            elapsed = time.monotonic() - start_time
            remaining_time = budget.timeout_seconds - elapsed
            if remaining_time <= 0:
                raise TimeoutError("Agent exceeded deadline budget.")

            async with asyncio.timeout(remaining_time):
                request = self._build_llm_request(
                    messages, self.executor.tools, config=config
                )
                response = await self._think(request, client)

            if response.usage:
                total_usage += response.usage
                if (
                    total_usage.input_tokens + total_usage.output_tokens
                ) > budget.max_tokens:
                    raise RuntimeError("Agent exceeded token budget.")

            if not response.tool_calls:
                return AgentResult(
                    response=response,
                    total_steps=step,
                    total_usage=total_usage,
                )

            messages.append(response.to_message())
            tool_results = await self._act(response)
            messages.append(
                ChatMessage(role=ChatRole.TOOL, tool_results=list(tool_results))
            )

        raise RuntimeError(f"Exceeded {max_turns} tool-use turns")

    async def _act(self, response):
        results = await asyncio.gather(
            *(self.executor.execute(call) for call in response.tool_calls)
        )
        return results

    async def _think(self, request: Request, client: LLMClient) -> Response:
        return await client.generate(request)

    @classmethod
    def _build_llm_request(
        cls,
        messages: list[ChatMessage],
        tools: list[Tool],
        config: GenerationConfig | None,
    ) -> Request:
        request = Request(
            model="amazon.nova-lite-v1:0", messages=messages, tools=tools, config=config
        )
        return request
