from app.ai.agent.models import AgentResult, Budget,Event
from app.ai.agent.tool_executioner import ToolExecutor
from app.ai.models import (
    Message,
    ChatRole,
    GenerationConfig,
    LLMClient,
    LLMRequest,
    LLMResponse,ToolCall,ToolResult,
    Tool,
    Usage,
)


import asyncio
import time


class AIAgent:
    name = "general_agent"

    def __init__(self, client: LLMClient,model: str,tools: list[Tool] | None = None,config: GenerationConfig | None = None):
        self.client= client
        self.executor = ToolExecutor(tools or [])
        self.tools = tools
        self.model= model
        self.config=config

    async def run(
        self,
        events: list[Event] | None,
        budget: Budget,
        max_turns: int = 5,
    ) -> AgentResult:
        start_time = time.monotonic()
        events = events or []
        total_usage = Usage()

        for step in range(1, max_turns + 1):
            elapsed = time.monotonic() - start_time
            remaining_time = budget.timeout_seconds - elapsed
            if remaining_time <= 0:
                raise TimeoutError("Agent exceeded deadline budget.")

            async with asyncio.timeout(remaining_time):
                request = self._build_llm_request(
                    self.events_to_messages(events), self.executor.tools, config=self.config, model=self.model
                )
                response = await self._think(request, self.client)

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

            events.append(Event(author="agent",contents=[response.to_message()]))
            tool_results = await self._act(response)
            events.append(
               Event(author="agent",contents= [Message(role=ChatRole.TOOL, tool_results=list(tool_results))])
            )

        raise RuntimeError(f"Exceeded {max_turns} tool-use turns")

    async def _act(self, response):
        results = await asyncio.gather(
            *(self.executor.execute(call) for call in response.tool_calls)
        )
        return results

    async def _think(self, request: LLMRequest, client: LLMClient) -> LLMResponse:
        return await client.generate(request)

    def events_to_messages(self,events:list[Event]) -> list[Message]:
        messages = []
        for event in events:
            for content in event.contents:
                if isinstance(content, Message):
                    messages.append(content)
                elif isinstance(content, ToolResult):
                    messages.append(Message(role=ChatRole.TOOL, tool_results=[content]))
                elif isinstance(content, ToolCall):
                    messages.append(Message(role=ChatRole.ASSISTANT, tool_calls=[content]))
        return messages

    @classmethod
    def _build_llm_request(
        cls,
        messages: list[Message],
        tools: list[Tool],
        model: str,
        config: GenerationConfig | None,
    ) -> LLMRequest:
        request = LLMRequest(model=model, messages=messages, tools=tools, config=config)
        return request
