from app.ai.agent.models import AgentResult, Budget, Event
from app.ai.agent.tool_executioner import ToolExecutor
from app.ai.models import (
    Message,
    ChatRole,
    GenerationConfig,
    LLMClient,
    LLMRequest,
    LLMResponse,
    ToolCall,
    ToolResult,
    Usage,
)
from app.ai.llm.tools import BaseTool as Tool


import asyncio
import time


class AIAgent:
    name = "general_agent"

    def __init__(
        self,
        client: LLMClient,
        model: str,
        tools: list[Tool] | None = None,
        config: GenerationConfig | None = None,
    ):
        self.client = client
        self.tools = tools or []
        self.executor = ToolExecutor(self.tools)
        self.model = model
        self.config = config

    async def run(
        self,
        events: list[Event] | None,
        budget: Budget,
        max_turns: int = 5,
    ) -> AgentResult:
        start_time = time.monotonic()
        events = events or []
        total_usage = Usage()
        new_events: list[Event] = []

        for step in range(1, max_turns + 1):
            elapsed = time.monotonic() - start_time
            remaining_time = budget.timeout_seconds - elapsed
            if remaining_time <= 0:
                raise TimeoutError("Agent exceeded deadline budget.")

            async with asyncio.timeout(remaining_time):
                request = self._build_llm_request(
                    self.process_events(events),
                    self.executor.tools,
                    config=self.config,
                    model=self.model,
                )
                response = await self._think(request, self.client)

            if response.usage:
                total_usage += response.usage
                if (
                    total_usage.input_tokens + total_usage.output_tokens
                ) > budget.max_tokens:
                    raise RuntimeError("Agent exceeded token budget.")

            if response.tool_calls:
                print("running tool calll")
                tool_call_event = Event(
                    author="agent", contents=[response.to_message()]
                )
                new_events.append(tool_call_event)
                events.append(tool_call_event)

                tool_results = await self._act(response)
                event = Event(
                    author="tool",
                    contents=[
                        Message(role=ChatRole.TOOL, tool_results=list(tool_results))
                    ],
                )
                events.append(event)
                new_events.append(event)
            else:
                new_events.append(
                    Event(author="agent", contents=[response.to_message()])
                )
                return AgentResult(
                    response=response,
                    total_steps=step,
                    total_usage=total_usage,
                    events=new_events,
                )

        raise RuntimeError(f"Exceeded {max_turns} tool-use turns")

    async def _act(self, response):
        results = await asyncio.gather(
            *(self.executor.execute(call) for call in response.tool_calls)
        )
        return results

    async def _think(self, request: LLMRequest, client: LLMClient) -> LLMResponse:
        return await client.generate(request)

    def process_events(self, events: list[Event]) -> list[Message]:
        ret = []
        for event in events:
            for content in event.contents:
                if isinstance(content, Message):
                    ret.append(content)
                elif isinstance(content, ToolResult):
                    ret.append(Message(role=ChatRole.TOOL, tool_results=[content]))
                elif isinstance(content, ToolCall):
                    ret.append(
                        Message(role=ChatRole.ASSISTANT, tool_calls=[content])
                    )
        return ret


    def _tools_to_spec(self):
        return [tool.spec for tool in self.tools]
    
    @classmethod
    def _build_llm_request(
        cls,
        messages: list[Message],
        tools: list[Tool],
        model: str,
        config: GenerationConfig | None,
    ) -> LLMRequest:
        request = LLMRequest(model=model, messages=messages, tools=Tool.tools_to_spec(tools), config=config)
        return request
