from app.ai.agent.models import AgentResult, Budget, Event
from app.ai.agent.tool_executioner import ToolExecutor
from app.ai.models import (
    Message,
    ChatRole,
    GenerationConfig,
    LLMClient,
    LLMRequest,
    LLMResponse,
    Usage,
)
from app.ai.llm.tools import BaseTool as Tool, StructuredOutputTool, tool
from typing import Callable

import asyncio
import time
from pydantic import BaseModel


class AIAgent:
    def __init__(
        self,
        client: LLMClient,
        model: str,
        tools: list[Tool] | None = None,
        config: GenerationConfig | None = None,
        instructions: list[Callable[[], str] | str] | None = None,
        name: str | None = None,
        prompt: str | None = None,
        output_schema: type[BaseModel] | None = None,
    ):
        self.name = name or "general_agent"
        self.prompt = prompt or "You are a useful agent"
        self.client = client
        self.tools = tools if tools is not None else []
        self.executor = ToolExecutor(self.tools)
        self.model = model
        self.config = config
        self.instructions = instructions if instructions is not None else []
        self.output_schema = output_schema
        self.output_tool = None
        self._setup_output_tools()

    def _setup_output_tools(self):
        if self.output_schema:
            output_tool = StructuredOutputTool(schema=self.output_schema)
            self.output_tool = output_tool
            if output_tool:
                self.executor.add_tool(output_tool)

    async def run(
        self,
        events: list[Event] | None,
        budget: Budget,
    ) -> AgentResult:
        start_time = time.monotonic()
        events = list(events or [])
        usage = Usage()
        new_events: list[Event] = []

        for step in range(1, budget.max_steps + 1):
            elapsed = time.monotonic() - start_time
            remaining_time = budget.timeout_seconds - elapsed
            if remaining_time <= 0:
                raise TimeoutError("Agent exceeded deadline budget.")

            # async with asyncio.timeout(remaining_time):
            request = self._build_llm_request(self.process_events(events))

            response = await self._think(request, self.client)

            if response.usage:
                usage += response.usage
                if (usage.input_tokens + usage.output_tokens) > budget.max_tokens:
                    raise RuntimeError("Agent exceeded token budget.")

            if response.tool_calls:
                # print("tool call", response.tool_calls[0].name)
                tool_call_event = Event(
                    author=self.name, contents=[response.to_message()]
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

                # If an output schema was expected and validated successfully:
                structured_output = self._extract_structured_output(tool_results)
                if structured_output is not None:
                    return AgentResult(
                        response=response,
                        total_steps=step,
                        total_usage=usage,
                        events=new_events,
                        structured_output=structured_output,
                    )
            else:
                # If output schema was required, but the LLM provided text instead of calling output_tool:
                if self._should_retry_missing_output(response):
                    retry_event = self._create_output_retry_event()
                    events.append(retry_event)
                    new_events.append(retry_event)
                    continue

                new_events.append(
                    Event(author=self.name, contents=[response.to_message()])
                )
                return AgentResult(
                    response=response,
                    total_steps=step,
                    total_usage=usage,
                    events=new_events,
                )

        if self.output_schema:
            raise RuntimeError(
                f"Agent failed to produce valid structured output for {self.output_schema.__name__} "
                f"within {budget.max_steps} steps."
            )
        raise RuntimeError(f"Exceeded {budget.max_steps} tool-use turns")

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
                ret.append(content)
        return ret

    def _tools_to_spec(self):
        return [tool.spec for tool in self.tools]

    def _extract_structured_output(self, tool_results: list) -> BaseModel | None:
        """Extract and parse validated BaseModel from tool results if output_tool succeeded."""
        if not self.output_tool or not self.output_schema:
            return None

        for result in tool_results:
            if result.name == self.output_tool.name and not result.is_error:
                return self.output_schema.model_validate_json(result.output)
        return None

    def _should_retry_missing_output(self, response: LLMResponse) -> bool:
        """Determine if a retry is required because LLM replied with text without calling output_tool."""
        return bool(self.output_schema and self.output_tool and not response.tool_calls)

    def _create_output_retry_event(self) -> Event:
        """Create a user reminder event instructing the LLM to call the required output tool."""
        assert self.output_tool and self.output_schema
        return Event(
            author="user",
            contents=[
                Message(
                    role=ChatRole.USER,
                    content=(
                        f"Your final response must be delivered by calling the '{self.output_tool.name}' tool "
                        f"conforming strictly to the {self.output_schema.__name__} schema. "
                        f"Please call '{self.output_tool.name}'."
                    ),
                )
            ],
        )

    def get_instructions(self) -> list[str]:
        """Resolve instructions dynamically if callable, or return static string."""
        instructions: list[str] = []
        if self.prompt:
            instructions.append(self.prompt)
        if self.output_tool:
            instructions.append(
                f"When ready to deliver the final response, you MUST call the '{self.output_tool.name}' tool "
                f"with arguments conforming strictly to the {self.output_tool.schema.__name__} schema."
            )

        for inst in self.instructions:
            if callable(inst):
                instructions.append(inst())
            elif isinstance(inst, str):
                instructions.append(inst)
        return instructions

    def _build_llm_request(
        self,
        messages: list[Message],
    ) -> LLMRequest:

        payload_messages = []
        instructions = self.get_instructions()
        for prompt in instructions:
            payload_messages.append(Message(role=ChatRole.SYSTEM, content=prompt))

        payload_messages.extend(messages)
        request = LLMRequest(
            model=self.model,
            messages=payload_messages,
            tools=Tool.tools_to_spec(self.executor.tools),
            config=self.config,
        )
        return request
