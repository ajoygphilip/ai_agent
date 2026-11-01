from pydantic import BaseModel,ConfigDict,Field
from ai.llm.models import LLMClient,Tool,ToolResult,ToolCall,Request,Response,ChatMessage,ChatRole,Usage
import asyncio
import json
import inspect
from ai.llm.observability import LLMTracer
from typing import Any,Protocol
import time

class Budget(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_steps: int = 8
    max_tokens: int = 50_000
    timeout_seconds: float = 60.0


class AgentResult(BaseModel):
    response: Response
    total_steps: int
    total_usage: Usage = Field(default_factory=Usage)


class BaseAgent:
    async def run(self,request: Request, max_turns: int = 5) -> AgentResult:...

class ToolExecutor:
    """Dispatches tool calls safely with error recovery."""
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())
    
    async def execute(self, call: ToolCall) -> ToolResult:
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
            return ToolResult(name=call.name, tool_call_id=call.tool_call_id, output=output)
        except Exception as exc:
            return ToolResult(
                name=call.name,
                tool_call_id=call.tool_call_id,
                output=f"{type(exc).__name__}: {exc}",
                is_error=True,
            )

class AIAgent:
    
    def __init__(
        self,
        tools: list[Tool] | None = None
    ):
        self.executor = ToolExecutor(tools or [])

    async def run(self,client:LLMClient,messages:list[ChatMessage]|None,budget:Budget,max_turns: int = 5) -> AgentResult:
        start_time = time.monotonic()
        messages = messages or []
        tools = self.executor.tools
        total_usage = Usage()
        request = self._build_llm_request(messages,self.executor.tools)

        for step in range(1,max_turns+1):

            elapsed = time.monotonic() - start_time
            remaining_time = budget.timeout_seconds - elapsed
            if remaining_time <= 0:
                raise TimeoutError("Agent exceeded deadline budget.")

            async with asyncio.timeout(remaining_time):
                step_req = request
                response = await client.generate(step_req)

            if response.usage:
                total_usage += response.usage
                if (total_usage.input_tokens + total_usage.output_tokens) > budget.max_tokens:
                    raise RuntimeError("Agent exceeded token budget.")

            if not response.tool_calls:
                return AgentResult(
                    response=response,
                    total_steps=step,
                    total_usage=total_usage,
                )


            messages.append(response.to_message())
            results = await asyncio.gather(
                *(self.executor.execute(call) for call in response.tool_calls)
            )
            messages.append(ChatMessage(role=ChatRole.TOOL, tool_results=list(results)))


        raise RuntimeError(f"Exceeded {max_turns} tool-use turns")

    @classmethod
    def _build_llm_request(cls,messages:list[ChatMessage],tools:list[Tool])->Request:
        request = Request(
                model="amazon.nova-lite-v1:0",
                messages=messages,
                tools=tools,
            )
        return request


class ChatSession:
    def __init__(self, agent: AIAgent,budget:Budget,system_prompt:str="You are a helpful AI agent. "):
        self.agent = agent
        self.budget = budget or Budget()
        self.history: list[ChatMessage] = [
            ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)
        ]
        if not self.history:
            self.history.append(
                ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)
            )

    async def chat(self, user_text: str,client:LLMClient) -> str:
        user_msg = ChatMessage(role=ChatRole.USER, content=user_text)
        self.history.append(user_msg)
        result: AgentResult = await self.agent.run(client=client,messages=self.history,budget=self.budget)
        assistant_msg = result.response.to_message()
        self.history.append(assistant_msg)
        return result.response.text