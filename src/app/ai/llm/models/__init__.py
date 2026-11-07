from __future__ import annotations
from enum import StrEnum
from typing import Any, AsyncIterator, Callable, Literal, Protocol
from pydantic import BaseModel, Field, ConfigDict
from app.ai.llm.tools import FunctionTool as Tool


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage):
        if not isinstance(other, Usage):
            return NotImplemented

        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class GenerationConfig(BaseModel):
    temperature: float | None = None
    max_tokens: int | None = None


class ToolCall(BaseModel):
    name: str
    arguments: Any
    tool_call_id: str | None = None


class ToolResult(BaseModel):
    name: str
    output: str
    tool_call_id: str | None = None
    is_error: bool = False


class Message(BaseModel):
    role: ChatRole
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class Request(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    messages: list[Message]
    model: str
    tools: list[Tool] = Field(default_factory=list)
    config: GenerationConfig | None = None


class Response(BaseModel):
    text: str
    model: str
    provider: str
    usage: Usage | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    def to_message(self) -> Message:
        return Message(
            role=ChatRole.ASSISTANT,
            content=self.text,
            tool_calls=self.tool_calls,
        )


class LLMClient(Protocol):
    async def generate(self, request: Request) -> Response: ...
