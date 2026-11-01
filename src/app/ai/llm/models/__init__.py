from __future__ import annotations
from enum import StrEnum
from typing import Any, AsyncIterator, Callable, Literal, Protocol
from pydantic import BaseModel, Field


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


class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable = Field(exclude=True)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    name: str
    arguments: Any
    tool_call_id: str | None = None


class ToolResult(BaseModel):
    name: str
    output: str
    tool_call_id: str | None = None
    is_error: bool = False


class ChatMessage(BaseModel):
    role: ChatRole
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)


class Request(BaseModel):
    messages: list[ChatMessage]
    model: str
    tools: list[Tool] = Field(default_factory=list)
    config: GenerationConfig|None = None

class Response(BaseModel):
    text: str
    model: str
    provider: str
    usage: Usage | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    def to_message(self) -> ChatMessage:
        return ChatMessage(
            role=ChatRole.ASSISTANT,
            content=self.text,
            tool_calls=self.tool_calls,
        )


class LLMClient(Protocol):
    async def generate(self, request: Request) -> Response: ...


__all__ = [
    "ChatRole",
    "ChatMessage",
    "Usage",
    "Response",
    "GenerationConfig",
    "Tool",
    "Request",
    "LLMClient",
]
