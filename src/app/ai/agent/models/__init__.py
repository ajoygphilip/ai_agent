from pydantic import BaseModel, ConfigDict, Field
from app.ai.models import (
    LLMRequest,
    LLMResponse,
    Usage,
    ToolCall,
    ToolResult,
    Message,
    ChatRole,
)
from app.ai.llm.observability import LLMTracer
from typing import Any, Protocol
import uuid
from datetime import datetime
from typing import List, Union


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_steps: int = 8
    max_tokens: int = 50_000
    timeout_seconds: float = 60.0


ContentItem = Union[Message, ToolCall, ToolResult]


class Event(BaseModel):
    """A recorded occurrence during agent execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    contents: List[ContentItem] = Field(default_factory=list)

    def has_final_response(self) -> bool:
        has_tool_calls = any(isinstance(c, ToolCall) for c in self.contents)
        has_tool_results = any(isinstance(c, ToolResult) for c in self.contents)
        return not has_tool_calls and not has_tool_results

    def extract_final_result(self) -> str | None:
        for item in self.contents:
            if isinstance(item, Message) and item.role == ChatRole.ASSISTANT:
                return item.content
        return None


class AgentResult(BaseModel):
    response: LLMResponse
    total_steps: int
    total_usage: Usage = Field(default_factory=Usage)
    events: List[Event]
