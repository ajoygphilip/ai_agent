from pydantic import BaseModel, ConfigDict, Field
from app.ai.models import (
    LLMRequest,
    LLMResponse,
    Usage,
    ToolCall,
    ToolResult,
    Message,
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


class AgentResult(BaseModel):
    response: LLMResponse
    total_steps: int
    total_usage: Usage = Field(default_factory=Usage)


class BaseAgent:
    async def run(self, request: LLMRequest, max_turns: int = 5) -> AgentResult: ...


ContentItem = Union[Message, ToolCall, ToolResult]


class Event(BaseModel):
    """A recorded occurrence during agent execution."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    # execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    contents: List[ContentItem] = Field(default_factory=list)
