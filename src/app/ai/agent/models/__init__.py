from pydantic import BaseModel, ConfigDict, Field
from ai.llm.models import (
    Request,
    Response,
    Usage
)
from ai.llm.observability import LLMTracer
from typing import Any, Protocol


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
    async def run(self, request: Request, max_turns: int = 5) -> AgentResult: ...

