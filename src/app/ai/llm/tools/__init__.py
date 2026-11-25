from __future__ import annotations
import inspect
from typing import Callable, Any
from abc import ABC, abstractmethod
import inspect
from app.ai.exceptions import ToolCallException
from app.ai.models import ToolSpec
from pydantic import create_model, ValidationError, BaseModel
import asyncio

class BaseTool(ABC):
    def __init__(
        self,
        handler: Callable,
        name: str,
        description: str = "",
        input_schema: dict | None = None,
    ) -> None:
        self._description = description
        self._name = name
        self.handler = handler
        self._input_schema = input_schema

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        pass

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await self.execute(*args, **kwargs)
        except Exception as e:
            raise ToolCallException(str(e)) from e

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def input_schema(self):
        return self._input_schema

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_schema or {},
        )

    @classmethod
    def tools_to_spec(cls, tools: list[BaseTool]) -> list[ToolSpec]:
        return [t.spec for t in tools]


class FunctionTool(BaseTool):
    def __init__(
        self,
        handler: Callable[..., Any],
        name: str,
        description: str = "",
        input_schema: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(handler, name, description, input_schema)
        self.model: type[BaseModel] | None = kwargs.get("model")

    async def execute(self, *args, **kwargs) -> Any:
        try:

            call_kwargs = (
                self.model(**kwargs).model_dump() if self.model else kwargs
            )
            if self._is_async_handler():
                result = await self.handler(**call_kwargs)
            else:
                result = await asyncio.to_thread(self.handler, **call_kwargs)
                if inspect.isawaitable(result):
                    result = await result
            return result

        except ValidationError as e:
            raise ToolCallException(f"Invalid tool arguments: {e.errors()}") from e

    @staticmethod
    def get_function_fields(func) -> dict:
        sig = inspect.signature(func)
        fields = {
            param.name: (
                param.annotation if param.annotation != inspect._empty else Any,
                param.default if param.default != inspect._empty else ...,
            )
            for param in sig.parameters.values()
        }

        return fields

    @classmethod
    def function_to_tool(cls, func) -> dict:
        fields = cls.get_function_fields(func)
        model = create_model(f"{func.__name__}_args", **fields)  # type: ignore

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "input_schema": model.model_json_schema(),  # type: ignore
                "handler": func,
                "model": model,
            },
        }

    @classmethod
    def from_function(cls, func: Callable) -> BaseTool:
        func_def = cls.function_to_tool(func).get("function")
        if not func_def:
            raise RuntimeError("Tool creation failed")
        return cls(**func_def)

    def _is_async_handler(self) -> bool:
        return inspect.iscoroutinefunction(self.handler) or inspect.iscoroutinefunction(
            getattr(self.handler, "__call__", None)
        )


def tool(func: Callable):
    if func:
        return FunctionTool.from_function(func)


class StructuredOutputTool(BaseTool):
    """Stateless terminal tool used to validate and deliver structured output conforming to a Pydantic schema."""

    def __init__(self, schema: type[BaseModel], name: str = "final_answer") -> None:
        super().__init__(
            handler=self._noop,
            name=name,
            description=(
                f"Submit the final response conforming strictly to the {schema.__name__} schema. "
                "Call this tool when you are ready to provide the final structured answer."
            ),
            input_schema=schema.model_json_schema(),
        )
        self.schema = schema

    def _noop(self, *args, **kwargs):
        pass

    async def execute(self, *args, **kwargs) -> Any:
        try:
            validated = self.schema.model_validate(kwargs)
            return validated.model_dump_json()
        except ValidationError as e:
            raise ToolCallException(
                f"Validation error for {self.schema.__name__}: {e}. "
                "Please correct your fields to strictly match the required schema and call again."
            ) from e
