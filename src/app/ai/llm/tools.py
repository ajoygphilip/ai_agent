from __future__ import annotations
import inspect
from typing import Callable,Any
from abc import ABC,abstractmethod
import inspect

class BaseTool(ABC):

    def __init__(self,func:dict) -> None:
        self._description = func.get("description")
        self._name = func.get("name")
        self.handler = func.get("handler")
        self._input_schema = func.get("input_schema")

    @abstractmethod
    async def execute(self,*args,**kwargs)->Any:
        pass

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.execute(*args,**kwargs)

    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        return self._description
    
    @property
    def input_schema(self):
        return self._input_schema
        
   
    
    @staticmethod
    def get_function_schema(func) -> dict:
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }
        try:
            signature = inspect.signature(func)  # type: ignore
        except Exception as e:
            raise ValueError(f"Unable to convert ")

        params = {}
        required = []
        for param in signature.parameters.values():
            try:
                param_type = type_map[param.annotation]
                params[param.name] = {"type": param_type}

                if param.default == inspect._empty:
                    required.append(param.name)

            except KeyError as e:
                raise KeyError(
                    f"Unsupported type annotation {param.annotation} for param {param.name}"
                ) from e
        return {
            "type": "object",
            "properties": params,
            "required": required,
        }

    @classmethod
    def function_to_tool(cls,func) -> dict:
        return {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": func.__doc__ or "",
                        "input_schema": cls.get_function_schema(func),
                        "handler":func
                    },
                }


class FunctionTool(BaseTool):
    async def execute(self,*args,**kwargs)->Any:
        result =  await self.execute(*args,**kwargs)
        if inspect.iscoroutine(result):
            return await result
        else:
            return result
        
    @classmethod
    def from_function(cls,func:Callable)->BaseTool:
        func_def = cls.function_to_tool(func).get("function")
        if not func_def:
            raise RuntimeError("Tool creation failed")
        return cls(func_def)

def tool(func:Callable):
    if func:
        return FunctionTool.from_function(func)