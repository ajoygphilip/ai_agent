import inspect
from typing import Callable


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


def format_tool_definition(name: str, description: str, input_schema: dict,handler:Callable) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "input_schema": input_schema,
            "handler":handler
        },
    }


def function_to_tool(func) -> dict:
    return format_tool_definition(
        func.__name__, func.__doc__ or "", get_function_schema(func),func
    )



