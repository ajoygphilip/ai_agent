from app.ai.llm.tools import tool
from app.config import settings
from tavily import TavilyClient,AsyncTavilyClient
import asyncio


@tool
async def web_search(query: str):
    """Search the web for latest details"""
    # print("searching web", query)
    client = AsyncTavilyClient(settings.TAVILY_API_KEY)
    responses = await asyncio.gather(
        *(client.search(q) for q in [query]),
        return_exceptions=True,
    )
    
    return responses[0]


import ast
import operator
import math


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to calculate.
        Examples:
            "10 + 20"
            "(100 / 5) * 2"
            "2 ** 10"
            "sqrt(144)"
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    functions = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "ceil": math.ceil,
        "floor": math.floor,
        "abs": abs,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent too large")

            return operators[type(node.op)](left, right)

        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Invalid function")

            func = functions.get(node.func.id)
            if func is None:
                raise ValueError(f"Unsupported function: {node.func.id}")

            if node.keywords:
                raise ValueError("Keyword arguments are not supported")

            return func(*(evaluate(arg) for arg in node.args))

        raise ValueError("Invalid mathematical expression")

    try:
        tree = ast.parse(expression, mode="eval")
        result = evaluate(tree.body)

        if isinstance(result, complex) or not math.isfinite(result):  # type: ignore
            raise ValueError("Result is not a finite real number")

        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"
