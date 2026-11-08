import asyncio
import inspect
import json
from pprint import pprint


from app.ai.models import GenerationConfig
from app.ai.llm.observability import LLMTracer
from app.ai.llm.providers import provider_factory
from app.ai.agent.models import Budget
from app.ai.agent.agent import AIAgent
from app.ai.agent.session import ChatSession

# from app.ai.llm.models import Tool
from app.ai.llm.tools import tool


@tool
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Return mock weather data for a location."""
    print("fetching weather")
    return {
        "location": location,
        "temperature": 28,
        "unit": unit,
        "condition": "partly cloudy",
    }


async def main():

    client = LLMTracer(provider_factory("bedrock"))
    budget = Budget()
    tools = [get_weather]
    agent = AIAgent(tools=tools)  # type: ignore
    config = GenerationConfig(temperature=0.5, max_tokens=300)

    try:
        session = ChatSession(agent=agent, budget=budget)
        agent_respone = await session.chat(
            client=client,
            user_text="Can you get me the weather at Trivandrum?",
            config=config,
            model="amazon.nova-lite-v1:0",
        )
        print(agent_respone)

    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
    pass
