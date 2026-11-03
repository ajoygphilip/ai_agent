import asyncio
import inspect
import json
from pprint import pprint



from app.ai.llm.models import  GenerationConfig
from app.ai.llm.observability import LLMTracer
from app.ai.llm.providers import provider_factory
from app.ai.llm.tools import weather_tool
from app.ai.agent.models import Budget
from app.ai.agent.AIAgent import AIAgent
from app.ai.agent.ChatSession import ChatSession


async def main():
    client = LLMTracer(provider_factory("bedrock"))
    budget = Budget()
    tools = [weather_tool]
    agent = AIAgent(tools=tools)
    config = GenerationConfig(temperature=0.5, max_tokens=300)

    try:
        session = ChatSession(agent=agent, budget=budget)
        agent_respone = await session.chat(
            client=client, user_text="My name is ajoy",config=config
        )
        print(agent_respone)
        agent_respone = await session.chat(
            client=client, user_text="What is my name",config=config
        )
        print(agent_respone)

    except Exception as exc:
        print(f"Request failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
