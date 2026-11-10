import asyncio

from tavily import TavilyClient
from app.ai.models import GenerationConfig
from app.ai.llm.observability import LLMTracer
from app.ai.llm.providers import provider_factory
from app.ai.agent.models import Budget
from app.ai.agent.agent import AIAgent
from app.ai.agent.session import ChatSession

from app.ai.llm.tools import tool
from app.config import settings


@tool
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Return mock weather data for a location."""
    return {
        "location": location,
        "temperature": 28,
        "unit": unit,
        "condition": "partly cloudy",
    }


@tool
def web_search(query: str):
    """Search the web for latest details"""
    print("searching web", query)
    client = TavilyClient(settings.TAVILY_API_KEY)
    response = client.search(
        search_depth="basic", query=query, include_images=False, max_results=10
    )
    return response


async def main():
    client = LLMTracer(provider_factory("bedrock"))

    budget = Budget()
    tools = [get_weather, web_search]

    config = GenerationConfig(
        temperature=0.5,
        max_tokens=300,
    )

    agent = AIAgent(
        tools=tools,
        client=client,
        config=config,
        model="amazon.nova-lite-v1:0",
    )

    session = ChatSession(
        agent=agent,
        budget=budget,
    )

    print("Agent started. Type 'exit' to quit.\n")

    while True:
        try:
            user_text = await asyncio.to_thread(input, "You: ")

            user_text = user_text.strip()

            if not user_text:
                continue

            if user_text.lower() in {"exit", "quit", "q"}:
                print("Agent stopped.")
                break

            await session.chat(user_text=user_text)

            print("\nAgent:")
            print(session.events[-1].extract_final_result())

            print("\nUsage:")
            print(session.usage)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nAgent interrupted.")
            break

        except EOFError:
            break

        except Exception as exc:
            print(f"Request failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
