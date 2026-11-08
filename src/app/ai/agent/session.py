from app.ai.agent.models import AgentResult, Budget, Event
from app.ai.agent.agent import AIAgent
from app.ai.models import Message, ChatRole, GenerationConfig, LLMClient


class ChatSession:
    def __init__(
        self,
        agent: AIAgent,
        budget: Budget,
    ):
        self.agent = agent
        self.budget = budget or Budget()
        self.events: list[Event] = []

    async def chat(
        self,
        user_text: str,
    ) -> str:
        user_msg = Message(role=ChatRole.USER, content=user_text)
        event = Event(author="user",contents=[user_msg])
        self.events.append(event)
        result: AgentResult = await self.agent.run(
            events=self.events,
            budget=self.budget,
        )
        assistant_msg = result.response.to_message()
        self.events.append(Event(author="agent",contents=[assistant_msg]))
        return result.response.text
