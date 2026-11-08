from app.ai.agent.models import AgentResult, Budget, Event,Usage
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
        self.usage = Usage()
        self.steps:int = 0

    async def chat(
        self,
        user_text: str,
    ):
        user_msg = Message(role=ChatRole.USER, content=user_text)
        event = Event(author="user", contents=[user_msg])
        self.events.append(event)
        result: AgentResult = await self.agent.run(
            events=self.events.copy(),
            budget=self.budget,
        )
        self.update_from_result(result)

    def update_from_result(self,result:AgentResult):
        self.events.extend(result.events)
        self.usage+=result.total_usage
        self.steps+= result.total_steps
