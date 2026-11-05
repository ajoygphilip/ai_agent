from app.ai.agent.models import AgentResult, Budget
from app.ai.agent.agent import AIAgent
from app.ai.llm.models import ChatMessage, ChatRole, GenerationConfig, LLMClient


class ChatSession:
    def __init__(
        self,
        agent: AIAgent,
        budget: Budget,
        system_prompt: str = "You are a helpful AI agent. ",
    ):
        self.agent = agent
        self.budget = budget or Budget()
        self.history: list[ChatMessage] = [
            ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)
        ]
        if not self.history:
            self.history.append(
                ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)
            )

    async def chat(
        self, user_text: str, client: LLMClient, model:str,config: GenerationConfig | None = None,
    ) -> str:
        user_msg = ChatMessage(role=ChatRole.USER, content=user_text)
        self.history.append(user_msg)
        result: AgentResult = await self.agent.run(
            client=client, messages=self.history, budget=self.budget, config=config,model=model
        )
        assistant_msg = result.response.to_message()
        self.history.append(assistant_msg)
        return result.response.text
