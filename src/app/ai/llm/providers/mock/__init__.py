from app.ai.models import LLMResponse,LLMRequest

class MockLLMClient:
    def __init__(self, responses: list[LLMResponse]|None=None):
        self.responses = responses or []
        self.calls: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if not self.responses:
            return LLMResponse(text="No more responses", model="mock", provider="mock")
        return self.responses.pop(0)