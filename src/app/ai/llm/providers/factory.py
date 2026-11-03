import boto3
from app.config import settings

from ..exceptions import InvalidLLMProviderException
from ..models import LLMClient


def provider_factory(provider: str) -> LLMClient:
    match provider:
        case "bedrock":
            from app.ai.llm.providers.bedrock import BedrockClient

            return BedrockClient.from_settings(settings)
        case _:
            raise InvalidLLMProviderException(f"Unknown provider: {provider}")
