import boto3
from config import settings
from ..models import ChatModel
from ..exceptions import InvalidLLMProviderException


def provider_factory(provider: str) -> ChatModel:
    match provider:
        case "bedrock":
            from .bedrock.client import BedrockClient

            return BedrockClient.from_settings(settings)
        case _:
            raise InvalidLLMProviderException(f"Unknown provider: {provider}")
