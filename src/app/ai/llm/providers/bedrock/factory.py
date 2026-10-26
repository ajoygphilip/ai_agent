import boto3
from config import settings
from .client import BedrockClient


def build_bedrock_client() -> BedrockClient:
    boto_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.BEDROCK_AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.BEDROCK_AWS_SECRET_ACCESS_KEY,
    )
    return BedrockClient(boto_client)
