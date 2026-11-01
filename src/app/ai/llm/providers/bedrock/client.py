from __future__ import annotations
import asyncio
from typing import Any
import json
import uuid
import boto3
from botocore.client import BaseClient
from botocore.exceptions import (
    ClientError,
    ReadTimeoutError,
    ConnectTimeoutError,
    EndpointConnectionError,
)
from botocore.config import Config
from ...models import (
    ChatMessage,
    Request,
    Response,
    ChatRole,
    GenerationConfig,
    Usage,
    ToolCall,
)
from ...exceptions import *

RETRYABLE = {
    "InternalServerException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
}
BEDROCK_ERROR_MAP = {
    "AccessDeniedException": InvalidLLMProviderException,
    "ValidationException": InvalidRequest,
    "ResourceNotFoundException": InvalidLLMProviderException,
    "ThrottlingException": RateLimited,
    "ModelTimeoutException": LLMTimeout,
    "InternalServerException": ProviderUnavailable,
    "ServiceUnavailableException": ProviderUnavailable,
    "ModelNotReadyException": ProviderUnavailable,
    "ModelErrorException": ProviderUnavailable,
}


class BedrockClient:
    provider = "bedrock"

    def __init__(self, client: BaseClient) -> None:
        self._client = client
        self._config = GenerationConfig(temperature=0.5, max_tokens=300)

    @classmethod
    def from_settings(cls, settings) -> BedrockClient:
        # print(settings)
        cfg = Config(
            region_name=settings.AWS_REGION,
            retries={"mode": "adaptive", "max_attempts": 4},
            # connect_timeout=5,
            read_timeout=120,
            max_pool_connections=50,
        )

        client = boto3.client(
            "bedrock-runtime",
            config=cfg,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        return cls(client)

    @staticmethod
    def _to_provider_messages(
        messages: list[ChatMessage],
    ) -> tuple[list[dict], list[dict]]:
        system: list[dict] = []
        items: list[dict] = []

        for msg in messages:
            if msg.role == ChatRole.SYSTEM:
                system.append({"text": msg.content})
            elif msg.role == ChatRole.TOOL:
                items.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": r.tool_call_id,
                                    "content": [{"text": r.output}],
                                    "status": "error" if r.is_error else "success",
                                }
                            }
                            for r in msg.tool_results
                        ],
                    }
                )
            else:
                blocks: list[dict] = []
                if msg.content:  # Bedrock rejects empty text blocks
                    blocks.append({"text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tc.tool_call_id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        }
                    )
                items.append({"role": msg.role.value, "content": blocks})

        return system, items

    @staticmethod
    def _to_inference_config(config: GenerationConfig) -> dict:
        mapping = {"temperature": config.temperature, "maxTokens": config.max_tokens}
        return {k: v for k, v in mapping.items() if v is not None}

    def _to_provider_tools(self, request: Request) -> dict | None:
        if not request.tools:
            return None

        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {"json": tool.input_schema},
                    }
                }
                for tool in request.tools
            ]
        }

    def _to_chat_response(self, raw_response: dict, model: str) -> Response:
        # print(raw_response)
        blocks = raw_response["output"]["message"]["content"]

        text = ""
        tool_calls = []

        for block in blocks:
            if "text" in block:
                text += block["text"]
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        tool_call_id=tu.get("toolUseId", str(uuid.uuid4())),
                        name=tu["name"],
                        arguments=tu.get("input", {}),
                    )
                )

        usage = None
        raw = raw_response.get("usage")
        if raw:
            usage = Usage(
                input_tokens=raw.get("inputTokens", 0),
                output_tokens=raw.get("outputTokens", 0),
            )

        return Response(
            text=text,
            model=model,
            provider=self.provider,
            usage=usage,
            finish_reason=raw_response.get("stopReason"),
            tool_calls=tool_calls,
        )

    async def generate(self, request: Request) -> Response:
        # print("from client generate",request,sep="\n")
        system, messages = self._to_provider_messages(request.messages)
        tool_config = self._to_provider_tools(request)

        kwargs: dict[str, Any] = {"modelId": request.model, "messages": messages}
        if system:
            kwargs["system"] = system
        if tool_config:
            kwargs["toolConfig"] = tool_config

        inference_config = self._to_inference_config(self._config)
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        # boto3 is synchronous; keep it off the event loop
        try:
            response = await asyncio.to_thread(self._client.converse, **kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            exception_class = BEDROCK_ERROR_MAP.get(code, LLMException)
            raise exception_class(str(e)) from e
        except (ReadTimeoutError, ConnectTimeoutError) as e:
            raise LLMTimeout(str(e)) from e
        except EndpointConnectionError as e:
            raise ProviderUnavailable(str(e)) from e
        except Exception as e:
            raise LLMException(str(e)) from e

        return self._to_chat_response(response, request.model)
