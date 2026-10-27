import asyncio
from typing import Any
import json
import uuid

from ...models import (
    ChatMessage,
    Request,
    Response,
    ChatRole,
    GenerationConfig,
    Usage,
    Tool,
    ToolCall,
)


class BedrockClient:
    provider = "bedrock"

    def __init__(self, client: Any) -> None:
        self._client = client

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

        inference_config = self._to_inference_config(request.config)
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        # boto3 is synchronous; keep it off the event loop
        response = await asyncio.to_thread(self._client.converse, **kwargs)
        return self._to_chat_response(response, request.model)
