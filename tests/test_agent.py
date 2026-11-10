import pytest
from app.ai.models import LLMResponse, ToolCall, Usage, ChatRole
from app.ai.llm.providers.mock import MockLLMClient
from app.ai.llm.tools import tool
from app.ai.agent.agent import AIAgent
from app.ai.agent.models import Budget
from app.ai.agent.session import ChatSession


# 1. Define a test tool with type annotations
@tool
def calculate_tax(amount: float, rate: float = 0.1) -> float:
    """Calculate tax for a given amount."""
    return amount * rate


@pytest.mark.asyncio
async def test_agent_single_turn_no_tools():
    """Verify that the agent handles simple text conversations without tools."""
    mock_client = MockLLMClient(
        responses=[
            LLMResponse(
                text="Hello! How can I help you?",
                model="mock-model",
                provider="mock",
                usage=Usage(input_tokens=10, output_tokens=5),
            )
        ]
    )

    agent = AIAgent(client=mock_client, model="mock-model")
    session = ChatSession(agent=agent, budget=Budget())

    await session.chat("Hello")

    # Assertions
    assert len(mock_client.calls) == 1
    assert session.usage.input_tokens == 10
    assert session.usage.output_tokens == 5
    assert session.steps == 1
    assert session.events[-1].contents[-1].content == "Hello! How can I help you?"  # type: ignore


@pytest.mark.asyncio
async def test_agent_react_tool_calling_loop():
    """Verify multi-turn tool calling:
    Turn 1: LLM requests tool execution.
    Turn 2: Agent executes tool, feeds result back, and LLM generates final answer.
    """
    mock_client = MockLLMClient(
        responses=[
            # Step 1: LLM decides to call the calculate_tax tool
            LLMResponse(
                text="",
                model="mock-model",
                provider="mock",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_1",
                        name="calculate_tax",
                        arguments={"amount": 100.0, "rate": 0.2},
                    )
                ],
                usage=Usage(input_tokens=20, output_tokens=10),
            ),
            # Step 2: LLM receives tool output (20.0) and gives final answer
            LLMResponse(
                text="The calculated tax is $20.00.",
                model="mock-model",
                provider="mock",
                usage=Usage(input_tokens=30, output_tokens=15),
            ),
        ]
    )

    agent = AIAgent(
        client=mock_client,
        model="mock-model",
        tools=[calculate_tax],
    )
    session = ChatSession(agent=agent, budget=Budget())

    await session.chat("Calculate 20% tax on 100")

    # Assertions
    assert len(mock_client.calls) == 2  # Proves 2 LLM turns occurred
    assert session.steps == 2
    assert session.usage.input_tokens == 50  # 20 + 30
    assert session.usage.output_tokens == 25  # 10 + 15

    # Verify the second LLM call included the tool output in its messages
    second_request = mock_client.calls[1]
    last_msg = second_request.messages[-1]
    assert last_msg.role == ChatRole.TOOL
    assert last_msg.tool_results[0].output == "20.0"
    assert last_msg.tool_results[0].tool_call_id == "call_1"


@pytest.mark.asyncio
async def test_tool_argument_validation_failure():
    """Verify that when the LLM provides invalid arguments,
    FunctionTool catches it and returns a readable validation error."""

    # Missing required argument "amount"
    result = await calculate_tax(rate=0.05)

    assert isinstance(result, str)
    assert "Invalid tool arguments" in result
    assert "amount" in result  # Pydantic notes "amount" is missing
