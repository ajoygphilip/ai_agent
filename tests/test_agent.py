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
    FunctionTool catches it and raises a readable ToolCallException."""
    from app.ai.exceptions import ToolCallException

    # Missing required argument "amount"
    with pytest.raises(ToolCallException) as exc_info:
        await calculate_tax(rate=0.05)

    assert "Invalid tool arguments" in str(exc_info.value)
    assert "amount" in str(exc_info.value)  # Pydantic notes "amount" is missing


from pydantic import BaseModel


class WeatherOutput(BaseModel):
    city: str
    temperature: float
    condition: str


@pytest.mark.asyncio
async def test_agent_structured_output_success():
    """Verify that agent parses and returns structured output when valid."""
    mock_client = MockLLMClient(
        responses=[
            LLMResponse(
                text="",
                model="mock-model",
                provider="mock",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_out_1",
                        name="final_answer",
                        arguments={
                            "city": "Paris",
                            "temperature": 22.5,
                            "condition": "Sunny",
                        },
                    )
                ],
                usage=Usage(input_tokens=15, output_tokens=15),
            )
        ]
    )

    agent = AIAgent(
        client=mock_client,
        model="mock-model",
        output_schema=WeatherOutput,
    )
    session = ChatSession(agent=agent, budget=Budget())
    result = await session.chat("Weather in Paris?")

    assert result.structured_output is not None
    assert isinstance(result.structured_output, WeatherOutput)
    assert result.structured_output.city == "Paris"
    assert result.structured_output.temperature == 22.5
    assert result.structured_output.condition == "Sunny"
    assert session.steps == 1


@pytest.mark.asyncio
async def test_agent_structured_output_validation_retry():
    """Verify that when the LLM outputs invalid structured data,
    the validation error is returned as a tool result and the LLM retries and succeeds."""
    mock_client = MockLLMClient(
        responses=[
            # Step 1: LLM calls final_answer with invalid temperature (string instead of float)
            LLMResponse(
                text="",
                model="mock-model",
                provider="mock",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_err_1",
                        name="final_answer",
                        arguments={
                            "city": "London",
                            "temperature": "not_a_number",
                            "condition": "Rainy",
                        },
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=10),
            ),
            # Step 2: LLM receives validation error, fixes arguments and calls final_answer again
            LLMResponse(
                text="",
                model="mock-model",
                provider="mock",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_fixed_1",
                        name="final_answer",
                        arguments={
                            "city": "London",
                            "temperature": 14.0,
                            "condition": "Rainy",
                        },
                    )
                ],
                usage=Usage(input_tokens=20, output_tokens=10),
            ),
        ]
    )

    agent = AIAgent(
        client=mock_client,
        model="mock-model",
        output_schema=WeatherOutput,
    )
    session = ChatSession(agent=agent, budget=Budget())
    result = await session.chat("Weather in London?")

    assert session.steps == 2
    assert result.structured_output is not None
    assert result.structured_output.temperature == 14.0
    assert result.structured_output.city == "London"

    # Verify the error was sent back in turn 1
    second_request = mock_client.calls[1]
    tool_msg = [m for m in second_request.messages if m.role == ChatRole.TOOL][-1]
    assert tool_msg.tool_results[0].is_error is True
    assert "Validation error for WeatherOutput" in tool_msg.tool_results[0].output


@pytest.mark.asyncio
async def test_agent_structured_output_missing_tool_call_retry():
    """Verify that if the LLM forgets to call the output tool and only responds with text,
    a reminder is sent back and the LLM retries by calling final_answer."""
    mock_client = MockLLMClient(
        responses=[
            # Step 1: LLM outputs plain text instead of calling final_answer
            LLMResponse(
                text="The weather in Rome is 26 degrees and sunny.",
                model="mock-model",
                provider="mock",
                usage=Usage(input_tokens=10, output_tokens=10),
            ),
            # Step 2: LLM is reminded and calls final_answer
            LLMResponse(
                text="",
                model="mock-model",
                provider="mock",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_rome_1",
                        name="final_answer",
                        arguments={
                            "city": "Rome",
                            "temperature": 26.0,
                            "condition": "Sunny",
                        },
                    )
                ],
                usage=Usage(input_tokens=25, output_tokens=15),
            ),
        ]
    )

    agent = AIAgent(
        client=mock_client,
        model="mock-model",
        output_schema=WeatherOutput,
    )
    session = ChatSession(agent=agent, budget=Budget())
    result = await session.chat("Weather in Rome?")

    assert session.steps == 2
    assert result.structured_output is not None
    assert result.structured_output.city == "Rome"
    assert result.structured_output.temperature == 26.0


@pytest.mark.asyncio
async def test_agent_structured_output_with_intermediate_tools():
    """Verify that agent can call normal tools first to gather data,
    and then call final_answer to provide the structured result."""
    mock_client = MockLLMClient(
        responses=[
            # Step 1: LLM calls calculate_tax
            LLMResponse(
                text="",
                model="mock-model",
                provider="mock",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_tax_1",
                        name="calculate_tax",
                        arguments={"amount": 200.0, "rate": 0.15},
                    )
                ],
                usage=Usage(input_tokens=10, output_tokens=10),
            ),
            # Step 2: LLM receives tax output (30.0) and calls final_answer
            LLMResponse(
                text="",
                model="mock-model",
                provider="mock",
                tool_calls=[
                    ToolCall(
                        tool_call_id="call_out_final",
                        name="final_answer",
                        arguments={
                            "city": "Madrid",
                            "temperature": 30.0,
                            "condition": "Hot",
                        },
                    )
                ],
                usage=Usage(input_tokens=20, output_tokens=10),
            ),
        ]
    )

    agent = AIAgent(
        client=mock_client,
        model="mock-model",
        tools=[calculate_tax],
        output_schema=WeatherOutput,
    )

    # Verify tool is completely stateless
    assert not hasattr(agent.output_tool, "latest_result")

    session = ChatSession(agent=agent, budget=Budget())
    result = await session.chat("Get weather for Madrid")

    assert session.steps == 2
    assert result.structured_output is not None
    assert isinstance(result.structured_output, WeatherOutput)
    assert result.structured_output.city == "Madrid"
    assert result.structured_output.temperature == 30.0
