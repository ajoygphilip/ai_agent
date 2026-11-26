import asyncio

from app.ai.models import GenerationConfig
from app.ai.llm.observability import LLMTracer
from app.ai.llm.providers import provider_factory
from app.ai.agent.models import Budget, AgentResult
from app.ai.agent.agent import AIAgent
from app.ai.agent.session import ChatSession
from app.ai.llm.tools.library import web_search, calculator

from pydantic import BaseModel
from datasets import load_dataset
from typing import Awaitable

GAIA_PROMPT = """You are a general AI assistant. I will ask you a question.
First, determine if you can solve this problem with your current capabilities and set "is_solvable" accordingly.
If you can solve it, set "is_solvable" to true and provide your answer in "final_answer".
If you cannot solve it, set "is_solvable" to false and explain why in "unsolvable_reason".
Your final answer should be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
If you are asked for a number, don't use a comma to write your number neither use units such as $ or percent sign unless specified otherwise.
If you are asked for a string, don't use articles, neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise.
If you are asked for a comma-separated list, apply the above rules depending on whether the element is a number or a string. For mathemtical operations always use tools."""


class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""


async def solve_problem(session: ChatSession, question: str) -> AgentResult|None:
    return await session.chat(user_text=question)

def is_correct(prediction: str | None, answer: str) -> bool:
    """Check exact match between prediction and answer (case-insensitive)."""
    print(f"******{prediction}******")
    print(f"******{answer}******")
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()

async def evaluate_problem(problem: dict, agent: AIAgent) -> dict:
    """Evaluate a single problem-model pair and return result."""
    try:
        
        output = await solve_problem(ChatSession(agent=agent, budget=Budget()), problem["Question"])
        final_answer = output.structured_output.final_answer if output else None #type: ignore
        is_solvable = output.structured_output.is_solvable if output else None #type: ignore
        unsolvable_reason = output.structured_output.unsolvable_reason if output else None #type: ignore

        ret = {
            "task_id": problem["task_id"],
            "model": agent.model,
            "correct": is_correct(final_answer, problem["Final answer"]),
            "is_solvable": is_solvable,
            "prediction": final_answer,
            "answer": problem["Final answer"],
            "unsolvable_reason": unsolvable_reason,
        }
    except Exception as e:
        ret = {
            "task_id": problem["task_id"],
            "model": agent.model,
            "correct": False,
            "is_solvable": None,
            "prediction": None,
            "answer": problem["Final answer"],
            "error": str(e),
        }
    print(problem["Question"],ret["prediction"],ret["answer"],ret.get("error"),ret["correct"],sep="\n")
    print("======================")
    return ret
    

async def run_eval():
    client = LLMTracer(provider_factory("bedrock"))
    tools = [web_search, calculator]
    config = GenerationConfig(
        temperature=0.5,
        max_tokens=2000,
    )
    agent = AIAgent(
        tools=tools,
        client=client,
        config=config,
        # model="amazon.nova-lite-v1:0",
        model="us.amazon.nova-pro-v1:0",
        output_schema=GaiaOutput,
        instructions=[GAIA_PROMPT],
    )

    try:
        level1_problems = load_dataset(
            "gaia-benchmark/GAIA", "2023_level1", split="validation"
        )
        problems = level1_problems.select(range(10))
        results = [
            evaluate_problem(problem ,agent) # type: ignore
            for problem in problems
        ]
        all_results = await asyncio.gather(*results)
        correct = sum(1 for r in all_results if r["correct"]) # type: ignore
        solvable = sum(1 for r in all_results if r.get("is_solvable")) # type: ignore
        total = len(all_results)
        print(f"{correct}/{total} correct, {solvable}/{total} judged solvable")
        
        return all_results

    except KeyboardInterrupt:
        print("\nAgent interrupted.")

    except EOFError:
        print("\nEOF")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"Request failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(run_eval())
