"""Question answering over the portfolio with LLM function calling, the tool allowlist and a capped tool loop."""

import json
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent.graph import STRUCTURAL_NUMBERS, message_text
from agent.guardrails import (
    GuardedToolbox,
    ToolCallError,
    ToolNotAllowedError,
    check_groundedness,
    numbers_in,
    with_disclaimer,
)
from agent.mcp_client import groww_tools

MAX_STEPS = 6

SYSTEM_PROMPT = """You answer questions about the user's Groww portfolio using the read-only tools provided.
- Call tools to get real data. Never guess prices, holdings or dates.
- You cannot place, modify or cancel orders. If asked to, say this assistant is read-only.
- Give no buy, sell or hold advice and no price predictions.
- Answer in a few sentences and quote the numbers exactly as the tools returned them."""


@dataclass
class AssistantResult:
    answer: str
    stopped: str
    steps: int
    verified: bool
    ungrounded: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


async def answer_question(question: str, llm, toolbox: GuardedToolbox, max_steps: int = MAX_STEPS) -> AssistantResult:
    model = llm.bind_tools(toolbox.tools())
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]
    tool_outputs: list[dict] = []

    for step in range(1, max_steps + 1):
        reply = await model.ainvoke(messages)
        messages.append(reply)

        if not reply.tool_calls:
            answer = message_text(reply)
            grounding = check_groundedness(answer, numbers_in(tool_outputs) | numbers_in([question]) | STRUCTURAL_NUMBERS)
            return AssistantResult(
                answer=with_disclaimer(answer),
                stopped="answered",
                steps=step,
                verified=grounding.passed,
                ungrounded=grounding.ungrounded,
                tool_calls=toolbox.drain(),
            )

        for call in reply.tool_calls:
            try:
                output = await toolbox.call(call["name"], call.get("args") or {})
                tool_outputs.append(output)
                content = json.dumps(output)
            except ToolNotAllowedError as exc:
                content = f"Blocked by guardrail: {exc} This assistant is read-only."
            except ToolCallError as exc:
                content = f"Tool error: {exc}"
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    return AssistantResult(
        answer=f"I couldn't finish within {max_steps} steps. Try a narrower question.",
        stopped="max_steps",
        steps=max_steps,
        verified=True,
        tool_calls=toolbox.drain(),
    )


async def ask(question: str, llm) -> AssistantResult:
    async with groww_tools() as tools:
        return await answer_question(question, llm, GuardedToolbox(tools))
