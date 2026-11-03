import asyncio

from langchain_core.messages import AIMessage, ToolMessage

from agent.assistant import answer_question
from agent.guardrails import DISCLAIMER, GuardedToolbox
from tests.conftest import make_tools


class ScriptedToolModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.bound_tools = []
        self.seen_messages = []

    def bind_tools(self, tools):
        self.bound_tools = [t.name for t in tools]
        return self

    async def ainvoke(self, messages):
        self.seen_messages = list(messages)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def test_blocks_order_tool_then_answers_from_real_data():
    model = ScriptedToolModel([
        tool_call("place_order", {"symbol": "INFY", "quantity": 10}, "1"),
        tool_call("get_quote", {"symbol": "INFY"}, "2"),
        AIMessage(content="INFY last traded at ₹1610.00, down 6.2% on the day."),
    ])
    toolbox = GuardedToolbox(make_tools(include_order_tool=True))
    result = asyncio.run(answer_question("Buy 10 INFY and tell me its price", model, toolbox))

    assert "place_order" not in model.bound_tools
    blocked = [m for m in model.seen_messages if isinstance(m, ToolMessage) and "Blocked by guardrail" in m.content]
    assert len(blocked) == 1
    assert [c["status"] for c in result.tool_calls] == ["blocked", "ok"]
    assert result.stopped == "answered" and result.steps == 3
    assert result.verified and DISCLAIMER in result.answer


def test_flags_numbers_not_in_tool_output():
    model = ScriptedToolModel([
        tool_call("get_quote", {"symbol": "INFY"}, "1"),
        AIMessage(content="INFY will reach ₹2500 next month."),
    ])
    result = asyncio.run(answer_question("Where is INFY going?", model, GuardedToolbox(make_tools())))
    assert not result.verified and result.ungrounded == ["₹2500"]


def test_tool_loop_is_capped():
    model = ScriptedToolModel([tool_call("get_portfolio", {}, "loop")])
    result = asyncio.run(answer_question("Loop forever", model, GuardedToolbox(make_tools()), max_steps=3))
    assert result.stopped == "max_steps" and result.steps == 3
    assert len(result.tool_calls) == 3
