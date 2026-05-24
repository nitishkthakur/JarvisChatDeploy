"""Unit tests for helper functions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.main import (
    OllamaCloudLLM,
    _history_context,
    _is_trivial,
    _parse_agent_steps,
    _tool_label,
)


class TestIsTrivial:
    """Tests for the _is_trivial helper."""

    def test_greetings_are_trivial(self):
        assert _is_trivial("hello") is True
        assert _is_trivial("hi") is True
        assert _is_trivial("hey") is True
        assert _is_trivial("bye") is True
        assert _is_trivial("goodbye") is True

    def test_greetings_case_insensitive(self):
        assert _is_trivial("Hello") is True
        assert _is_trivial("HI") is True
        assert _is_trivial("GOODBYE") is True

    def test_greetings_with_whitespace(self):
        assert _is_trivial("  hello  ") is True
        assert _is_trivial("  hey  ") is True

    def test_prefixed_trivials(self):
        assert _is_trivial("hi there") is True
        assert _is_trivial("thanks a lot") is True
        assert _is_trivial("thank you very much") is True
        assert _is_trivial("how are you doing") is True
        assert _is_trivial("what is your name") is True

    def test_non_trivial_questions(self):
        assert _is_trivial("What is quantum computing?") is False
        assert _is_trivial("Explain machine learning") is False
        assert _is_trivial("Write a Python sort function") is False

    def test_empty_string_is_not_trivial(self):
        assert _is_trivial("") is False


class TestHistoryContext:
    """Tests for the _history_context helper."""

    def test_empty_history_returns_empty(self):
        assert _history_context([]) == ""

    def test_single_turn(self):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = _history_context(history)
        assert "User: Hello" in result
        assert "Assistant: Hi!" in result
        assert result.startswith("Previous conversation:")

    def test_truncates_to_last_8_turns(self):
        history = [
            {"role": "user", "content": f"msg-{i}"}
            for i in range(12)
        ]
        result = _history_context(history)
        assert "msg-0" not in result
        assert "msg-4" in result
        assert "msg-11" in result


class TestOllamaCloudLLMHelpers:
    """Tests for OllamaCloudLLM helper methods."""

    def test_to_api_messages(self):
        llm = OllamaCloudLLM(model_name="test")
        messages = [
            SystemMessage(content="You are a helper."),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi!"),
        ]
        result = llm._to_api_messages(messages)
        assert result == [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_to_api_messages_empty(self):
        llm = OllamaCloudLLM(model_name="test")
        assert llm._to_api_messages([]) == []

    def test_llm_type(self):
        llm = OllamaCloudLLM(model_name="test")
        assert llm._llm_type == "ollama-cloud"

    def test_default_values(self):
        llm = OllamaCloudLLM()
        assert llm.model_name == "qwen3.5:397b-cloud"
        assert llm.base_url == "https://api.ollama.com"
        assert llm.temperature == 0.7


class TestToolLabel:
    """Tests for _tool_label helper."""

    def test_known_tool(self):
        assert _tool_label("todo_list") == "planning tool"

    def test_unknown_tool(self):
        assert _tool_label("web_search") == "web_search tool"
        assert _tool_label("calculator") == "calculator tool"


class TestParseAgentSteps:
    """Tests for _parse_agent_steps helper."""

    def test_empty_steps(self):
        assert _parse_agent_steps([]) == []

    def test_step_with_thought_and_action(self):
        class FakeAction:
            tool = "todo_list"
            tool_input = "create|Step 1"
            log = "Thought: I need to plan\nAction: todo_list"

        steps = [(FakeAction(), "Current todo list:\n  1. [todo] Step 1")]
        events = _parse_agent_steps(steps)

        # Should have thinking, tool_call, tool_result
        types = [e["type"] for e in events]
        assert "thinking" in types
        assert "tool_call" in types
        assert "tool_result" in types

        # Check thinking text extracted from Thought: line
        thinking = [e for e in events if e["type"] == "thinking"][0]
        assert thinking["text"] == "I need to plan"

        # Check tool_call text
        tool_call = [e for e in events if e["type"] == "tool_call"][0]
        assert "planning tool" in tool_call["text"]
