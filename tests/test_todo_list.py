"""Unit tests for TodoListState and TodoListTool."""

import sys
import os

# Ensure backend module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.main import TodoListState, TodoListTool, todo_states


class TestTodoListState:
    """Tests for the TodoListState class."""

    def test_initial_state_is_empty(self):
        state = TodoListState()
        assert state.items == []
        assert state.format() == "Todo list is empty."

    def test_create_populates_items(self):
        state = TodoListState()
        result = state.create(["Step 1", "Step 2", "Step 3"])
        assert len(state.items) == 3
        assert state.items[0] == {"id": 1, "task": "Step 1", "done": False}
        assert state.items[2] == {"id": 3, "task": "Step 3", "done": False}
        assert "[todo] Step 1" in result
        assert "[todo] Step 3" in result

    def test_create_overwrites_previous_items(self):
        state = TodoListState()
        state.create(["Old task"])
        state.create(["New task 1", "New task 2"])
        assert len(state.items) == 2
        assert state.items[0]["task"] == "New task 1"

    def test_complete_marks_item_done(self):
        state = TodoListState()
        state.create(["Task A", "Task B"])
        result = state.complete(1)
        assert state.items[0]["done"] is True
        assert state.items[1]["done"] is False
        assert "[done] Task A" in result

    def test_complete_nonexistent_id_is_noop(self):
        state = TodoListState()
        state.create(["Task A"])
        result = state.complete(999)
        assert state.items[0]["done"] is False
        assert "[todo] Task A" in result

    def test_add_appends_new_item(self):
        state = TodoListState()
        state.create(["Task 1"])
        result = state.add("Task 2")
        assert len(state.items) == 2
        assert state.items[1] == {"id": 2, "task": "Task 2", "done": False}
        assert "[todo] Task 2" in result

    def test_add_to_empty_list(self):
        state = TodoListState()
        result = state.add("First task")
        assert len(state.items) == 1
        assert state.items[0] == {"id": 1, "task": "First task", "done": False}
        assert "[todo] First task" in result

    def test_format_shows_all_items_with_status(self):
        state = TodoListState()
        state.create(["A", "B", "C"])
        state.complete(2)
        output = state.format()
        assert "1. [todo] A" in output
        assert "2. [done] B" in output
        assert "3. [todo] C" in output
        assert output.startswith("Current todo list:")


class TestTodoListTool:
    """Tests for the TodoListTool LangChain tool."""

    def setup_method(self):
        """Clear global state before each test."""
        todo_states.clear()

    def test_create_action(self):
        tool = TodoListTool(conv_id="test-1")
        result = tool._run("create|Research topic|Write draft|Review")
        assert "[todo] Research topic" in result
        assert "[todo] Write draft" in result
        assert "[todo] Review" in result

    def test_create_empty_tasks_returns_error(self):
        tool = TodoListTool(conv_id="test-2")
        result = tool._run("create|")
        assert "Error" in result

    def test_complete_action(self):
        tool = TodoListTool(conv_id="test-3")
        tool._run("create|Task 1|Task 2")
        result = tool._run("complete|1")
        assert "[done] Task 1" in result
        assert "[todo] Task 2" in result

    def test_complete_invalid_number(self):
        tool = TodoListTool(conv_id="test-4")
        tool._run("create|Task 1")
        result = tool._run("complete|abc")
        assert "Error" in result

    def test_add_action(self):
        tool = TodoListTool(conv_id="test-5")
        tool._run("create|Existing task")
        result = tool._run("add|New task")
        assert "[todo] New task" in result
        assert "[todo] Existing task" in result

    def test_add_empty_task_returns_error(self):
        tool = TodoListTool(conv_id="test-6")
        result = tool._run("add|")
        assert "Error" in result

    def test_view_action(self):
        tool = TodoListTool(conv_id="test-7")
        tool._run("create|Task A|Task B")
        result = tool._run("view")
        assert "[todo] Task A" in result
        assert "[todo] Task B" in result

    def test_unknown_action(self):
        tool = TodoListTool(conv_id="test-8")
        result = tool._run("delete|1")
        assert "Unknown action" in result

    def test_per_conversation_isolation(self):
        tool_a = TodoListTool(conv_id="conv-a")
        tool_b = TodoListTool(conv_id="conv-b")
        tool_a._run("create|Task for A")
        tool_b._run("create|Task for B")
        result_a = tool_a._run("view")
        result_b = tool_b._run("view")
        assert "Task for A" in result_a
        assert "Task for B" not in result_a
        assert "Task for B" in result_b
        assert "Task for A" not in result_b

    def test_whitespace_handling(self):
        tool = TodoListTool(conv_id="test-ws")
        result = tool._run("  create|  Step 1  |  Step 2  ")
        assert "[todo] Step 1" in result
        assert "[todo] Step 2" in result
