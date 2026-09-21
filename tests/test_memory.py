"""Focused tests for MemoryAgent storage, retrieval, and context grounding."""

import unittest

from src.agents.memory.agent import MemoryAgent
from src.agents.memory.schemas import MemoryAgentInput, MemoryOperation
from src.common.schemas import AgentStatus, AgentType


class TestMemoryAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MemoryAgent()
        self.agent.initialize()

    def tearDown(self):
        self.agent.shutdown()

    def test_coordinator_task_is_stored_and_exposed(self):
        output = self.agent.process({
            "task_id": "task-1",
            "task_status": "VALID",
            "action": "pick_and_place",
            "target": "bottle",
            "location": "LEFT",
        })

        self.assertTrue(output.success)
        self.assertEqual(output.last_target, "bottle")
        self.assertEqual(output.last_action, "pick_and_place")
        self.assertEqual(output.last_location, "LEFT")
        self.assertEqual(output.current_task["task_id"], "task-1")
        self.assertEqual(output.total_entries_count, 1)

    def test_pronoun_resolution_uses_previous_task(self):
        self.agent.process({"action": "inspect_object", "target": "cup", "location": "CENTER"})
        output = self.agent.process({"action": "pick_and_place", "target": "it"})

        self.assertEqual(output.last_target, "cup")
        self.assertTrue(output.resolved_context["resolved_from_previous"])
        self.assertEqual(output.resolved_context["location"], "CENTER")

    def test_string_operation_and_dictionary_query(self):
        self.agent.process({"action": "inspect_object", "target": "book"})
        output = self.agent.process(MemoryAgentInput(
            operation="retrieve",
            query={"target": "book", "limit": 1},
        ))

        self.assertEqual(output.operation, MemoryOperation.RETRIEVE.value)
        self.assertEqual(len(output.relevant_entries), 1)
        self.assertEqual(output.relevant_entries[0].content["target"], "book")

    def test_lifecycle_metadata(self):
        self.assertEqual(self.agent.status, AgentStatus.READY)
        self.assertEqual(self.agent.agent_type, AgentType.MEMORY)


if __name__ == "__main__":
    unittest.main()