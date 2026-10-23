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

    # ------------------------------------------------------------------
    # 6 Core HRI Reference & Context Scenarios
    # ------------------------------------------------------------------

    def test_case_1_explicit_target(self):
        """Scenario 1: Explicit target 'Pick up the bottle.' -> target = bottle"""
        output = self.agent.process({"action": "pick_and_place", "target": "bottle", "location": "CENTER"})
        self.assertEqual(output.resolved_context["target"], "bottle")
        self.assertFalse(output.resolved_context["resolved_from_previous"])
        self.assertEqual(output.resolved_context["resolution_source"], "explicit_input")

    def test_case_2_current_turn_multimodal_grounding_received(self):
        """Scenario 2: Current-turn multimodal grounding ('Pick it up' + pointing at bottle -> grounded target = bottle)"""
        # Coordinator resolves 'it' -> 'bottle' in current turn; Memory receives grounded target 'bottle'
        output = self.agent.process({"action": "pick_and_place", "target": "bottle", "location": "LEFT", "task_status": "VALID"})
        self.assertEqual(output.resolved_context["target"], "bottle")
        self.assertFalse(output.resolved_context["resolved_from_previous"])
        self.assertEqual(output.last_target, "bottle")

    def test_case_3_historical_pronoun_resolution_bottle(self):
        """Scenario 3: Turn 1: 'Pick up the bottle.' | Turn 2: 'Move it to the left.' -> Memory resolves 'it' = bottle"""
        # Turn 1
        self.agent.process({"action": "pick_and_place", "target": "bottle", "location": "CENTER"})
        # Turn 2
        output = self.agent.process({"action": "navigate_to", "target": "it", "location": "LEFT"})
        self.assertEqual(output.resolved_context["target"], "bottle")
        self.assertTrue(output.resolved_context["resolved_from_previous"])
        self.assertEqual(output.resolved_context["resolution_source"], "historical context")
        self.assertEqual(output.resolved_context["location"], "LEFT")

    def test_case_4_historical_pronoun_resolution_cup(self):
        """Scenario 4: Turn 1: 'Pick up the cup.' | Turn 2: 'Move it to the right.' -> Memory resolves 'it' = cup"""
        # Turn 1
        self.agent.process({"action": "pick_and_place", "target": "cup", "location": "CENTER"})
        # Turn 2
        output = self.agent.process({"action": "navigate_to", "target": "it", "location": "RIGHT"})
        self.assertEqual(output.resolved_context["target"], "cup")
        self.assertTrue(output.resolved_context["resolved_from_previous"])
        self.assertEqual(output.resolved_context["resolution_source"], "historical context")
        self.assertEqual(output.resolved_context["location"], "RIGHT")

    def test_case_5_empty_memory_referential_unresolved(self):
        """Scenario 5: 'Move it to the left.' with empty memory -> unresolved / appropriate safety intervention"""
        output = self.agent.process({"action": "navigate_to", "target": "it", "location": "LEFT"})
        self.assertIsNone(output.resolved_context["target"])
        self.assertFalse(output.resolved_context["resolved_from_previous"])
        self.assertEqual(output.resolved_context["resolution_source"], "unresolved")

    def test_case_6_explicit_target_overrides_old_memory(self):
        """Scenario 6: Turn 1: 'Pick up the bottle.' | Turn 2: 'Move the cup to the left.' -> target = cup, NOT bottle"""
        # Turn 1
        self.agent.process({"action": "pick_and_place", "target": "bottle", "location": "CENTER"})
        # Turn 2 (explicit new target cup)
        out2 = self.agent.process({"action": "navigate_to", "target": "cup", "location": "LEFT"})
        self.assertEqual(out2.resolved_context["target"], "cup")
        self.assertFalse(out2.resolved_context["resolved_from_previous"])
        self.assertEqual(out2.resolved_context["resolution_source"], "explicit_input")

        # Turn 3 ('it' should now resolve to cup, not bottle)
        out3 = self.agent.process({"action": "inspect_object", "target": "it"})
        self.assertEqual(out3.resolved_context["target"], "cup")
        self.assertTrue(out3.resolved_context["resolved_from_previous"])
        self.assertEqual(out3.resolved_context["resolution_source"], "historical context")



if __name__ == "__main__":
    unittest.main()