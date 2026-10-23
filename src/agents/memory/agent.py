"""Memory Agent: maintains interaction history, task state, and contextual reference grounding."""

from typing import Any, Dict, List, Optional, Union
import uuid

from src.agents.base import BaseAgent
from src.agents.memory.memory_store import BaseMemoryStore, InMemoryStore
from src.agents.memory.schemas import (
    CoordinatorTaskInput,
    MemoryAgentInput,
    MemoryAgentOutput,
    MemoryEntry,
    MemoryEntryType,
    MemoryOperation,
    MemoryQuery,
    TaskStatus,
)
from src.common.exceptions import AgentExecutionError
from src.common.schemas import AgentType


class MemoryAgent(BaseAgent):
    """
    Memory & Context Agent for Human-Robot Interaction.

    Positioned between the Coordinator and the Task Planner:
    Voice / Vision / Gesture -> Coordinator -> Memory Agent -> Task Planner

    Responsibilities:
    - Maintain structured conversation and task execution history
    - Associate visual/spatial observations with high-level tasks
    - Resolve referential ambiguity (e.g. "pick it up" -> target="bottle")
    - Provide grounded historical context for downstream Task Planning
    """

    REFERENTIAL_PRONOUNS = {
        "it",
        "that",
        "this",
        "them",
        "the object",
        "the item",
        "object",
        "item",
        "one",
        "the same",
    }

    def __init__(
        self,
        name: str = "MemoryAgent",
        store: Optional[BaseMemoryStore] = None,
        config: Optional[dict] = None,
    ):
        super().__init__(
            name=name,
            agent_type=AgentType.MEMORY,
            version="0.1.0",
            description="Maintains multimodal task history and resolves contextual references for HRI.",
            config=config or {},
        )
        self.store = store
        mem_cfg = self.config.get("memory", {})
        self._max_history = mem_cfg.get("max_history_size", 200)
        self._enable_pronoun_resolution = mem_cfg.get("enable_pronoun_resolution", True)

    def _initialize(self) -> bool:
        """Initialize memory persistence backend."""
        if self.store is None:
            self.store = InMemoryStore(max_entries=self._max_history)
        self.logger.info("MemoryAgent initialized successfully with active store.")
        return True

    # ------------------------------------------------------------------
    # High-level public API methods
    # ------------------------------------------------------------------

    def store_entry(
        self,
        content: Dict[str, Any],
        entry_type: MemoryEntryType = MemoryEntryType.TASK,
        task_id: Optional[str] = None,
        source: str = "coordinator",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a structured memory entry directly."""
        if self.store is None:
            self._initialize()

        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            entry_type=entry_type,
            content=content,
            task_id=task_id,
            source=source,
            metadata=metadata or {},
        )
        return self.store.store(entry)

    def retrieve_entries(
        self,
        query: Union[MemoryQuery, str],
        limit: int = 5,
        entry_type: Optional[MemoryEntryType] = None,
    ) -> List[MemoryEntry]:
        """Retrieve memory entries matching query."""
        if self.store is None:
            self._initialize()

        if isinstance(query, str):
            q = MemoryQuery(text=query, limit=limit, entry_type=entry_type)
        else:
            q = query
            if limit:
                q.limit = limit

        return self.store.retrieve(q)

    def get_recent(
        self,
        limit: int = 5,
        entry_type: Optional[MemoryEntryType] = None,
    ) -> List[MemoryEntry]:
        """Retrieve most recent memory entries."""
        if self.store is None:
            self._initialize()
        return self.store.get_recent(limit=limit, entry_type=entry_type)

    def get_current_task(self) -> Optional[Dict[str, Any]]:
        """Retrieve the current active task if any."""
        if self.store is None:
            self._initialize()
        task_entry = self.store.get_current_task()
        return task_entry.to_dict() if task_entry else None

    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update execution status for a given task."""
        if self.store is None:
            self._initialize()
        return self.store.update_task_status(task_id, status)

    def clear(self) -> None:
        """Purge all stored entries."""
        if self.store is not None:
            self.store.clear()

    # ------------------------------------------------------------------
    # Context Resolution Helpers
    # ------------------------------------------------------------------

    def _resolve_target_and_context(
        self,
        target: Optional[str],
        action: Optional[str],
        location: Optional[str],
    ) -> Dict[str, Any]:
        """
        Check for referential language (e.g. 'it', 'that') and ground
        it against recent conversation and task history.
        """
        recent_entries = self.store.get_recent(limit=10)
        resolved_target = target
        resolved_location = location
        is_referential = False
        previous_task_info = None
        previous_target_found = None
        resolution_source = "explicit_input"

        norm_target = (target or "").strip().lower()

        # Check if the target is an explicit anaphora / pronoun (e.g. 'it', 'this', 'that')
        if self._enable_pronoun_resolution and norm_target and (norm_target in self.REFERENTIAL_PRONOUNS):
            is_referential = True
            for entry in recent_entries:
                c = entry.content
                prev_target = c.get("target") or entry.metadata.get("target")
                if prev_target and str(prev_target).lower() not in self.REFERENTIAL_PRONOUNS:
                    previous_target_found = prev_target
                    resolved_target = prev_target
                    resolution_source = "historical context"
                    if not resolved_location:
                        resolved_location = c.get("location") or entry.metadata.get("location")
                    previous_task_info = {
                        "task_id": entry.task_id or entry.id,
                        "action": c.get("action"),
                        "target": prev_target,
                        "location": resolved_location,
                    }
                    break

            if not previous_target_found:
                resolved_target = None
                resolution_source = "unresolved"
        else:
            resolution_source = "explicit_input"

        return {
            "resolved_target": resolved_target,
            "resolved_location": resolved_location,
            "is_referential": is_referential,
            "previous_target_found": previous_target_found,
            "previous_task_info": previous_task_info,
            "resolution_source": resolution_source,
        }

    # ------------------------------------------------------------------
    # BaseAgent lifecycle implementations
    # ------------------------------------------------------------------

    def _process(self, input_data: Any) -> MemoryAgentOutput:
        """
        Execute memory operations, resolve context, and formulate structured state.
        Accepts MemoryAgentInput, CoordinatorTaskInput, dict, or query string.
        """
        if self.store is None:
            raise AgentExecutionError("Memory store is not initialized.")

        operation = MemoryOperation.RESOLVE_CONTEXT
        coordinator_dict = None
        query = None
        task_id = None
        new_status = None
        limit = 5

        # ---------------------------------------------------------
        # 1. Parse input payload
        # ---------------------------------------------------------
        if isinstance(input_data, MemoryAgentInput):
            operation = input_data.operation
            coordinator_dict = input_data.coordinator_output
            if input_data.task_input:
                coordinator_dict = input_data.task_input.to_dict()
            query = input_data.query
            task_id = input_data.task_id
            new_status = input_data.new_status
            limit = input_data.limit

            if input_data.entry_to_store:
                self.store.store(input_data.entry_to_store)

        elif isinstance(input_data, CoordinatorTaskInput):
            operation = MemoryOperation.RESOLVE_CONTEXT
            coordinator_dict = input_data.to_dict()

        elif isinstance(input_data, dict):
            # Check if this is a Coordinator task dictionary or generic operation
            if "action" in input_data or "task_status" in input_data:
                operation = MemoryOperation.RESOLVE_CONTEXT
                coordinator_dict = input_data
            else:
                op_val = input_data.get("operation", MemoryOperation.RESOLVE_CONTEXT.value)
                try:
                    operation = MemoryOperation(op_val)
                except ValueError:
                    operation = MemoryOperation.RESOLVE_CONTEXT

                coordinator_dict = input_data.get("coordinator_output")
                query = input_data.get("query")
                task_id = input_data.get("task_id")
                new_status = input_data.get("new_status")
                limit = input_data.get("limit", 5)

        elif isinstance(input_data, str):
            # Direct query string
            operation = MemoryOperation.RETRIEVE
            query = MemoryQuery(text=input_data, limit=5)

        else:
            raise AgentExecutionError(
                f"Unsupported input type for MemoryAgent: {type(input_data)}."
            )

        if isinstance(operation, str):
            try:
                operation = MemoryOperation(operation.lower())
            except ValueError as exc:
                raise AgentExecutionError(
                    f"Unsupported memory operation: {operation}"
                ) from exc

        if isinstance(query, dict):
            query = MemoryQuery(**query)

        # ---------------------------------------------------------
        # 2. Dispatch operations
        # ---------------------------------------------------------
        relevant_entries: List[MemoryEntry] = []
        resolved_context: Dict[str, Any] = {}
        summary_text = ""
        current_target = None
        current_action = None
        current_location = None

        if operation == MemoryOperation.CLEAR:
            self.store.clear()
            summary_text = "Memory cleared."

        elif operation == MemoryOperation.UPDATE_TASK:
            if not task_id or not new_status:
                raise AgentExecutionError("Task ID and new status required for UPDATE_TASK operation.")
            updated = self.store.update_task_status(task_id, new_status)
            summary_text = f"Task '{task_id}' status update: {'success' if updated else 'failed'}."

        elif operation == MemoryOperation.GET_RECENT:
            relevant_entries = self.store.get_recent(limit=limit)
            summary_text = f"Retrieved {len(relevant_entries)} recent memory entries."

        elif operation == MemoryOperation.GET_CURRENT_TASK:
            cur = self.store.get_current_task()
            if cur:
                relevant_entries = [cur]
                summary_text = f"Current active task: {cur.content.get('action')} on {cur.content.get('target')}."
            else:
                summary_text = "No active task found."

        elif operation == MemoryOperation.RETRIEVE:
            if isinstance(query, str):
                query = MemoryQuery(text=query, limit=limit)
            elif query is None:
                query = MemoryQuery(limit=limit)
            relevant_entries = self.store.retrieve(query)
            summary_text = f"Retrieved {len(relevant_entries)} entries matching query."

        else:
            # Default / Primary: RESOLVE_CONTEXT from Coordinator Output
            if coordinator_dict:
                raw_action = coordinator_dict.get("action", "unknown")
                raw_target = coordinator_dict.get("target")
                raw_location = coordinator_dict.get("location")
                task_status = coordinator_dict.get("task_status", "VALID")
                confidence = float(coordinator_dict.get("confidence", 1.0))
                assigned_task_id = coordinator_dict.get("task_id") or str(uuid.uuid4())

                # Contextual grounding / pronoun resolution
                res = self._resolve_target_and_context(
                    target=raw_target,
                    action=raw_action,
                    location=raw_location,
                )

                resolved_target = res["resolved_target"]
                resolved_location = res["resolved_location"]
                is_referential = res["is_referential"]
                prev_target_found = res["previous_target_found"]
                prev_info = res["previous_task_info"]
                resolution_source = res["resolution_source"]

                # Trace logging for referential grounding
                if is_referential:
                    print("\n" + "=" * 50)
                    print("[MEMORY]")
                    print(f"  Reference detected    : \"{raw_target or 'it'}\"")
                    if prev_target_found:
                        print(f"  Previous target found : \"{prev_target_found}\"")
                        print(f"  Resolved target       : \"{resolved_target}\"")
                        print(f"  Resolution source     : historical context")
                    else:
                        print(f"  Previous target found : NONE")
                        print(f"  Resolved target       : UNRESOLVED")
                        print(f"  Resolution source     : unresolved")
                    print("=" * 50 + "\n", flush=True)

                # Assemble resolved content
                resolved_context = {
                    "task_id": assigned_task_id,
                    "task_status": task_status,
                    "action": raw_action,
                    "target": resolved_target,
                    "location": resolved_location,
                    "confidence": confidence,
                    "original_target": raw_target,
                    "reference_detected": is_referential,
                    "previous_target_found": prev_target_found,
                    "resolved_from_previous": is_referential and bool(prev_target_found),
                    "resolution_source": resolution_source,
                    "previous_task": prev_info,
                    "status": TaskStatus.REQUESTED.value if (task_status == "VALID" and resolved_target) else TaskStatus.INVALID.value,
                }

                # Store this task in memory (only store resolved or explicit targets)
                entry_content = dict(resolved_context)
                if not resolved_target:
                    entry_content["target"] = raw_target

                entry = MemoryEntry(
                    id=assigned_task_id,
                    entry_type=MemoryEntryType.TASK,
                    content=entry_content,
                    task_id=assigned_task_id,
                    source="coordinator",
                    metadata={
                        "raw_coordinator_output": coordinator_dict,
                        "resolved_from_previous": resolved_context["resolved_from_previous"],
                        "resolution_source": resolution_source,
                    },
                )
                self.store.store(entry)
                relevant_entries = [entry]

                current_target = resolved_target or raw_target
                current_action = raw_action
                current_location = resolved_location

                if resolved_context["resolved_from_previous"]:
                    summary_text = (
                        f"Resolved contextual task '{raw_action}': "
                        f"target grounded to '{resolved_target}' (from historical context) at '{resolved_location}'."
                    )
                else:
                    summary_text = (
                        f"Tracked task '{raw_action}' targeting '{resolved_target or raw_target}' at '{resolved_location}'."
                    )
            else:
                # No specific coordinator output provided; return recent context
                recent = self.store.get_recent(limit=3)
                relevant_entries = recent
                summary_text = f"Retrieved {len(recent)} recent memory entries."

        # Fetch latest state for output container
        current_task_entry = self.store.get_current_task()
        current_task_dict = current_task_entry.to_dict() if current_task_entry else None

        if not current_target and current_task_entry:
            current_target = current_task_entry.content.get("target")
        if not current_action and current_task_entry:
            current_action = current_task_entry.content.get("action")
        if not current_location and current_task_entry:
            current_location = current_task_entry.content.get("location")

        return MemoryAgentOutput(
            agent_name=self.name,
            agent_type=self.agent_type.value,
            success=True,
            operation=operation.value if isinstance(operation, MemoryOperation) else str(operation),
            resolved_context=resolved_context,
            relevant_entries=relevant_entries,
            current_task=current_task_dict,
            last_target=current_target,
            last_action=current_action,
            last_location=current_location,
            total_entries_count=self.store.count(),
            summary_text=summary_text,
        )

    def _reset(self) -> None:
        """Reset Memory Agent internal state and clear working store."""
        if self.store is not None:
            self.store.clear()

    def _shutdown(self) -> None:
        """Release Memory Agent resources."""
        self.store = None
