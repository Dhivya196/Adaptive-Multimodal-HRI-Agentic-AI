"""Pluggable memory storage engines for the Memory Agent."""

from abc import ABC, abstractmethod
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional

from src.agents.memory.schemas import (
    MemoryEntry,
    MemoryEntryType,
    MemoryQuery,
    TaskStatus,
)
from src.common.logger import get_logger


class BaseMemoryStore(ABC):
    """Abstract interface for memory persistence backends (In-Memory, SQLite, Chroma, FAISS)."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def store(self, entry: MemoryEntry) -> str:
        """Store a new memory entry. Returns the unique entry ID."""
        pass

    @abstractmethod
    def retrieve(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Retrieve memory entries matching query criteria."""
        pass

    @abstractmethod
    def get_recent(
        self,
        limit: int = 5,
        entry_type: Optional[MemoryEntryType] = None,
    ) -> List[MemoryEntry]:
        """Get the most recent memory entries (newest first)."""
        pass

    @abstractmethod
    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Fetch a single entry by unique ID."""
        pass

    @abstractmethod
    def get_current_task(self) -> Optional[MemoryEntry]:
        """Retrieve the most recent active or requested task."""
        pass

    @abstractmethod
    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update the status of an existing task."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Purge all stored entries."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Total number of stored memory entries."""
        pass


class InMemoryStore(BaseMemoryStore):
    """
    Lightweight, thread-safe in-memory store for task context and conversational history.
    Provides fast structured lookup and entity filtering without external database overhead.
    """

    def __init__(self, max_entries: int = 200):
        super().__init__()
        self.max_entries = max_entries
        self._entries: List[MemoryEntry] = []
        self._lock = threading.Lock()

    def store(self, entry: MemoryEntry) -> str:
        with self._lock:
            if len(self._entries) >= self.max_entries:
                # Evict oldest entry
                evicted = self._entries.pop(0)
                self.logger.debug(f"Evicted oldest memory entry '{evicted.id}'.")

            self._entries.append(entry)
            self.logger.debug(f"Stored memory entry '{entry.id}' of type '{entry.entry_type}'.")
            return entry.id

    def retrieve(self, query: MemoryQuery) -> List[MemoryEntry]:
        with self._lock:
            results: List[MemoryEntry] = []

            # Traverse newest to oldest
            for entry in reversed(self._entries):
                # Filter by entry_type
                if query.entry_type is not None:
                    e_type = entry.entry_type.value if isinstance(entry.entry_type, MemoryEntryType) else str(entry.entry_type)
                    q_type = query.entry_type.value if isinstance(query.entry_type, MemoryEntryType) else str(query.entry_type)
                    if e_type != q_type:
                        continue

                # Filter by task_id
                if query.task_id is not None:
                    if entry.task_id != query.task_id and entry.id != query.task_id:
                        continue

                # Filter by target entity
                if query.target is not None:
                    q_target = query.target.strip().lower()
                    e_target = str(entry.content.get("target") or entry.metadata.get("target") or "").lower()
                    if q_target not in e_target:
                        continue

                # Filter by action
                if query.action is not None:
                    q_action = query.action.strip().lower()
                    e_action = str(entry.content.get("action") or entry.metadata.get("action") or "").lower()
                    if q_action != e_action:
                        continue

                # Filter by status
                if query.status is not None:
                    q_status = query.status.strip().lower()
                    e_status = str(entry.content.get("status") or entry.content.get("task_status") or "").lower()
                    if q_status != e_status:
                        continue

                # Filter by free-text match
                if query.text is not None:
                    q_text = query.text.strip().lower()
                    content_str = (
                        str(entry.content) + " " + str(entry.metadata)
                    ).lower()
                    if q_text not in content_str:
                        continue

                results.append(entry)
                if len(results) >= query.limit:
                    break

            return results

    def get_recent(
        self,
        limit: int = 5,
        entry_type: Optional[MemoryEntryType] = None,
    ) -> List[MemoryEntry]:
        with self._lock:
            matched: List[MemoryEntry] = []
            for entry in reversed(self._entries):
                if entry_type is not None:
                    e_type = entry.entry_type.value if isinstance(entry.entry_type, MemoryEntryType) else str(entry.entry_type)
                    q_type = entry_type.value if isinstance(entry_type, MemoryEntryType) else str(entry_type)
                    if e_type != q_type:
                        continue

                matched.append(entry)
                if len(matched) >= limit:
                    break

            return matched

    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        with self._lock:
            for entry in self._entries:
                if entry.id == entry_id:
                    return entry
            return None

    def get_current_task(self) -> Optional[MemoryEntry]:
        """Find the most recent active/requested task entry."""
        with self._lock:
            active_statuses = {"requested", "in_progress", "valid"}
            for entry in reversed(self._entries):
                e_type = entry.entry_type.value if isinstance(entry.entry_type, MemoryEntryType) else str(entry.entry_type)
                if e_type == MemoryEntryType.TASK.value:
                    status = str(
                        entry.content.get("status") or entry.content.get("task_status") or ""
                    ).lower()
                    if status in active_statuses or not status:
                        return entry
            return None

    def update_task_status(self, task_id: str, status: str) -> bool:
        with self._lock:
            for entry in reversed(self._entries):
                if entry.task_id == task_id or entry.id == task_id:
                    entry.content["status"] = status
                    entry.metadata["last_status_update"] = datetime.utcnow().timestamp()
                    self.logger.info(f"Updated status of task '{task_id}' to '{status}'.")
                    return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.logger.info("Cleared all entries from InMemoryStore.")

    def count(self) -> int:
        with self._lock:
            return len(self._entries)
