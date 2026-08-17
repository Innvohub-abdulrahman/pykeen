"""The blackboard: artifacts by reference, never by payload.

Agents publish here and read from here. What crosses a handoff is a reference
-- a URI, a digest, a size, a confidence -- not the bytes. Passing payloads
between agents is the single largest avoidable cost in a multi-agent system,
and it is avoidable by construction rather than by discipline: the API for
reading another agent's work returns a reference, and fetching the body is a
separate, deliberate call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from .artifacts import Artifact


@dataclass
class Entry:
    artifact: Artifact
    task_id: str | None
    producer: str
    superseded_by: str | None = None

    @property
    def live(self) -> bool:
        return self.superseded_by is None


class Blackboard:
    """Shared, append-mostly artifact store."""

    def __init__(self) -> None:
        self._entries: dict[str, Entry] = {}
        self.bytes_published = 0
        self.bytes_passed_by_reference = 0

    # -- writing -----------------------------------------------------------

    def publish(self, artifact: Artifact, *, supersedes: str | None = None) -> str:
        entry = Entry(
            artifact=artifact,
            task_id=artifact.task_id,
            producer=artifact.producer,
        )
        self._entries[artifact.uri] = entry
        self.bytes_published += len(artifact.content.encode("utf-8"))
        if supersedes and supersedes in self._entries:
            self._entries[supersedes].superseded_by = artifact.uri
        return artifact.uri

    # -- reading -----------------------------------------------------------

    def reference(self, uri: str) -> Mapping[str, Any]:
        """The cheap read. This is what gets handed to another agent."""
        ref = self._entries[uri].artifact.reference()
        self.bytes_passed_by_reference += len(str(ref).encode("utf-8"))
        return ref

    def fetch(self, uri: str) -> Artifact:
        """The expensive read. Deliberately a separate call."""
        return self._entries[uri].artifact

    def get(self, uri: str) -> Artifact | None:
        entry = self._entries.get(uri)
        return entry.artifact if entry else None

    def for_task(self, task_id: str) -> list[Artifact]:
        return [e.artifact for e in self._entries.values() if e.task_id == task_id]

    def live(self) -> list[Artifact]:
        return [e.artifact for e in self._entries.values() if e.live]

    def references_for(self, uris: Sequence[str]) -> list[Mapping[str, Any]]:
        return [self.reference(u) for u in uris if u in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[Artifact]:
        return (e.artifact for e in self._entries.values())

    # -- accounting --------------------------------------------------------

    def savings_ratio(self) -> float:
        """Bytes that would have been passed inline, over bytes actually passed.

        A crude number, but it makes the reference-passing lever visible in
        the same place the artifacts live.
        """
        if not self.bytes_passed_by_reference:
            return 0.0
        return self.bytes_published / self.bytes_passed_by_reference


__all__ = ["Blackboard"]
