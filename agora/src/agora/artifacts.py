"""Artifacts, sources and the untrusted boundary.

Agents put artifacts on the blackboard; verifiers read them. Two properties
of this module carry weight beyond bookkeeping:

*Reference passing.* An artifact is addressed by URI. Handing a downstream
agent a payload instead of a reference is the single largest avoidable cost in
a multi-agent system (see :mod:`agora.economics`).

*The untrusted boundary.* Retrieved content is tagged as data, never as
instruction, and instruction-shaped text is stripped at every handoff. A
downstream confidence is clamped to the weakest upstream input, so a chain of
confident restatements cannot launder a shaky source into a firm claim.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class Trust(str, Enum):
    """How much weight a source's content may carry."""

    PRIMARY = "primary"        # the system of record
    SECONDARY = "secondary"    # a derived but named source
    UNTRUSTED = "untrusted"    # anything retrieved from the open web

    @property
    def ceiling(self) -> float:
        """The most confidence any claim resting on this source may carry."""
        return {
            Trust.PRIMARY: 1.0,
            Trust.SECONDARY: 0.90,
            Trust.UNTRUSTED: 0.86,
        }[self]


@dataclass(frozen=True)
class Source:
    """A piece of supplied evidence a claim may trace back to."""

    id: str
    text: str
    uri: str | None = None
    trust: Trust = Trust.SECONDARY
    retrieved_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trust", Trust(self.trust))


# --------------------------------------------------------------------------
# Instruction stripping
# --------------------------------------------------------------------------

#: Text shapes that look like instructions to a model rather than content.
_INSTRUCTION_SHAPES = (
    re.compile(r"(?im)^\s*(?:system|assistant|user)\s*:\s*.*$"),
    re.compile(r"(?is)<\s*/?\s*(?:system|instructions?|prompt)[^>]*>"),
    re.compile(
        r"(?im)^.*\b(?:ignore|disregard|forget|override)\b[^.\n]*\b"
        r"(?:previous|prior|above|earlier|all)\b[^.\n]*"
        r"(?:instruction|prompt|rule|direction|context)s?\b.*$"
    ),
    re.compile(
        r"(?im)^.*\byou\s+(?:are|must|should|will)\s+now\b[^.\n]*"
        r"(?:instead|act as|pretend|roleplay)\b.*$"
    ),
    re.compile(
        r"(?im)^.*\b(?:new|updated|revised)\s+(?:instructions?|system prompt|rules?)\b"
        r"\s*[:.-].*$"
    ),
    re.compile(r"(?im)^\s*###\s*(?:instruction|system)\b.*$"),
)


@dataclass
class StripResult:
    text: str
    removed: list[str] = field(default_factory=list)

    @property
    def was_modified(self) -> bool:
        return bool(self.removed)


def strip_instruction_shaped(text: str) -> StripResult:
    """Remove instruction-shaped spans from retrieved content.

    Retrieved content is *data*. Anything in it that reads as a directive to
    the reader is removed before the content crosses a handoff, and what was
    removed is recorded so the removal itself is auditable.
    """
    removed: list[str] = []
    cleaned = text
    for pattern in _INSTRUCTION_SHAPES:
        def _capture(match: re.Match[str]) -> str:
            span = match.group(0).strip()
            if span:
                removed.append(span)
            return "[stripped: instruction-shaped content]"

        cleaned = pattern.sub(_capture, cleaned)
    return StripResult(text=cleaned, removed=removed)


def clamp_confidence(
    own_confidence: float, upstream: Iterable[float] = ()
) -> float:
    """Clamp a confidence to the weakest input it rests on."""
    values = [float(own_confidence), *[float(u) for u in upstream]]
    return max(0.0, min(1.0, min(values)))


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------


@dataclass
class Artifact:
    """A unit of produced work, addressable by URI and verifiable in isolation."""

    content: str
    capability: str = ""
    producer: str = ""
    task_id: str | None = None
    sections: dict[str, str] = field(default_factory=dict)
    sources: tuple[Source, ...] = ()
    touched: tuple[str, ...] = ()
    unavailable_inputs: tuple[str, ...] = ()
    confidence: float = 1.0
    reasoning: str | None = None
    kind: str = "text"
    id: str = field(default_factory=lambda: f"art-{uuid.uuid4().hex[:12]}")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sources = tuple(self.sources)
        self.touched = tuple(self.touched)
        self.unavailable_inputs = tuple(self.unavailable_inputs)
        if not self.sections:
            self.sections = extract_sections(self.content)

    # -- addressing --------------------------------------------------------

    @property
    def uri(self) -> str:
        return f"artifact://{self.id}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def reference(self) -> dict[str, Any]:
        """The handle passed between agents in place of the payload."""
        return {
            "uri": self.uri,
            "digest": self.digest,
            "kind": self.kind,
            "capability": self.capability,
            "producer": self.producer,
            "bytes": len(self.content.encode("utf-8")),
            "confidence": self.confidence,
        }

    # -- the untrusted boundary -------------------------------------------

    def sanitized(self) -> "Artifact":
        """A copy with instruction-shaped content stripped from every source."""
        clean_sources: list[Source] = []
        removed: list[str] = []
        for source in self.sources:
            if source.trust is Trust.UNTRUSTED:
                result = strip_instruction_shaped(source.text)
                removed.extend(result.removed)
                clean_sources.append(replace(source, text=result.text))
            else:
                clean_sources.append(source)
        body = strip_instruction_shaped(self.content)
        removed.extend(body.removed)
        ceiling = clamp_confidence(
            self.confidence, (s.trust.ceiling for s in clean_sources)
        )
        meta = dict(self.metadata)
        if removed:
            meta["stripped_spans"] = removed
        return replace(
            self,
            content=body.text,
            sources=tuple(clean_sources),
            confidence=ceiling,
            metadata=meta,
            sections=extract_sections(body.text),
        )

    def blind_view(self) -> "Artifact":
        """The artifact as a blind verifier sees it: no producer reasoning.

        The verifier receives the artifact and the acceptance criteria and
        nothing else. Showing it the producer's reasoning is how a verifier
        gets talked into a bad answer.
        """
        return replace(self, reasoning=None, producer="<blinded>")

    # -- convenience -------------------------------------------------------

    def source_text(self) -> str:
        return "\n".join(s.text for s in self.sources)

    def source_ids(self) -> set[str]:
        return {s.id for s in self.sources}

    def with_sections(self, sections: Mapping[str, str]) -> "Artifact":
        return replace(self, sections=dict(sections))


_HEADING = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s+(?P<hash>.+?)|(?P<bold>\*\*.+?\*\*))\s*$")


def extract_sections(content: str) -> dict[str, str]:
    """Split markdown-ish content into ``{heading: body}``."""
    sections: dict[str, str] = {}
    matches = list(_HEADING.finditer(content))
    for i, match in enumerate(matches):
        title = (match.group("hash") or match.group("bold") or "").strip().strip("*")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        sections[title] = content[start:end].strip()
    return sections


def sections_present(
    artifact: Artifact, required: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Return ``(present, missing)`` for the required section names."""
    have = {k.strip().lower() for k in artifact.sections}
    present, missing = [], []
    for name in required:
        key = name.strip().lower()
        if key in have or any(key in h for h in have):
            present.append(name)
        else:
            missing.append(name)
    return present, missing


__all__ = [
    "Artifact",
    "Source",
    "Trust",
    "StripResult",
    "strip_instruction_shaped",
    "clamp_confidence",
    "extract_sections",
    "sections_present",
]
