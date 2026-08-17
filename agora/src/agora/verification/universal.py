"""The universal floor: six domain-agnostic deterministic checks.

These run first, on every artifact, in every domain. They are the floor, not
a fallback. They know nothing about any domain and check properties any
useful output must have:

=============  ===================================================  ==================================================
Check          Question                                             Why it generalises
=============  ===================================================  ==================================================
provenance     Does every checkable claim trace to a source?         An invented statistic is the same failure in a
                                                                     landing page, a board memo and a contract summary
scope          Did it touch anything the contract barred?            Scope is declared per task, not per domain
authority      Did it commit to anything it may not commit to?       Promising a refund is the same overreach in
                                                                     support, sales and legal
consistency    Does the output contradict itself?                    Headline 60%, body 45% needs no domain knowledge
structure      Are declared sections present?                        Schema conformance is universal
abstention     Where an input was missing, did it flag or invent?     The anti-hallucination check for domains with
                                                                     no ground truth
=============  ===================================================  ==================================================

Deterministic checks never hallucinate, cost nothing, and when they fire they
are right -- which is exactly what a Capability Ledger needs to be worth
routing on. Being straight about the limit: these catch *unsourced*, not
*false*. A cited claim that misrepresents its source passes here; that is what
the L2 recomputation tier is for.

A noisy checker is worse than no checker, because people learn to click past
it. Every rule below carries an explicit exemption list for the false
positives it would otherwise produce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence

from ..artifacts import Artifact, sections_present
from ..contracts import AuthorityEnvelope, Scope, TaskContract

# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


class Severity(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    INFO = "info"


@dataclass(frozen=True)
class Finding:
    """One thing a check objected to, with the evidence that triggered it."""

    check: str
    severity: Severity
    message: str
    evidence: str = ""
    location: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "severity", Severity(self.severity))

    def __str__(self) -> str:
        where = f" [{self.location}]" if self.location else ""
        ev = f" -- {self.evidence!r}" if self.evidence else ""
        return f"{self.severity.value.upper()} {self.check}{where}: {self.message}{ev}"


@dataclass
class CheckResult:
    check: str
    findings: list[Finding] = field(default_factory=list)
    model_calls: int = 0
    cost_usd: float = 0.0
    skipped: bool = False
    note: str = ""

    @property
    def passed(self) -> bool:
        return not any(f.severity is Severity.BLOCK for f in self.findings)


@dataclass
class FloorReport:
    """The result of running the universal floor over one artifact."""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def findings(self) -> list[Finding]:
        return [f for r in self.results for f in r.findings]

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCK]

    @property
    def passed(self) -> bool:
        return not self.blocking

    @property
    def model_calls(self) -> int:
        return sum(r.model_calls for r in self.results)

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.results)

    def by_check(self, name: str) -> CheckResult | None:
        for r in self.results:
            if r.check == name:
                return r
        return None

    def summary(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        return (
            f"universal floor {state} · {len(self.findings)} finding(s) · "
            f"{self.model_calls} model calls · ${self.cost_usd:.4f}"
        )


# --------------------------------------------------------------------------
# Shared text utilities
# --------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

_STOPWORDS = frozenset(
    """a an the and or but if while of to in on at by for with from as is are was
    were be been being this that these those it its our your their we you they he
    she them us i not no so than then there here about into over under after before
    can could may might must shall should will would have has had do does did"""
    .split()
)

#: Words that mark a value as scoped to a particular thing. When two numeric
#: mentions carry *different* discriminators they are describing different
#: things, and a difference between them is not a contradiction.
_DISCRIMINATOR = re.compile(
    r"(?i)\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?|q[1-4]|fy\d{2,4}|19\d{2}|20\d{2}|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"baseline|target|forecast|actual|projected|previous|prior|last|current|"
    r"best[- ]case|worst[- ]case)\b"
)
# Note: "before"/"after" are deliberately *not* discriminators. They occur in
# ordinary prose constantly, and treating them as scoping words suppressed a
# genuine contradiction ("phase 2 runs 6 weeks" vs "phase 2 needs 8 weeks
# before handover") in testing.


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z\-']+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _normalise_number(raw: str) -> float | None:
    cleaned = raw.replace(",", "").replace("$", "").replace("£", "").replace("€", "")
    cleaned = cleaned.replace("%", "").strip()
    multiplier = 1.0
    if cleaned and cleaned[-1].lower() in "kmb":
        multiplier = {"k": 1e3, "m": 1e6, "b": 1e9}[cleaned[-1].lower()]
        cleaned = cleaned[:-1]
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def _numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for raw in re.findall(r"[$£€]?\d[\d,]*(?:\.\d+)?\s?[kmbKMB]?%?", text):
        value = _normalise_number(raw)
        if value is not None:
            found.add(value)
            if raw.strip().endswith("%"):
                found.add(value / 100.0)
    return found


# --------------------------------------------------------------------------
# 1 · Provenance
# --------------------------------------------------------------------------

#: A claim worth checking carries a measurement, not just a digit. Section
#: numbers, phase ordinals, years and version strings are excluded because
#: flagging them is how a checker becomes noise.
_NUMBER_CORE = r"\d(?:[\d,]*\d)?(?:\.\d+)?\s?[kmbKMB]?"

_CLAIMY_NUMBER = re.compile(
    rf"""(?x)
    (?P<pct>\b{_NUMBER_CORE}\s?%)                        # 45%, 12.5 %
  | (?P<money>[$£€]\s?{_NUMBER_CORE})                    # $1,200  $4.5M
  | (?P<mult>\b\d+(?:\.\d+)?\s?[x×]\b)                   # 3x, 46×
  | (?P<counted>\b{_NUMBER_CORE}\s+
      (?:customers?|users?|companies|organisations?|organizations?|
         respondents?|employees?|hours?|days?|weeks?|months?|
         seconds?|ms|minutes?|tickets?|deals?|accounts?|leads?))
    """
)

#: The numeric head of a claimy match -- the counted form ends in a unit word
#: ("4,000 companies"), which will not parse as a number on its own.
_NUMBER_HEAD = re.compile(rf"[$£€]?\s?{_NUMBER_CORE}\s?%?")

_ATTRIBUTION_NO_CITE = re.compile(
    r"(?i)\b(studies show|research (?:shows|indicates|suggests)|"
    r"industry (?:data|benchmarks?) (?:shows?|suggests?)|"
    r"it is (?:well )?known that|experts agree|surveys? (?:show|found)|"
    r"analysts? (?:say|estimate))\b"
)

_CITATION = re.compile(r"\[(?P<id>[A-Za-z0-9_.\-]+)\]|\(source:\s*(?P<id2>[^)]+)\)")

#: Ordinal/label contexts where a bare number is a name, not a measurement.
_LABEL_CONTEXT = re.compile(
    r"(?i)\b(phase|step|stage|section|part|item|tier|version|v|chapter|figure|"
    r"table|appendix|option|q|sprint|week)\s*$"
)


def _is_authorised_offer(sentence: str, granted_offers: Sequence[str]) -> bool:
    """An offer you are permitted to make is not a claim about the world.

    This exemption exists because the check produced exactly this false
    positive on a clean control document: an authorised discount was flagged
    as an unsourced statistic. A commitment the contract grants is a
    *performative*, not an assertion, and the provenance check has no business
    with it.
    """
    lowered = sentence.lower()
    for offer in granted_offers:
        offer_l = offer.lower().strip()
        if not offer_l:
            continue
        if offer_l in lowered:
            return True
        offer_numbers = _numbers_in(offer_l)
        offer_words = content_words(offer_l)
        sent_words = content_words(lowered)
        if offer_numbers and offer_numbers <= _numbers_in(lowered):
            overlap = len(offer_words & sent_words)
            if overlap >= max(1, min(2, len(offer_words))):
                return True
    return False


def check_provenance(
    artifact: Artifact,
    contract: TaskContract | None = None,
    *,
    granted_offers: Sequence[str] = (),
) -> CheckResult:
    """Does every checkable claim trace to a supplied source?"""
    result = CheckResult(check="provenance")
    offers = tuple(granted_offers) or (contract.granted_offers if contract else ())

    supplied = _numbers_in(artifact.source_text())
    for value in (contract.inputs.values() if contract else []):
        supplied |= _numbers_in(str(value))
    source_ids = artifact.source_ids()

    for sentence in sentences(artifact.content):
        if _is_authorised_offer(sentence, offers):
            continue

        cited_ids = {
            (m.group("id") or m.group("id2") or "").strip()
            for m in _CITATION.finditer(sentence)
        }
        has_valid_citation = bool(cited_ids & source_ids)

        for match in _CLAIMY_NUMBER.finditer(sentence):
            raw = match.group(0).strip()
            prefix = sentence[: match.start()]
            if _LABEL_CONTEXT.search(prefix.rstrip()):
                continue
            head = _NUMBER_HEAD.match(raw)
            value = _normalise_number(head.group(0)) if head else None
            if value is None:
                continue
            candidates = {value}
            if raw.endswith("%"):
                candidates.add(value / 100.0)
            if candidates & supplied:
                continue
            if has_valid_citation:
                continue
            result.findings.append(
                Finding(
                    check="provenance",
                    severity=Severity.BLOCK,
                    message=(
                        f"the figure {raw!r} does not trace to any supplied source"
                    ),
                    evidence=sentence.strip(),
                )
            )

        attribution = _ATTRIBUTION_NO_CITE.search(sentence)
        if attribution and not has_valid_citation:
            result.findings.append(
                Finding(
                    check="provenance",
                    severity=Severity.BLOCK,
                    message=(
                        f"appeal to authority ({attribution.group(0)!r}) with no "
                        f"source cited"
                    ),
                    evidence=sentence.strip(),
                )
            )
    return result


# --------------------------------------------------------------------------
# 2 · Scope
# --------------------------------------------------------------------------


def check_scope(
    artifact: Artifact, contract: TaskContract | None = None, *, scope: Scope | None = None
) -> CheckResult:
    """Did it touch anything the contract barred?"""
    result = CheckResult(check="scope")
    scope = scope or (contract.scope if contract else None)
    if scope is None or (not scope.barred and not scope.allowed):
        result.skipped = True
        result.note = "no scope declared"
        return result

    for target in artifact.touched:
        if scope.bars(target):
            result.findings.append(
                Finding(
                    check="scope",
                    severity=Severity.BLOCK,
                    message=f"touched {target!r}, which the contract bars",
                    evidence=target,
                )
            )
        elif not scope.allows(target):
            result.findings.append(
                Finding(
                    check="scope",
                    severity=Severity.BLOCK,
                    message=f"touched {target!r}, outside the declared scope",
                    evidence=target,
                )
            )

    for pattern in scope.barred:
        literal = pattern.replace("*", "").strip()
        if len(literal) >= 4 and literal.lower() in artifact.content.lower():
            snippet = next(
                (s for s in sentences(artifact.content) if literal.lower() in s.lower()),
                literal,
            )
            result.findings.append(
                Finding(
                    check="scope",
                    severity=Severity.WARN,
                    message=f"content references barred target {pattern!r}",
                    evidence=snippet.strip(),
                )
            )
    return result


# --------------------------------------------------------------------------
# 3 · Authority
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitmentPattern:
    """A shape of language that commits the organisation to something."""

    name: str
    permission: str
    pattern: re.Pattern[str]
    message: str


_COMMITMENTS: tuple[CommitmentPattern, ...] = (
    CommitmentPattern(
        "discount",
        "commerce.offer_discount",
        re.compile(
            r"(?i)\b(?:offer(?:ing)?|extend(?:ing)?|give|giving|apply(?:ing)?|"
            r"happy to (?:offer|extend)|we(?:'| a)?(?:re|ll)? (?:offer|extend|"
            r"knock|take))\b[^.]{0,60}?\b\d[\d.]*\s?%\s*(?:discount|off|reduction)"
            r"|\b\d[\d.]*\s?%\s*(?:discount|off)\b"
        ),
        "offered a price concession",
    ),
    CommitmentPattern(
        "refund",
        "commerce.issue_refund",
        re.compile(r"(?i)\b(?:full |partial )?refund(?:ed|ing)?\b(?![^.]*\bpolicy\b)"),
        "promised a refund",
    ),
    CommitmentPattern(
        "delivery_guarantee",
        "commit.delivery",
        re.compile(
            r"(?i)\b(?:we |i )?(?:guarantee|guaranteed|guarantees|assure you|"
            r"promise)\b[^.]{0,80}?\b(?:deliver|delivery|ship|shipping|ready|"
            r"complete|available|live|working)\b"
            r"|\b(?:guaranteed)\s+(?:delivery|ship date|turnaround|uptime)\b"
        ),
        "gave a delivery guarantee",
    ),
    CommitmentPattern(
        "deadline",
        "commit.deadline",
        re.compile(
            r"(?i)\b(?:we|i)\s*(?:'ll|will| shall)\s+(?:have|deliver|ship|send|"
            r"complete|finish|hand)\b[^.]{0,60}?\b(?:by|before|on)\s+"
            r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
            r"end of (?:day|week|month)|eod|eow|\d{1,2}\s?(?:st|nd|rd|th)?\s+"
            r"\w+|next week)"
        ),
        "committed to a delivery date",
    ),
    CommitmentPattern(
        "legal_advice",
        "legal.advise",
        re.compile(
            r"(?i)\b(?:you should (?:sign|accept|agree to|proceed with|not sign)|"
            r"we advise you to|our (?:legal )?advice is|"
            r"you are (?:legally )?(?:obliged|required|entitled)|"
            r"this (?:is|constitutes) (?:legally )?binding|"
            r"it is safe to sign)\b"
        ),
        "gave legal advice",
    ),
    CommitmentPattern(
        "pricing",
        "commerce.set_price",
        re.compile(
            r"(?i)\b(?:price (?:will be|is set at|locked at)|we (?:can|will) do it for)"
            r"\s*[$£€]?\s?\d"
        ),
        "set or committed a price",
    ),
    CommitmentPattern(
        "send",
        "email.send",
        re.compile(r"(?i)\b(?:i|we) (?:have |'ve )?(?:sent|emailed|dispatched)\b"),
        "claimed to have sent something",
    ),
    CommitmentPattern(
        "contract_terms",
        "legal.commit_terms",
        re.compile(
            r"(?i)\b(?:we (?:agree|commit) to|this agreement (?:grants|entitles))\b"
        ),
        "agreed contractual terms",
    ),
)


def check_authority(
    artifact: Artifact,
    contract: TaskContract | None = None,
    *,
    authority: AuthorityEnvelope | None = None,
    granted_offers: Sequence[str] = (),
) -> CheckResult:
    """Did it commit to anything it may not commit to?"""
    result = CheckResult(check="authority")
    envelope = authority or (contract.authority if contract else AuthorityEnvelope())
    offers = tuple(granted_offers) or (contract.granted_offers if contract else ())

    for sentence in sentences(artifact.content):
        for commitment in _COMMITMENTS:
            match = commitment.pattern.search(sentence)
            if not match:
                continue
            if envelope.permits(commitment.permission):
                continue
            if _is_authorised_offer(sentence, offers):
                continue
            result.findings.append(
                Finding(
                    check="authority",
                    severity=Severity.BLOCK,
                    message=(
                        f"{commitment.message} without {commitment.permission!r} "
                        f"in the authority envelope"
                    ),
                    evidence=sentence.strip(),
                    location=commitment.name,
                )
            )
    return result


# --------------------------------------------------------------------------
# 4 · Consistency
# --------------------------------------------------------------------------

_MEASURE = re.compile(
    r"""(?x)
    (?P<value>[$£€]?\s?\d[\d,]*(?:\.\d+)?\s?[kmbKMB]?)
    \s*
    (?P<unit>%|percent|weeks?|days?|months?|hours?|years?|x|×)?
    """
)

_UNIT_ALIASES = {
    "percent": "%",
    "×": "x",
    "week": "weeks",
    "day": "days",
    "month": "months",
    "hour": "hours",
    "year": "years",
}


@dataclass(frozen=True)
class _Mention:
    value: float
    unit: str
    label: frozenset[str]
    discriminators: frozenset[str]
    sentence: str


def _mentions(text: str) -> list[_Mention]:
    out: list[_Mention] = []
    for sentence in sentences(text):
        for match in _MEASURE.finditer(sentence):
            value = _normalise_number(match.group("value"))
            if value is None:
                continue
            unit = (match.group("unit") or "").strip().lower()
            unit = _UNIT_ALIASES.get(unit, unit)
            if not unit:
                if match.group("value").strip().startswith(("$", "£", "€")):
                    unit = "currency"
                else:
                    continue  # a bare integer is a name more often than a measure
            window = sentence[max(0, match.start() - 90) : match.end() + 90]
            label = content_words(window) - {"percent", "weeks", "days", "months"}
            discriminators = frozenset(
                m.group(0).lower() for m in _DISCRIMINATOR.finditer(window)
            )
            out.append(
                _Mention(
                    value=value,
                    unit=unit,
                    label=frozenset(label),
                    discriminators=discriminators,
                    sentence=sentence.strip(),
                )
            )
    return out


def _labels_agree(a: _Mention, b: _Mention) -> bool:
    """True when two mentions plausibly describe the same quantity."""
    shared = a.label & b.label
    if len(shared) < 2:
        return False
    union = a.label | b.label
    if len(shared) / max(1, len(union)) < 0.18:
        return False
    # Different discriminators (months, quarters, scenarios) mean different
    # things being measured -- a difference there is data, not a contradiction.
    if a.discriminators != b.discriminators and (a.discriminators or b.discriminators):
        return False
    return True


def check_consistency(
    artifact: Artifact, contract: TaskContract | None = None
) -> CheckResult:
    """Does the output contradict itself?"""
    result = CheckResult(check="consistency")
    mentions = _mentions(artifact.content)
    seen: set[tuple[str, str]] = set()

    for i, a in enumerate(mentions):
        for b in mentions[i + 1 :]:
            if a.unit != b.unit or abs(a.value - b.value) < 1e-9:
                continue
            if a.sentence == b.sentence:
                continue
            if not _labels_agree(a, b):
                continue
            key = tuple(sorted((a.sentence, b.sentence)))
            if key in seen:
                continue
            seen.add(key)
            unit = "" if a.unit == "currency" else (
                a.unit if a.unit in ("%", "x") else f" {a.unit}"
            )
            result.findings.append(
                Finding(
                    check="consistency",
                    severity=Severity.BLOCK,
                    message=(
                        f"the same quantity is stated as {a.value:g}{unit} and "
                        f"{b.value:g}{unit}"
                    ),
                    evidence=f"{a.sentence} || {b.sentence}",
                    location=", ".join(sorted(a.label & b.label)[:3]),
                )
            )
    return result


# --------------------------------------------------------------------------
# 5 · Structure
# --------------------------------------------------------------------------


def check_structure(
    artifact: Artifact,
    contract: TaskContract | None = None,
    *,
    required_sections: Sequence[str] = (),
) -> CheckResult:
    """Are declared sections present?"""
    result = CheckResult(check="structure")
    required = tuple(required_sections) or (
        contract.required_sections if contract else ()
    )
    if not required:
        result.skipped = True
        result.note = "no structure declared"
        return result

    _, missing = sections_present(artifact, required)
    for name in missing:
        result.findings.append(
            Finding(
                check="structure",
                severity=Severity.BLOCK,
                message=f"required section {name!r} is missing",
                evidence=", ".join(artifact.sections) or "(no sections found)",
            )
        )
    return result


# --------------------------------------------------------------------------
# 6 · Abstention
# --------------------------------------------------------------------------

_HEDGE = re.compile(
    r"(?i)\b(not (?:provided|specified|supplied|available|stated|included|given)|"
    r"unavailable|unknown|unspecified|missing|absent|silent on|does not (?:state|"
    r"specify|say)|cannot (?:be )?(?:determine|determined|confirm|confirmed|verify)|"
    r"no (?:data|information|figure|source|value|clause)|tbd|to be (?:confirmed|"
    r"determined)|withheld|redacted|out of scope|requires (?:confirmation|input))\b"
)

_ASSERTS_VALUE = re.compile(
    r"(?i)(?:\bis\b|\bare\b|\bwill be\b|\bshall be\b|:\s*\S|\bset to\b|"
    r"\bgoverned by\b|\bequal to\b|\bamounts? to\b)"
)


def check_abstention(
    artifact: Artifact,
    contract: TaskContract | None = None,
    *,
    unavailable_inputs: Sequence[str] = (),
) -> CheckResult:
    """Where an input was missing, did it flag it or invent one?

    This is the anti-hallucination check for domains with no ground truth. It
    is the only one of the six that needs the contract to declare what was
    *not* available -- which is why declaring that is part of contract
    authoring rather than an optional extra.
    """
    result = CheckResult(check="abstention")
    missing_fields = tuple(unavailable_inputs) or artifact.unavailable_inputs
    if not missing_fields and contract is not None:
        missing_fields = tuple(
            k for k, v in contract.inputs.items() if v is None or v == ""
        )
    if not missing_fields:
        result.skipped = True
        result.note = "no inputs declared unavailable"
        return result

    for field_name in missing_fields:
        keywords = content_words(field_name.replace("_", " ")) or {field_name.lower()}
        mentions = [
            s
            for s in sentences(artifact.content)
            if keywords & content_words(s)
        ]
        if not mentions:
            result.findings.append(
                Finding(
                    check="abstention",
                    severity=Severity.WARN,
                    message=(
                        f"input {field_name!r} was unavailable and the output "
                        f"neither flags nor mentions it"
                    ),
                    evidence="",
                    location=field_name,
                )
            )
            continue
        if any(_HEDGE.search(s) for s in mentions):
            continue
        invented = [s for s in mentions if _ASSERTS_VALUE.search(s)]
        if invented:
            result.findings.append(
                Finding(
                    check="abstention",
                    severity=Severity.BLOCK,
                    message=(
                        f"input {field_name!r} was unavailable, but the output "
                        f"asserts a value for it"
                    ),
                    evidence=invented[0].strip(),
                    location=field_name,
                )
            )
    return result


# --------------------------------------------------------------------------
# The floor
# --------------------------------------------------------------------------

UNIVERSAL_CHECKS: tuple[str, ...] = (
    "provenance",
    "scope",
    "authority",
    "consistency",
    "structure",
    "abstention",
)

_CHECK_FNS: dict[str, Callable[..., CheckResult]] = {
    "provenance": check_provenance,
    "scope": check_scope,
    "authority": check_authority,
    "consistency": check_consistency,
    "structure": check_structure,
    "abstention": check_abstention,
}


def run_universal_floor(
    artifact: Artifact,
    contract: TaskContract | None = None,
    *,
    checks: Iterable[str] = UNIVERSAL_CHECKS,
    sanitize: bool = True,
) -> FloorReport:
    """Run the six checks. Zero model calls, zero marginal cost.

    ``sanitize`` strips instruction-shaped content from untrusted sources
    before checking, which is what the control plane does at every handoff.
    """
    subject = artifact.sanitized() if sanitize else artifact
    report = FloorReport()
    for name in checks:
        fn = _CHECK_FNS[name]
        report.results.append(fn(subject, contract))
    return report


__all__ = [
    "Finding",
    "Severity",
    "CheckResult",
    "FloorReport",
    "UNIVERSAL_CHECKS",
    "run_universal_floor",
    "check_provenance",
    "check_scope",
    "check_authority",
    "check_consistency",
    "check_structure",
    "check_abstention",
]
