"""Tool allowlist, content sanitization, groundedness check and prediction disclaimers."""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from bs4 import BeautifulSoup

ALLOWED_TOOLS = frozenset({"get_portfolio", "get_quote", "get_historical_candles"})

DISCLAIMER = (
    "Not financial advice. Figures come from your portfolio data and the cited sources; "
    "any estimate is illustrative only."
)
PREDICTION_LABEL = "Illustrative estimate, not financial advice."


class ToolNotAllowedError(Exception):
    pass


class ToolCallError(Exception):
    pass


def check_tool_allowed(name: str) -> None:
    if name not in ALLOWED_TOOLS:
        raise ToolNotAllowedError(
            f"Tool '{name}' is not on the read-only allowlist ({', '.join(sorted(ALLOWED_TOOLS))})."
        )


def parse_tool_output(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        text = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    else:
        text = str(raw)
    if text.startswith("Error executing tool"):
        raise ToolCallError(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolCallError(f"Tool returned non-JSON output: {text[:200]}") from exc
    if not isinstance(parsed, dict):
        raise ToolCallError("Tool returned JSON that is not an object.")
    return parsed


class GuardedToolbox:
    """Holds only allowlisted tools, checks every call against the allowlist and logs it."""

    def __init__(self, tools: Iterable):
        self._tools = {}
        self.rejected_at_load: list[str] = []
        for tool in tools:
            if tool.name in ALLOWED_TOOLS:
                self._tools[tool.name] = tool
            else:
                self.rejected_at_load.append(tool.name)
        self.log: list[dict] = []

    def tools(self) -> list:
        return list(self._tools.values())

    async def call(self, name: str, args: dict | None = None) -> dict:
        args = args or {}
        entry = {"tool": name, "args": args}
        started = time.perf_counter()
        try:
            check_tool_allowed(name)
            if name not in self._tools:
                raise ToolCallError(f"Tool '{name}' is not available from the MCP server.")
            result = parse_tool_output(await self._tools[name].ainvoke(args))
            entry["status"] = "ok"
            return result
        except ToolNotAllowedError as exc:
            entry.update(status="blocked", error=str(exc))
            raise
        except ToolCallError as exc:
            entry.update(status="error", error=str(exc))
            raise
        except Exception as exc:
            entry.update(status="error", error=str(exc))
            raise ToolCallError(str(exc)) from exc
        finally:
            entry["duration_ms"] = round((time.perf_counter() - started) * 1000)
            self.log.append(entry)

    def drain(self) -> list[dict]:
        entries, self.log = self.log, []
        return entries


INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier|all)\b.{0,40}\b(instructions?|prompts?|rules?)\b",
    r"\byou are now\b",
    r"\bsystem prompt\b",
    r"\b(new|updated|override) instructions?\b",
    r"\bpretend (to be|you are)\b",
    r"\bact as (an?|the)\b",
    r"\b(call|use|invoke) the \w+ tool\b",
    r"\bplace (an? )?(buy|sell)? ?orders?\b",
    r"</?\s*(system|assistant|user|instructions?)\s*>",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_ROLE_BLOCK_RE = re.compile(r"<\s*(system|assistant|user|instructions?)\b[^>]*>.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class SanitizedText:
    text: str
    removed: list[str] = field(default_factory=list)


def sanitize_text(raw: str, max_chars: int = 4000) -> SanitizedText:
    """Strip HTML and drop sentences that look like instructions aimed at the model."""
    removed = [m.group(0) for m in _ROLE_BLOCK_RE.finditer(raw or "")]
    text = _ROLE_BLOCK_RE.sub(" ", raw or "")
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"\s+", " ", text).strip()
    kept = []
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence.strip():
            continue
        (removed if _INJECTION_RE.search(sentence) else kept).append(sentence.strip())
    return SanitizedText(text=" ".join(kept)[:max_chars], removed=removed)


_CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
_NUMBER_RE = re.compile(r"(?P<currency>(?:₹|rs\.?|inr)\s?)?(?P<num>\d[\d,]*(?:\.\d+)?)(?P<pct>\s?%|\s?percent)?", re.IGNORECASE)


@dataclass
class ExtractedNumber:
    raw: str
    value: float
    decimals: int
    is_percent: bool
    is_currency: bool

    @property
    def is_structural(self) -> bool:
        """Small counts, days and years that never need a source."""
        if self.is_percent or self.is_currency or self.decimals:
            return False
        return self.value <= 31 or 1990 <= self.value <= 2100


def extract_numbers(text: str) -> list[ExtractedNumber]:
    cleaned = _CITATION_RE.sub(" ", text or "")
    numbers = []
    for m in _NUMBER_RE.finditer(cleaned):
        start = m.start("num")
        prev = cleaned[start - 1] if start > 0 else ""
        prev2 = cleaned[start - 2] if start > 1 else ""
        # Skip digits inside identifiers such as ISINs, "Q2" or "COVID-19".
        if not m.group("currency") and (prev.isalpha() or prev == "_" or (prev == "-" and prev2.isalnum())):
            continue
        num = m.group("num").replace(",", "")
        if num.endswith("."):
            num = num[:-1]
        decimals = len(num.split(".")[1]) if "." in num else 0
        numbers.append(ExtractedNumber(
            raw=m.group(0).strip(),
            value=float(num),
            decimals=decimals,
            is_percent=bool(m.group("pct")),
            is_currency=bool(m.group("currency")),
        ))
    return numbers


def numbers_in(values: Iterable[Any]) -> set[float]:
    """Collect every number from facts (numbers, strings, nested dicts and lists)."""
    found: set[float] = set()
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            found.add(abs(float(value)))
        elif isinstance(value, str):
            found.update(n.value for n in extract_numbers(value))
        elif isinstance(value, dict):
            found |= numbers_in(value.values())
        elif isinstance(value, (list, tuple, set)):
            found |= numbers_in(value)
    return found


@dataclass
class GroundednessResult:
    passed: bool
    ungrounded: list[str]
    checked: int


def check_groundedness(text: str, allowed: Iterable[float]) -> GroundednessResult:
    """Every non-structural number in `text` must be a rounding of some allowed value."""
    allowed_values = [abs(a) for a in allowed]
    ungrounded, checked = [], 0
    for number in extract_numbers(text):
        if number.is_structural:
            continue
        checked += 1
        tolerance = 0.5 * 10 ** (-number.decimals) + 1e-9
        if not any(abs(a - number.value) <= tolerance for a in allowed_values):
            ungrounded.append(number.raw)
    return GroundednessResult(passed=not ungrounded, ungrounded=ungrounded, checked=checked)


def with_disclaimer(text: str) -> str:
    return text if DISCLAIMER in text else f"{text.rstrip()}\n\n_{DISCLAIMER}_"
