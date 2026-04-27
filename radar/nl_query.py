from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ParsedQuery:
    search_text: str
    days: int | None
    limit: int | None
    category: str | None


_TimeConverter = Callable[[re.Match[str]], int]
_PatternSpec = tuple[re.Pattern[str], _TimeConverter]


def _to_days(multiplier: int) -> _TimeConverter:
    def convert(match: re.Match[str]) -> int:
        return int(match.group(2)) * multiplier

    return convert


_TIME_PATTERNS: tuple[_PatternSpec, ...] = (
    (re.compile(r"(최근|지난)\s*(\d+)\s*일"), _to_days(1)),
    (re.compile(r"(최근|지난)\s*(\d+)\s*주"), _to_days(7)),
    (re.compile(r"(최근|지난)\s*(\d+)\s*개월"), _to_days(30)),
    (
        re.compile(r"\blast\s+(\d+)\s*(day|days|week|weeks|month|months)\b", re.IGNORECASE),
        lambda match: (
            int(match.group(1))
            * {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30}[
                match.group(2).lower()
            ]
        ),
    ),
)

_LIMIT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(\d+)\s*개\b"),
    re.compile(r"\btop\s+(\d+)\b", re.IGNORECASE),
)


def _remove_span(text: str, start: int, end: int) -> str:
    collapsed = f"{text[:start]} {text[end:]}"
    return re.sub(r"\s+", " ", collapsed).strip()


def _extract_time(text: str) -> tuple[int | None, str]:
    best_match: re.Match[str] | None = None
    best_converter: _TimeConverter | None = None
    for pattern, converter in _TIME_PATTERNS:
        match = pattern.search(text)
        if match is not None and (best_match is None or match.start() < best_match.start()):
            best_match = match
            best_converter = converter
    if best_match is None or best_converter is None:
        return None, text
    return best_converter(best_match), _remove_span(text, best_match.start(), best_match.end())


def _extract_limit(text: str) -> tuple[int | None, str]:
    best_match: re.Match[str] | None = None
    for pattern in _LIMIT_PATTERNS:
        match = pattern.search(text)
        if match is not None and (best_match is None or match.start() < best_match.start()):
            best_match = match
    if best_match is None:
        return None, text
    return int(best_match.group(1)), _remove_span(text, best_match.start(), best_match.end())


def parse_query(raw: str) -> ParsedQuery:
    text = raw.strip()
    days, text_without_time = _extract_time(text)
    limit, text_without_filters = _extract_limit(text_without_time)
    search_text = re.sub(r"\s+", " ", text_without_filters).strip()
    return ParsedQuery(search_text=search_text, days=days, limit=limit, category=None)
