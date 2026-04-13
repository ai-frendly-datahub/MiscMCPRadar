from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast


class _ParsedQuery(Protocol):
    search_text: str
    days: int | None
    limit: int | None
    category: str | None


class _ParseQuery(Protocol):
    def __call__(self, raw: str) -> _ParsedQuery: ...


parse_query = cast(_ParseQuery, import_module("radar.nl_query").parse_query)


def test_parse_time_filter_days() -> None:
    parsed = parse_query("최근 3일 와인")

    assert parsed.days == 3
    assert "와인" in parsed.search_text


def test_parse_time_filter_weeks() -> None:
    parsed = parse_query("최근 1주 와인 뉴스")

    assert parsed.days == 7
    assert "와인" in parsed.search_text


def test_parse_time_filter_months() -> None:
    parsed = parse_query("지난 3개월 보르도")

    assert parsed.days == 90
    assert "보르도" in parsed.search_text


def test_parse_time_filter_english() -> None:
    parsed = parse_query("last 7 days wine news")

    assert parsed.days == 7
    assert "wine" in parsed.search_text


def test_parse_limit_korean() -> None:
    parsed = parse_query("와인 뉴스 10개")

    assert parsed.limit == 10
    assert "와인" in parsed.search_text


def test_parse_limit_english() -> None:
    parsed = parse_query("top 5 wines")

    assert parsed.limit == 5
    assert "wines" in parsed.search_text


def test_parse_combined_filters() -> None:
    parsed = parse_query("최근 2주 보르도 와인 5개")

    assert parsed.days == 14
    assert parsed.limit == 5
    assert parsed.search_text == "보르도 와인"


def test_parse_no_filters() -> None:
    parsed = parse_query("cabernet sauvignon")

    assert parsed.days is None
    assert parsed.limit is None
    assert parsed.search_text == "cabernet sauvignon"


def test_parse_category_always_none() -> None:
    parsed = parse_query("최근 1주 와인")

    assert parsed.category is None


def test_parse_empty_string() -> None:
    parsed = parse_query("")

    assert parsed.search_text == ""
    assert parsed.days is None
    assert parsed.limit is None
    assert parsed.category is None


def test_parse_whitespace_cleanup() -> None:
    parsed = parse_query("  최근 2주   보르도   와인   5개  ")

    assert parsed.days == 14
    assert parsed.limit == 5
    assert parsed.search_text == "보르도 와인"
