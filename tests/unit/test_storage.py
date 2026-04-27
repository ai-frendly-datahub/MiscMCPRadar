from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import pytest


StorageError = cast(type[Exception], import_module("radar.exceptions").StorageError)


class _Article(Protocol):
    title: str
    link: str
    summary: str
    published: datetime | None
    source: str
    category: str
    matched_entities: dict[str, list[str]]
    collected_at: datetime | None


class _ArticleCtor(Protocol):
    def __call__(
        self,
        *,
        title: str,
        link: str,
        summary: str,
        published: datetime | None,
        source: str,
        category: str,
        matched_entities: dict[str, list[str]] = ...,
        collected_at: datetime | None = ...,
    ) -> _Article: ...


class _RadarStorage(Protocol):
    def upsert_articles(self, articles: Iterable[_Article]) -> None: ...

    def recent_articles(
        self, category: str, *, days: int = 7, limit: int = 200
    ) -> list[_Article]: ...

    def delete_older_than(self, days: int) -> int: ...

    def close(self) -> None: ...


class _RadarStorageCtor(Protocol):
    def __call__(self, db_path: Path) -> _RadarStorage: ...


Article = cast(_ArticleCtor, import_module("radar.models").Article)
RadarStorage = cast(_RadarStorageCtor, import_module("radar.storage").RadarStorage)


def _make_article(
    *,
    title: str,
    link: str,
    summary: str,
    published: datetime | None,
    source: str = "Example RSS",
    category: str = "tech",
    matched_entities: dict[str, list[str]] | None = None,
) -> _Article:
    return Article(
        title=title,
        link=link,
        summary=summary,
        published=published,
        source=source,
        category=category,
        matched_entities=matched_entities or {},
    )


def test_upsert_articles_inserts_new_article(tmp_duckdb: Path, sample_article: object) -> None:
    storage = RadarStorage(tmp_duckdb)
    article = cast(_Article, sample_article)

    try:
        storage.upsert_articles([article])
        results = storage.recent_articles(category="tech", days=30)
    finally:
        storage.close()

    assert len(results) == 1
    assert results[0].link == article.link
    assert results[0].title == article.title
    assert results[0].matched_entities == article.matched_entities


def test_upsert_articles_updates_duplicate_link(tmp_duckdb: Path) -> None:
    storage = RadarStorage(tmp_duckdb)
    link = "https://example.com/dup"
    first = _make_article(
        title="First title",
        link=link,
        summary="first version",
        published=datetime.now(UTC),
    )
    second = _make_article(
        title="Updated title",
        link=link,
        summary="second version",
        published=datetime.now(UTC),
    )

    try:
        storage.upsert_articles([first])
        storage.upsert_articles([second])
        results = storage.recent_articles(category="tech", days=30)
    finally:
        storage.close()

    assert len(results) == 1
    assert results[0].title == "Updated title"
    assert results[0].summary == "second version"


def test_upsert_atomicity_rollback_preserves_data(tmp_duckdb: Path) -> None:
    storage = RadarStorage(tmp_duckdb)
    existing = _make_article(
        title="Existing",
        link="https://example.com/existing",
        summary="stable",
        published=datetime.now(UTC),
    )
    valid = _make_article(
        title="Valid",
        link="https://example.com/valid",
        summary="should rollback",
        published=datetime.now(UTC),
    )
    invalid = _make_article(
        title="Invalid",
        link="https://example.com/invalid",
        summary="should fail",
        published=datetime.now(UTC),
    )
    invalid.link = None

    try:
        storage.upsert_articles([existing])

        with pytest.raises(StorageError):
            storage.upsert_articles([valid, invalid])

        results = storage.recent_articles(category="tech", days=30)
    finally:
        storage.close()

    assert len(results) == 1
    assert results[0].link == existing.link
    assert results[0].title == existing.title


def test_batch_upsert_100_articles(tmp_duckdb: Path) -> None:
    storage = RadarStorage(tmp_duckdb)
    articles = [
        _make_article(
            title=f"Article {idx}",
            link=f"https://example.com/batch-{idx}",
            summary=f"summary {idx}",
            published=datetime.now(UTC),
        )
        for idx in range(100)
    ]

    try:
        storage.upsert_articles(articles)
        results = storage.recent_articles(category="tech", days=30, limit=200)
    finally:
        storage.close()

    assert len(results) == 100
    assert {article.link for article in results} == {article.link for article in articles}


def test_upsert_on_conflict_updates_existing(tmp_duckdb: Path) -> None:
    storage = RadarStorage(tmp_duckdb)
    link = "https://example.com/on-conflict"
    first = _make_article(
        title="Original title",
        link=link,
        summary="original",
        published=datetime.now(UTC),
    )
    updated = _make_article(
        title="Updated by conflict",
        link=link,
        summary="updated",
        published=datetime.now(UTC),
    )

    try:
        storage.upsert_articles([first])
        storage.upsert_articles([updated])
        results = storage.recent_articles(category="tech", days=30)
    finally:
        storage.close()

    assert len(results) == 1
    assert results[0].title == "Updated by conflict"
    assert results[0].summary == "updated"


def test_upsert_articles_accepts_empty_iterable(tmp_storage: object) -> None:
    storage = cast(_RadarStorage, tmp_storage)

    storage.upsert_articles([])
    results = storage.recent_articles(category="tech", days=30)

    assert results == []


def test_recent_articles_filters_by_period(tmp_storage: object) -> None:
    storage = cast(_RadarStorage, tmp_storage)
    recent_article = _make_article(
        title="Recent",
        link="https://example.com/recent",
        summary="inside window",
        published=datetime.now(UTC) - timedelta(days=1),
    )
    old_article = _make_article(
        title="Old",
        link="https://example.com/old",
        summary="outside window",
        published=datetime.now(UTC) - timedelta(days=20),
    )

    storage.upsert_articles([recent_article, old_article])
    results = storage.recent_articles(category="tech", days=7)

    assert len(results) == 1
    assert results[0].link == recent_article.link


def test_recent_articles_filters_by_category(tmp_storage: object) -> None:
    storage = cast(_RadarStorage, tmp_storage)
    tech_article = _make_article(
        title="Tech",
        link="https://example.com/tech",
        summary="tech",
        published=datetime.now(UTC),
        category="tech",
    )
    policy_article = _make_article(
        title="Policy",
        link="https://example.com/policy",
        summary="policy",
        published=datetime.now(UTC),
        category="policy",
    )

    storage.upsert_articles([tech_article, policy_article])
    tech_results = storage.recent_articles(category="tech", days=30)
    policy_results = storage.recent_articles(category="policy", days=30)

    assert len(tech_results) == 1
    assert len(policy_results) == 1
    assert tech_results[0].category == "tech"
    assert policy_results[0].category == "policy"


def test_delete_older_than_preserves_recent_articles(tmp_storage: object) -> None:
    storage = cast(_RadarStorage, tmp_storage)
    recent_article = _make_article(
        title="Recent",
        link="https://example.com/recent-keep",
        summary="should remain",
        published=datetime.now(UTC) - timedelta(days=2),
    )

    storage.upsert_articles([recent_article])
    deleted = storage.delete_older_than(days=7)
    results = storage.recent_articles(category="tech", days=30)

    assert deleted == 0
    assert len(results) == 1
    assert results[0].link == recent_article.link


def test_delete_older_than_removes_old_articles(tmp_storage: object) -> None:
    storage = cast(_RadarStorage, tmp_storage)
    old_article = _make_article(
        title="Old",
        link="https://example.com/old-delete",
        summary="should be deleted",
        published=datetime.now(UTC) - timedelta(days=40),
    )

    storage.upsert_articles([old_article])
    deleted = storage.delete_older_than(days=7)
    results = storage.recent_articles(category="tech", days=365)

    assert deleted == 1
    assert results == []


def test_storage_close_then_reuse_raises_error(tmp_duckdb: Path) -> None:
    storage = RadarStorage(tmp_duckdb)
    storage.close()

    with pytest.raises(StorageError):
        storage.upsert_articles(
            [
                _make_article(
                    title="After close",
                    link="https://example.com/closed",
                    summary="cannot write",
                    published=datetime.now(UTC),
                )
            ]
        )
