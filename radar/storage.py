from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import duckdb

from .exceptions import StorageError
from .models import Article


def _utc_naive(dt: datetime | None) -> datetime | None:
    """Convert tz-aware datetime to UTC naive for DuckDB."""
    if dt is None:
        return None
    if dt.tzinfo:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


class RadarStorage:
    """DuckDB 기반 경량 스토리지."""

    def __init__(self, db_path: Path):
        self.db_path: Path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: duckdb.DuckDBPyConnection = duckdb.connect(str(self.db_path))
        self._ensure_tables()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> RadarStorage:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _ensure_tables(self) -> None:
        _ = self.conn.execute(
            """
            CREATE SEQUENCE IF NOT EXISTS articles_id_seq START 1;
            CREATE TABLE IF NOT EXISTS articles (
                id BIGINT PRIMARY KEY DEFAULT nextval('articles_id_seq'),
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                link TEXT NOT NULL UNIQUE,
                summary TEXT,
                published TIMESTAMP,
                collected_at TIMESTAMP NOT NULL,
                entities_json TEXT
            );
            """
        )
        _ = self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_articles_category_time ON articles (category, published, collected_at);"
        )

    def upsert_articles(self, articles: Iterable[Article]) -> None:
        """중복 링크는 덮어쓰고 최신 수집 시각을 기록."""
        now = _utc_naive(datetime.now(UTC))
        rows: list[tuple[object, ...]] = []
        for article in articles:
            rows.append(
                (
                    article.category,
                    article.source,
                    article.title,
                    article.link,
                    article.summary,
                    _utc_naive(article.published),
                    now,
                    json.dumps(article.matched_entities, ensure_ascii=False),
                )
            )

        if not rows:
            return

        try:
            _ = self.conn.begin()
            _ = self.conn.executemany(
                """
                INSERT INTO articles (category, source, title, link, summary, published, collected_at, entities_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(link) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    published = EXCLUDED.published,
                    collected_at = EXCLUDED.collected_at,
                    entities_json = EXCLUDED.entities_json
                """,
                rows,
            )
            _ = self.conn.commit()
        except Exception as exc:
            try:
                _ = self.conn.rollback()
            except duckdb.Error:
                pass
            raise StorageError("Failed to upsert articles") from exc

    def recent_articles(self, category: str, *, days: int = 7, limit: int = 200) -> list[Article]:
        """최근 N일 내 기사 반환."""
        since = _utc_naive(datetime.now(UTC) - timedelta(days=days))
        cur = self.conn.execute(
            """
            SELECT category, source, title, link, summary, published, collected_at, entities_json
            FROM articles
            WHERE category = ? AND COALESCE(published, collected_at) >= ?
            ORDER BY COALESCE(published, collected_at) DESC
            LIMIT ?
            """,
            [category, since, limit],
        )
        rows = cast(
            list[
                tuple[str, str, str, str, str | None, datetime | None, datetime | None, str | None]
            ],
            cur.fetchall(),
        )

        results: list[Article] = []
        for row in rows:
            category_value, source, title, link, summary, published, collected_at, raw_entities = (
                row
            )
            published_at = published if isinstance(published, datetime) else None
            collected = collected_at if isinstance(collected_at, datetime) else None

            entities: dict[str, list[str]] = {}
            if raw_entities:
                try:
                    parsed_entities = cast(object, json.loads(raw_entities))
                    if isinstance(parsed_entities, dict):
                        parsed_map = cast(dict[object, object], parsed_entities)
                        entities = {}
                        for name, keywords in parsed_map.items():
                            if not isinstance(name, str) or not isinstance(keywords, list):
                                continue
                            normalized_keywords: list[str] = []
                            for keyword in cast(list[object], keywords):
                                normalized_keywords.append(str(keyword))
                            entities[name] = normalized_keywords
                except json.JSONDecodeError:
                    entities = {}

            results.append(
                Article(
                    title=str(title),
                    link=str(link),
                    summary=str(summary) if summary is not None else "",
                    published=published_at,
                    source=str(source),
                    category=str(category_value),
                    matched_entities=entities,
                    collected_at=collected,
                )
            )
        return results

    def delete_older_than(self, days: int) -> int:
        """보존 기간 밖 데이터 삭제."""
        cutoff = _utc_naive(datetime.now(UTC) - timedelta(days=days))
        count_row = self.conn.execute(
            "SELECT COUNT(*) FROM articles WHERE COALESCE(published, collected_at) < ?", [cutoff]
        ).fetchone()
        to_delete = count_row[0] if count_row else 0
        _ = self.conn.execute(
            "DELETE FROM articles WHERE COALESCE(published, collected_at) < ?", [cutoff]
        )
        return to_delete

    def create_daily_snapshot(self, snapshot_dir: str | None = None) -> Path | None:
        from .date_storage import snapshot_database

        snapshot_root = Path(snapshot_dir) if snapshot_dir else self.db_path.parent / "daily"
        return snapshot_database(self.db_path, snapshot_root=snapshot_root)

    def cleanup_old_snapshots(self, snapshot_dir: str | None = None, keep_days: int = 90) -> int:
        from .date_storage import cleanup_date_directories

        snapshot_root = Path(snapshot_dir) if snapshot_dir else self.db_path.parent / "daily"
        return cleanup_date_directories(snapshot_root, keep_days=keep_days)
