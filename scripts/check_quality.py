#!/usr/bin/env python3
"""Run DuckDB checks and write MCP radar quality JSON."""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import yaml
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from radar.common.quality_checks import (  # noqa: E402
    check_dates,
    check_duplicate_urls,
    check_missing_fields,
    check_text_lengths,
)
from radar.config_loader import (  # noqa: E402
    load_category_config,
    load_category_quality_config,
)
from radar.models import Article, CategoryConfig  # noqa: E402
from radar.quality_report import build_quality_report, write_quality_report  # noqa: E402
from radar.storage import RadarStorage  # noqa: E402


def _category_name(project_root: Path = PROJECT_ROOT) -> str:
    configs = sorted((project_root / "config" / "categories").glob("*.yaml"))
    if len(configs) != 1:
        raise RuntimeError(f"Expected exactly one category config in {project_root / 'config' / 'categories'}")
    return configs[0].stem


def _project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _load_runtime_config(project_root: Path) -> dict[str, Any]:
    raw = yaml.safe_load((project_root / "config" / "config.yaml").read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def _coerce_date(value: object) -> date | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(UTC).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                return date.fromisoformat(text[:10])
            except ValueError:
                return None
    return None


def _latest_article_date(db_path: Path, category_name: str) -> date | None:
    if not db_path.exists():
        return None
    try:
        with duckdb.connect(str(db_path), read_only=True) as con:
            row = con.execute(
                """
                SELECT MAX(COALESCE(published, collected_at))
                FROM articles
                WHERE category = ?
                """,
                [category_name],
            ).fetchone()
    except duckdb.Error:
        return None
    if not row:
        return None
    return _coerce_date(row[0])


def _lookback_days(target_date: date | None, *, minimum_days: int = 14) -> int:
    if target_date is None:
        return minimum_days
    age_days = (datetime.now(UTC).date() - target_date).days + 1
    return max(minimum_days, age_days)


def generate_quality_artifacts(
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Path], dict[str, Any], list[Article]]:
    category_name = _category_name(project_root)
    runtime_config = _load_runtime_config(project_root)
    db_path = _project_path(
        project_root,
        str(runtime_config.get("database_path", "data/radar_data.duckdb")),
    )
    report_dir = _project_path(
        project_root,
        str(runtime_config.get("report_dir", "reports")),
    )
    categories_dir = project_root / "config" / "categories"
    category = load_category_config(category_name, categories_dir=categories_dir)
    quality_config = load_category_quality_config(category_name, categories_dir=categories_dir)
    generated_at: datetime | None = None

    if db_path.exists():
        lookback_days = _lookback_days(_latest_article_date(db_path, category.category_name))
        with RadarStorage(db_path) as storage:
            articles = storage.recent_articles_by_collected_at(
                category.category_name,
                days=lookback_days,
                limit=500,
            )
    else:
        articles, generated_at = _articles_from_existing_report(category, report_dir)

    report = build_quality_report(
        category=category,
        articles=articles,
        errors=[],
        quality_config=quality_config,
        generated_at=generated_at,
    )
    paths = write_quality_report(
        report,
        output_dir=report_dir,
        category_name=category.category_name,
    )
    return paths, report, articles


def main() -> None:
    category_name = _category_name(PROJECT_ROOT)
    runtime_config = _load_runtime_config(PROJECT_ROOT)
    db_path = _project_path(
        PROJECT_ROOT,
        str(runtime_config.get("database_path", "data/radar_data.duckdb")),
    )

    if db_path.exists():
        with duckdb.connect(str(db_path), read_only=True) as con:
            total = con.execute("SELECT COUNT(*) FROM articles").fetchone()
            print(f"Total records: {total[0] if total else 0}")
            check_missing_fields(
                con,
                table_name="articles",
                null_conditions={
                    "title": "title IS NULL OR title = ''",
                    "link": "link IS NULL OR link = ''",
                    "summary": "summary IS NULL OR summary = ''",
                    "published": "published IS NULL",
                },
            )
            check_duplicate_urls(con, table_name="articles", url_column="link")
            check_text_lengths(con, table_name="articles", text_columns=["title", "summary"])
            check_dates(con, table_name="articles", date_column="published")
    else:
        print(f"Database not found: {db_path}")
        print("Using existing HTML report fallback for quality JSON.")

    paths, report, articles = generate_quality_artifacts(PROJECT_ROOT)
    summary = report["summary"]
    if isinstance(summary, dict):
        print(f"category={category_name}")
        print(f"scoped_articles={len(articles)}")
        print(f"quality_report={paths['latest']}")
        print(f"tracked_sources={summary.get('tracked_sources', 0)}")
        print(f"fresh_sources={summary.get('fresh_sources', 0)}")
        print(f"stale_sources={summary.get('stale_sources', 0)}")
        print(f"missing_sources={summary.get('missing_sources', 0)}")
        print(f"not_tracked_sources={summary.get('not_tracked_sources', 0)}")
        print(f"mcp_signal_event_count={summary.get('mcp_signal_event_count', 0)}")


def _articles_from_existing_report(
    category: CategoryConfig,
    report_dir: Path,
) -> tuple[list[Article], datetime | None]:
    report_path = _latest_report_path(category.category_name, report_dir)
    if report_path is None:
        print(f"Report not found in {report_dir}")
        sys.exit(1)

    html = report_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    generated_at = _report_generated_at(html) or _summary_generated_at(category.category_name, report_dir)
    published = generated_at or datetime.now(UTC)
    source_name = category.sources[0].name if category.sources else ""

    articles: list[Article] = []
    for article_tag in soup.find_all("article"):
        link_tag = article_tag.find("a", href=True)
        if link_tag is None:
            continue
        title = link_tag.get_text(" ", strip=True)
        link = str(link_tag.get("href", "")).strip()
        summary_tag = article_tag.find("p")
        summary = summary_tag.get_text(" ", strip=True) if summary_tag else ""
        entities = _entities_from_chips(article_tag.select(".chips span, .chip"))
        if not title or not link:
            continue
        articles.append(
            Article(
                title=title,
                link=link,
                summary=summary,
                published=published,
                source=source_name,
                category=category.category_name,
                matched_entities=entities,
                collected_at=published,
            )
        )
    return articles, generated_at


def _latest_report_path(category_name: str, report_dir: Path) -> Path | None:
    latest_path = report_dir / f"{category_name}_report.html"
    if latest_path.exists():
        return latest_path
    candidates = sorted(report_dir.glob(f"{category_name}_20*.html"), key=lambda path: path.name)
    return candidates[-1] if candidates else None


def _entities_from_chips(chips: list[object]) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {}
    for chip in chips:
        text = chip.get_text(" ", strip=True) if hasattr(chip, "get_text") else ""
        if ":" not in text:
            continue
        name, raw_values = text.split(":", 1)
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if name.strip() and values:
            entities[name.strip()] = values
    return entities


def _report_generated_at(html: str) -> datetime | None:
    match = re.search(r"Generated:\s*([0-9T:\-+\.]+)", html)
    if not match:
        return None
    return _parse_datetime(match.group(1))


def _summary_generated_at(category_name: str, report_dir: Path) -> datetime | None:
    summary_files = sorted(report_dir.glob(f"{category_name}_*_summary.json"), key=lambda path: path.name)
    if not summary_files:
        return None
    data = json.loads(summary_files[-1].read_text(encoding="utf-8"))
    raw_generated_at = data.get("generated_at") if isinstance(data, dict) else None
    return _parse_datetime(str(raw_generated_at or ""))


def _parse_datetime(value: str) -> datetime | None:
    raw = value.strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


if __name__ == "__main__":
    main()
