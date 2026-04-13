from __future__ import annotations

import html
import os
import re
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import feedparser
import requests

from .exceptions import NetworkError, SourceError
from .models import Article, Source
from .resilience import get_circuit_breaker_manager


_DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (compatible; MCPRadarBot/1.0; +https://github.com/zzragida/ai-frendly-datahub)",
}
_SECTION_RE = re.compile(r"^###\s+(?:[^\w\s]+\s+)?(?P<section>[A-Za-z][^#]+?)\s*$")
_ITEM_RE = re.compile(r"^\*\*\[(?P<title>[^\]]+)\]\((?P<link>[^)]+)\)\*\*\s*[–-]\s*(?P<summary>.+)$")


class RateLimiter:
    def __init__(self, min_interval: float = 0.5):
        self._min_interval = min_interval
        self._last_request = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request = time.monotonic()


def _resolve_max_workers(max_workers: int | None = None) -> int:
    if max_workers is None:
        try:
            parsed = int(os.environ.get("RADAR_MAX_WORKERS", "5"))
        except ValueError:
            parsed = 5
    else:
        parsed = max_workers
    return max(1, min(parsed, 10))


def _fetch_url_with_retry(
    url: str,
    *,
    timeout: int,
    session: requests.Session | None = None,
    max_attempts: int = 3,
) -> requests.Response:
    last_error: requests.exceptions.RequestException | None = None
    for attempt in range(max_attempts):
        try:
            if session is not None:
                response = session.get(url, timeout=timeout, headers=_DEFAULT_HEADERS)
            else:
                response = requests.get(url, timeout=timeout, headers=_DEFAULT_HEADERS)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt == max_attempts - 1:
                raise
    raise RuntimeError(f"Retry loop exited unexpectedly: {last_error}")


def collect_sources(
    sources: list[Source],
    *,
    category: str,
    limit_per_source: int = 30,
    timeout: int = 15,
    min_interval_per_host: float = 0.5,
    max_workers: int | None = None,
) -> tuple[list[Article], list[str]]:
    if not sources:
        return [], []

    workers = _resolve_max_workers(max_workers)
    manager = get_circuit_breaker_manager()
    session = requests.Session()
    source_hosts = {
        source.name: (urlparse(source.url).netloc.lower() or source.name) for source in sources
    }
    rate_limiters = {
        host: RateLimiter(min_interval=min_interval_per_host) for host in set(source_hosts.values())
    }

    def _collect_for_source(source: Source) -> tuple[list[Article], list[str]]:
        rate_limiters[source_hosts[source.name]].acquire()
        try:
            breaker = manager.get_breaker(source.name)
            articles = breaker.call(
                _collect_single,
                source,
                category=category,
                limit=limit_per_source,
                timeout=timeout,
                session=session,
            )
            return articles, []
        except SourceError as exc:
            return [], [str(exc)]
        except NetworkError as exc:
            return [], [f"{source.name}: {exc}"]
        except Exception as exc:
            return [], [f"{source.name}: Unexpected error - {type(exc).__name__}: {exc}"]

    try:
        articles: list[Article] = []
        errors: list[str] = []

        if workers == 1:
            for source in sources:
                source_articles, source_errors = _collect_for_source(source)
                articles.extend(source_articles)
                errors.extend(source_errors)
            return articles, errors

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: list[Future[tuple[list[Article], list[str]]]] = [
                executor.submit(_collect_for_source, source) for source in sources
            ]
            for future in futures:
                source_articles, source_errors = future.result()
                articles.extend(source_articles)
                errors.extend(source_errors)
        return articles, errors
    finally:
        session.close()


def _collect_single(
    source: Source,
    *,
    category: str,
    limit: int,
    timeout: int,
    session: requests.Session | None = None,
) -> list[Article]:
    source_type = source.type.lower()
    if source_type == "rss":
        return _collect_rss(source, category=category, limit=limit, timeout=timeout, session=session)
    if source_type == "github_readme_section":
        return _collect_github_readme_section(
            source,
            category=category,
            limit=limit,
            timeout=timeout,
            session=session,
        )
    raise SourceError(source.name, f"Unsupported source type '{source.type}'")


def _collect_github_readme_section(
    source: Source,
    *,
    category: str,
    limit: int,
    timeout: int,
    session: requests.Session | None = None,
) -> list[Article]:
    try:
        response = _fetch_url_with_retry(source.url, timeout=timeout, session=session)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise NetworkError(f"Network error fetching {source.name}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise SourceError(source.name, f"Request failed: {exc}") from exc

    text = response.text
    section = source.section.strip()
    if not section:
        raise SourceError(source.name, "github_readme_section source requires a section")

    articles: list[Article] = []
    for item in parse_markdown_section_items(text, section)[:limit]:
        articles.append(
            Article(
                title=item["title"],
                link=item["link"],
                summary=item["summary"],
                published=datetime.now(UTC),
                source=source.name,
                category=category,
            )
        )
    return articles


def parse_markdown_section_items(markdown: str, section: str) -> list[dict[str, str]]:
    in_section = False
    items: list[dict[str, str]] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        section_match = _SECTION_RE.match(line)
        if section_match:
            heading = section_match.group("section").strip()
            if in_section and heading != section:
                break
            in_section = heading == section
            continue
        if not in_section:
            continue

        item_match = _ITEM_RE.match(line)
        if not item_match:
            continue
        title = html.unescape(item_match.group("title").strip())
        link = html.unescape(item_match.group("link").strip())
        summary = html.unescape(item_match.group("summary").strip())
        items.append({"title": title, "link": link, "summary": summary})
    return items


def _collect_rss(
    source: Source,
    *,
    category: str,
    limit: int,
    timeout: int,
    session: requests.Session | None = None,
) -> list[Article]:
    try:
        response = _fetch_url_with_retry(source.url, timeout=timeout, session=session)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        raise NetworkError(f"Network error fetching {source.name}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise SourceError(source.name, f"Request failed: {exc}") from exc

    try:
        feed = feedparser.parse(response.content)
        articles: list[Article] = []
        for entry in feed.entries[:limit]:
            title = html.unescape(_entry_text(entry, "title").strip())
            link = _entry_text(entry, "link").strip()
            if not title or not link:
                continue
            summary = _entry_text(entry, "summary") or _entry_text(entry, "description")
            articles.append(
                Article(
                    title=title,
                    link=link,
                    summary=html.unescape(summary.strip()),
                    published=_extract_datetime(entry),
                    source=source.name,
                    category=category,
                )
            )
        return articles
    except Exception as exc:
        raise SourceError(source.name, f"Failed to parse feed: {exc}") from exc


def _extract_datetime(entry: Mapping[str, object]) -> datetime | None:
    for key in ("published", "updated", "date"):
        raw = entry.get(key)
        if raw:
            try:
                parsed = parsedate_to_datetime(str(raw))
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=UTC)
                return parsed.astimezone(UTC)
            except Exception:
                continue
    return None


def _entry_text(entry: Mapping[str, object], key: str) -> str:
    value = entry.get(key)
    return value if isinstance(value, str) else ""
