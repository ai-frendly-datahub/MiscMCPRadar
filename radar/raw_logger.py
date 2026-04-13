from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .models import Article


class RawLogger:
    def __init__(self, raw_dir: Path):
        self.raw_dir = raw_dir

    def log(
        self,
        articles: Iterable[Article],
        *,
        source_name: str,
        run_id: str | None = None,
    ) -> Path:
        now = datetime.now(UTC)
        date_dir = self.raw_dir / now.date().isoformat()
        safe_source_name = source_name.replace("/", "_").replace("\\", "_")
        filename = f"{safe_source_name}_{run_id}.jsonl" if run_id is not None else f"{safe_source_name}.jsonl"
        output_path = date_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_links: set[str] = set()
        if run_id is not None and output_path.exists():
            try:
                for line in output_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        record = json.loads(line)
                        existing_links.add(str(record.get("link", "")))
            except (OSError, json.JSONDecodeError):
                existing_links = set()

        with output_path.open("a", encoding="utf-8") as handle:
            for article in articles:
                if run_id is not None and article.link in existing_links:
                    continue
                record = {
                    "title": article.title,
                    "link": article.link,
                    "summary": article.summary,
                    "published": article.published.isoformat() if article.published else None,
                    "source": article.source,
                    "category": article.category,
                    "matched_entities": article.matched_entities,
                    "logged_at": now.isoformat(),
                }
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                if run_id is not None:
                    existing_links.add(article.link)
        return output_path
