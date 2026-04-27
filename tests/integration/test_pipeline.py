from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from main import run


class _FakeResponse:
    status_code: int
    content: bytes

    def __init__(self, content: bytes) -> None:
        self.status_code = 200
        self.content = content

    def raise_for_status(self) -> None:
        return None


def test_full_pipeline_creates_all_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    categories_dir = tmp_path / "categories"
    categories_dir.mkdir(parents=True, exist_ok=True)

    db_path = tmp_path / "data" / "radar_data.duckdb"
    report_dir = tmp_path / "reports"
    raw_dir = tmp_path / "data" / "raw"
    search_db_path = tmp_path / "data" / "search_index.db"

    _ = config_path.write_text(
        yaml.safe_dump(
            {
                "database_path": str(db_path),
                "report_dir": str(report_dir),
                "raw_data_dir": str(raw_dir),
                "search_db_path": str(search_db_path),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    category_file = categories_dir / "test_cat.yaml"
    _ = category_file.write_text(
        yaml.safe_dump(
            {
                "category_name": "test_cat",
                "display_name": "Test Category",
                "sources": [
                    {
                        "name": "Mock RSS",
                        "type": "rss",
                        "url": "https://example.com/feed.xml",
                    }
                ],
                "entities": [
                    {
                        "name": "Bean",
                        "display_name": "Bean",
                        "keywords": ["arabica"],
                    }
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    rss_payload = b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<rss version=\"2.0\"><channel><title>Mock</title>
<item>
  <title>Arabica market update</title>
  <link>https://example.com/article-1</link>
  <description>arabica demand is up</description>
  <pubDate>Wed, 04 Mar 2026 10:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

    with patch("radar.collector.requests.Session.get", return_value=_FakeResponse(rss_payload)):
        output_path = run(
            category="test_cat",
            config_path=config_path,
            categories_dir=categories_dir,
            per_source_limit=5,
            recent_days=7,
            timeout=5,
            keep_days=30,
        )

    assert db_path.exists()
    assert raw_dir.exists()
    assert list(raw_dir.rglob("*.jsonl"))
    assert search_db_path.exists()
    assert output_path.exists()
    assert output_path.suffix == ".html"
