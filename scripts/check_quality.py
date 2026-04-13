#!/usr/bin/env python3
"""Run DuckDB data quality checks."""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

from radar.common.quality_checks import run_all_checks


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    db_path = PROJECT_ROOT / "data" / "radar_data.duckdb"
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    with duckdb.connect(str(db_path), read_only=True) as con:
        run_all_checks(
            con,
            table_name="articles",
            null_conditions={
                "title": "title IS NULL OR title = ''",
                "link": "link IS NULL OR link = ''",
                "summary": "summary IS NULL OR summary = ''",
                "published": "published IS NULL",
            },
            text_columns=["title", "summary"],
            url_column="link",
            date_column="published",
        )


if __name__ == "__main__":
    main()
