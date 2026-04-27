from __future__ import annotations

import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path


def snapshot_database(
    db_path: Path,
    *,
    snapshot_date: date | None = None,
    snapshot_root: Path | None = None,
) -> Path | None:
    if not db_path.exists():
        return None

    if snapshot_date is None:
        snapshot_date = datetime.now(UTC).date()

    if snapshot_root is None:
        snapshot_root = db_path.parent / "daily"

    snapshot_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_root / f"{snapshot_date.isoformat()}.duckdb"

    shutil.copy2(db_path, snapshot_path)
    return snapshot_path


def cleanup_date_directories(base_dir: Path, *, keep_days: int, today: date | None = None) -> int:
    if today is None:
        today = datetime.now(UTC).date()

    cutoff = today - timedelta(days=keep_days)
    removed = 0

    if not base_dir.exists():
        return 0

    for item in base_dir.iterdir():
        if not item.is_dir():
            continue

        try:
            stamp: date | None = None
            if len(item.name) == 10 and item.name.count("-") == 2:
                stamp = date.fromisoformat(item.name)
        except ValueError:
            continue

        if stamp and stamp < cutoff:
            shutil.rmtree(item)
            removed += 1

    return removed


def cleanup_dated_reports(report_dir: Path, *, keep_days: int, today: date | None = None) -> int:
    if today is None:
        today = datetime.now(UTC).date()

    cutoff = today - timedelta(days=keep_days)
    removed = 0

    if not report_dir.exists():
        return 0

    for item in report_dir.glob("*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].html"):
        try:
            date_str = item.stem.split("_")[-1]
            if len(date_str) == 8:
                stamp = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=UTC).date()
                if stamp < cutoff:
                    item.unlink()
                    removed += 1
        except (ValueError, IndexError):
            continue

    return removed


def apply_date_storage_policy(
    *,
    database_path: Path,
    raw_data_dir: Path,
    report_dir: Path,
    keep_raw_days: int,
    keep_report_days: int,
    snapshot_db: bool,
) -> dict[str, object]:
    snapshot_path = snapshot_database(database_path) if snapshot_db else None
    raw_removed = cleanup_date_directories(raw_data_dir, keep_days=keep_raw_days)
    report_removed = cleanup_dated_reports(report_dir, keep_days=keep_report_days)
    return {
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
        "raw_removed": raw_removed,
        "report_removed": report_removed,
    }
