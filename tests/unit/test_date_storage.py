from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path


def test_snapshot_database_creates_file(tmp_path: Path) -> None:
    from radar.date_storage import snapshot_database

    db_file = tmp_path / "radar_data.duckdb"
    db_file.write_text("fake-db-content")

    result = snapshot_database(db_file)

    assert result is not None
    assert result.exists()
    today_iso = datetime.now(UTC).date().isoformat()
    assert result.name == f"{today_iso}.duckdb"
    assert result.parent == tmp_path / "daily"
    assert result.read_text() == "fake-db-content"


def test_snapshot_database_with_custom_date(tmp_path: Path) -> None:
    from radar.date_storage import snapshot_database

    db_file = tmp_path / "radar_data.duckdb"
    db_file.write_text("fake-db")
    custom_date = date(2026, 1, 15)

    result = snapshot_database(db_file, snapshot_date=custom_date)

    assert result is not None
    assert result.name == "2026-01-15.duckdb"


def test_snapshot_database_returns_none_for_missing_source(tmp_path: Path) -> None:
    from radar.date_storage import snapshot_database

    missing_db = tmp_path / "nonexistent.duckdb"
    result = snapshot_database(missing_db)
    assert result is None


def test_snapshot_database_custom_snapshot_root(tmp_path: Path) -> None:
    from radar.date_storage import snapshot_database

    db_file = tmp_path / "radar_data.duckdb"
    db_file.write_text("content")
    custom_root = tmp_path / "backups" / "snapshots"

    result = snapshot_database(db_file, snapshot_root=custom_root)

    assert result is not None
    assert result.parent == custom_root


def test_cleanup_date_directories_removes_old(tmp_path: Path) -> None:
    from radar.date_storage import cleanup_date_directories

    today = date(2026, 3, 13)
    # given: old dir (100 days ago) + recent dir (2 days ago)
    old_dir = tmp_path / "2025-12-03"
    old_dir.mkdir()
    (old_dir / "some_file.txt").write_text("old")

    recent_dir = tmp_path / "2026-03-11"
    recent_dir.mkdir()
    (recent_dir / "some_file.txt").write_text("recent")

    removed = cleanup_date_directories(tmp_path, keep_days=30, today=today)

    assert removed == 1
    assert not old_dir.exists()
    assert recent_dir.exists()


def test_cleanup_date_directories_keeps_recent(tmp_path: Path) -> None:
    from radar.date_storage import cleanup_date_directories

    today = date(2026, 3, 13)
    # given: all directories within 30 days
    for offset in range(5):
        d = today - timedelta(days=offset)
        (tmp_path / d.isoformat()).mkdir()

    removed = cleanup_date_directories(tmp_path, keep_days=30, today=today)

    assert removed == 0
    assert len(list(tmp_path.iterdir())) == 5


def test_cleanup_date_directories_ignores_non_date_dirs(tmp_path: Path) -> None:
    from radar.date_storage import cleanup_date_directories

    today = date(2026, 3, 13)
    (tmp_path / "not-a-date").mkdir()
    (tmp_path / "readme.txt").write_text("hi")

    removed = cleanup_date_directories(tmp_path, keep_days=7, today=today)

    assert removed == 0
    assert (tmp_path / "not-a-date").exists()


def test_cleanup_date_directories_missing_base_dir(tmp_path: Path) -> None:
    from radar.date_storage import cleanup_date_directories

    missing = tmp_path / "nonexistent"
    removed = cleanup_date_directories(missing, keep_days=7)
    assert removed == 0


def test_cleanup_dated_reports(tmp_path: Path) -> None:
    from radar.date_storage import cleanup_dated_reports

    today = date(2026, 3, 13)
    # given: old report, recent report, and non-matching file
    old_report = tmp_path / "tech_20251203.html"
    old_report.write_text("<html>old</html>")

    recent_report = tmp_path / "tech_20260311.html"
    recent_report.write_text("<html>recent</html>")

    other_file = tmp_path / "index.html"
    other_file.write_text("<html>index</html>")

    removed = cleanup_dated_reports(tmp_path, keep_days=30, today=today)

    assert removed == 1
    assert not old_report.exists()
    assert recent_report.exists()
    assert other_file.exists()


def test_cleanup_dated_reports_missing_dir(tmp_path: Path) -> None:
    from radar.date_storage import cleanup_dated_reports

    missing = tmp_path / "nonexistent"
    removed = cleanup_dated_reports(missing, keep_days=7)
    assert removed == 0


def test_storage_create_daily_snapshot(tmp_path: Path) -> None:
    from radar.storage import RadarStorage

    db_path = tmp_path / "data" / "radar_data.duckdb"
    storage = RadarStorage(db_path)
    try:
        result = storage.create_daily_snapshot()

        assert result is not None
        assert result.exists()
        today_iso = datetime.now(UTC).date().isoformat()
        assert result.name == f"{today_iso}.duckdb"
        assert result.parent == db_path.parent / "daily"
    finally:
        storage.close()


def test_storage_create_daily_snapshot_custom_dir(tmp_path: Path) -> None:
    from radar.storage import RadarStorage

    db_path = tmp_path / "data" / "radar_data.duckdb"
    custom_dir = str(tmp_path / "custom_snapshots")
    storage = RadarStorage(db_path)
    try:
        result = storage.create_daily_snapshot(snapshot_dir=custom_dir)

        assert result is not None
        assert result.parent == Path(custom_dir)
    finally:
        storage.close()


def test_storage_cleanup_old_snapshots(tmp_path: Path) -> None:
    from radar.storage import RadarStorage

    db_path = tmp_path / "data" / "radar_data.duckdb"
    storage = RadarStorage(db_path)

    # given: old snapshot directory beyond keep_days
    snapshot_root = db_path.parent / "daily"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    old_dir = snapshot_root / "2025-01-01"
    old_dir.mkdir()
    (old_dir / "data.txt").write_text("old")

    try:
        removed = storage.cleanup_old_snapshots(keep_days=30)
        assert removed == 1
        assert not old_dir.exists()
    finally:
        storage.close()
