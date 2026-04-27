"""DuckDB 데이터 품질 점검 유틸리티.

테이블명과 필드 조건을 매개변수로 받아 범용적으로 사용 가능합니다.
"""

from __future__ import annotations

from typing import cast

import duckdb


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _fetchone_required(
    con: duckdb.DuckDBPyConnection,
    query: str,
    params: list[object] | None = None,
) -> tuple[object, ...]:
    row = cast(tuple[object, ...] | None, con.execute(query, params or []).fetchone())
    if row is None:
        raise RuntimeError("DuckDB query returned no rows")
    return row


def _to_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str, bytes, bytearray)):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value).__name__}")


def _to_optional_int(value: object) -> int | None:
    return None if value is None else _to_int(value)


def _to_optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float, str, bytes, bytearray)):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value).__name__}")


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===\n")


def _column_exists(
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    column_name: str,
) -> bool:
    rows = cast(
        list[tuple[object]],
        con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchall(),
    )
    return any(str(row[0]) == column_name for row in rows)


def check_missing_fields(
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    null_conditions: dict[str, str],
) -> None:
    _print_section("Missing Field Check")
    total = _to_int(
        _fetchone_required(con, f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}")[0]
    )

    if total == 0:
        print("No records found.")
        return

    for field, condition in null_conditions.items():
        count = _to_int(
            _fetchone_required(
                con,
                f"SELECT COUNT(*) FROM {_quote_identifier(table_name)} WHERE {condition}",
            )[0]
        )
        ratio = (count / total) * 100
        print(f"  {field}: {count} / {total} ({ratio:.1f}%)")


def check_duplicate_urls(
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    url_column: str = "url",
    limit: int = 10,
) -> None:
    _print_section("Duplicate URL Check")

    raw_rows = cast(
        list[tuple[object, object]],
        con.execute(
            f"""
        SELECT {_quote_identifier(url_column)} AS url_value, COUNT(*) AS cnt
        FROM {_quote_identifier(table_name)}
        GROUP BY {_quote_identifier(url_column)}
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC, url_value
        LIMIT ?
        """,
            [limit],
        ).fetchall(),
    )
    rows: list[tuple[str | None, int]] = [
        (None if row[0] is None else str(row[0]), _to_int(row[1])) for row in raw_rows
    ]

    if not rows:
        print("No duplicate URLs found.")
        return

    for url_value, cnt in rows:
        print(f"  {cnt}x: {url_value}")


def check_text_lengths(
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    text_columns: list[str],
) -> None:
    _print_section("Text Length Statistics")

    if not text_columns:
        print("No text columns provided.")
        return

    for column in text_columns:
        avg_len_raw, min_len_raw, max_len_raw = _fetchone_required(
            con,
            f"""
            SELECT
                AVG(LENGTH({_quote_identifier(column)})) AS avg_len,
                MIN(LENGTH({_quote_identifier(column)})) AS min_len,
                MAX(LENGTH({_quote_identifier(column)})) AS max_len
            FROM {_quote_identifier(table_name)}
            """,
        )

        avg_len = _to_optional_float(avg_len_raw)
        min_len = _to_optional_int(min_len_raw)
        max_len = _to_optional_int(max_len_raw)

        avg_text = "N/A" if avg_len is None else f"{avg_len:.1f}"
        print(f"  {column}: avg/min/max = {avg_text} / {min_len} / {max_len}")


def check_language_values(
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    language_column: str = "language",
    allowed_languages: set[str] | None = None,
) -> None:
    _print_section("Language Value Check")

    raw_rows = cast(
        list[tuple[object, object]],
        con.execute(
            f"""
        SELECT {_quote_identifier(language_column)} AS language_value, COUNT(*) AS cnt
        FROM {_quote_identifier(table_name)}
        GROUP BY {_quote_identifier(language_column)}
        ORDER BY cnt DESC, language_value
        """
        ).fetchall(),
    )
    rows: list[tuple[str | None, int]] = [
        (None if row[0] is None else str(row[0]), _to_int(row[1])) for row in raw_rows
    ]

    if not rows:
        print("No language values found.")
        return

    print("Distribution:")
    for language_value, cnt in rows:
        print(f"  {language_value}: {cnt}")

    if allowed_languages is None:
        return

    invalid = [
        (language_value, cnt)
        for language_value, cnt in rows
        if language_value is not None and language_value not in allowed_languages
    ]

    if invalid:
        print("Invalid language values:")
        for language_value, cnt in invalid:
            print(f"  {language_value}: {cnt}")
    else:
        print("All language values are allowed.")


def check_dates(
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    date_column: str = "published_at",
) -> None:
    _print_section("Date Check")

    future_count = _to_int(
        _fetchone_required(
            con,
            f"""
            SELECT COUNT(*)
            FROM {_quote_identifier(table_name)}
            WHERE {_quote_identifier(date_column)} > CURRENT_TIMESTAMP
            """,
        )[0]
    )

    oldest, newest = _fetchone_required(
        con,
        f"""
        SELECT
            MIN({_quote_identifier(date_column)}) AS oldest,
            MAX({_quote_identifier(date_column)}) AS newest
        FROM {_quote_identifier(table_name)}
        """,
    )

    print(f"  oldest: {oldest}")
    print(f"  newest: {newest}")
    print(f"  future dates: {future_count}")


def run_all_checks(
    con: duckdb.DuckDBPyConnection,
    *,
    table_name: str,
    null_conditions: dict[str, str],
    text_columns: list[str] | None = None,
    language_column: str = "language",
    allowed_languages: set[str] | None = None,
    url_column: str = "url",
    date_column: str = "published_at",
) -> None:
    total = _to_int(
        _fetchone_required(con, f"SELECT COUNT(*) FROM {_quote_identifier(table_name)}")[0]
    )
    print(f"Total records: {total}")

    check_missing_fields(con, table_name=table_name, null_conditions=null_conditions)
    check_duplicate_urls(con, table_name=table_name, url_column=url_column)
    check_text_lengths(con, table_name=table_name, text_columns=text_columns or [])
    if _column_exists(con, table_name=table_name, column_name=language_column):
        check_language_values(
            con,
            table_name=table_name,
            language_column=language_column,
            allowed_languages=allowed_languages,
        )
    else:
        _print_section("Language Value Check")
        print(f"Skipped: column '{language_column}' not found.")
    check_dates(con, table_name=table_name, date_column=date_column)
