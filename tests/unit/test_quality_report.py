from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from radar.config_loader import load_category_config, load_category_quality_config
from radar.models import Article
from radar.quality_report import build_quality_report, write_quality_report


def _category_name() -> str:
    configs = sorted(Path("config/categories").glob("*.yaml"))
    assert len(configs) == 1
    return configs[0].stem


def test_quality_report_tracks_directory_and_risk_scope_events() -> None:
    category = load_category_config(_category_name())
    source = category.sources[0]
    now = datetime(2026, 4, 13, tzinfo=UTC)
    article = Article(
        title="KIS_MCP_Server",
        link="https://github.com/example/KIS_MCP_Server",
        summary="KIS REST API stock lookup and order MCP server",
        published=now - timedelta(hours=3),
        source=source.name,
        category=category.category_name,
        matched_entities={
            "MCPDomain": ["finance", "tax"],
            "Provider": ["kis"],
            "Capability": ["lookup", "order"],
            "RiskScope": ["order", "api"],
            "ProjectHealth": ["mcp", "server"],
        },
        collected_at=now - timedelta(hours=3),
    )

    report = build_quality_report(
        category=category,
        articles=[article],
        quality_config=load_category_quality_config(_category_name()),
        generated_at=now,
    )

    summary = report["summary"]
    mcp_sources = [source for source in category.sources if source.type == "mcp_server"]
    assert summary["total_sources"] == len(category.sources)
    assert summary["tracked_sources"] >= 1
    assert summary["fresh_sources"] == 0
    assert summary["skipped_disabled_sources"] >= 1
    assert summary["mcp_directory_entry_events"] == 1
    assert summary["linked_repository_metadata_events"] == 0
    assert summary["risk_scope_signal_events"] == 1
    assert summary["mcp_signal_event_count"] >= 2
    assert summary["directory_seed_source_count"] >= 1
    assert summary["mcp_server_candidate_count"] == len(mcp_sources)
    assert summary["blocked_mcp_server_source_count"] == sum(
        1 for source in mcp_sources if not source.enabled
    )
    assert "daily_review_item_count" in summary

    source_row = report["sources"][0]
    assert source_row["event_model"] == "mcp_directory_entry"
    assert source_row["status"] == "skipped_disabled"
    assert source_row["trust_tier"] == "T4_community"
    assert source_row["freshness_sla_days"] == 7.0
    assert source_row["latest_required_field_proxy"] == {
        "repository": True,
        "source_section": True,
        "source_url": True,
    }
    assert isinstance(report["daily_review_items"], list)


def test_quality_report_skips_disabled_directory_source() -> None:
    category = load_category_config(_category_name())
    report = build_quality_report(
        category=category,
        articles=[],
        quality_config=load_category_quality_config(_category_name()),
        generated_at=datetime(2026, 4, 13, tzinfo=UTC),
    )

    summary = report["summary"]
    assert summary["tracked_sources"] >= 1
    assert summary["missing_sources"] == 0
    assert summary["skipped_disabled_sources"] >= 1
    assert summary["mcp_directory_entry_events"] == 0
    assert summary["daily_review_item_count"] == 0
    assert report["sources"][0]["status"] == "skipped_disabled"
    assert report["daily_review_items"] == []


def test_write_quality_report_writes_latest_and_dated_files(tmp_path: Path) -> None:
    category_name = "finance_tax_mcp"
    report = {
        "category": category_name,
        "generated_at": "2026-04-13T00:00:00+00:00",
        "summary": {},
    }

    paths = write_quality_report(report, output_dir=tmp_path, category_name=category_name)

    assert paths["latest"] == tmp_path / "finance_tax_mcp_quality.json"
    assert paths["dated"] == tmp_path / "finance_tax_mcp_20260413_quality.json"
    assert paths["latest"].exists()
    assert paths["dated"].exists()
