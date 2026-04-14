from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from radar.config_loader import load_category_config
from radar.models import Article
from radar.reporter import generate_report


def _category_name() -> str:
    configs = sorted(Path("config/categories").glob("*.yaml"))
    assert len(configs) == 1
    return configs[0].stem


def test_generate_report_includes_mcp_quality_panel(tmp_path: Path) -> None:
    category = load_category_config(_category_name())
    source = category.sources[0]
    now = datetime(2026, 4, 14, tzinfo=UTC)
    article = Article(
        title="MCP directory entry",
        link="https://github.com/example/mcp-server",
        summary="Read-only MCP lookup server",
        published=now,
        source=source.name,
        category=category.category_name,
        matched_entities={"MCPDomain": ["finance"], "RiskScope": ["read"]},
        collected_at=now,
    )
    quality_report = {
        "generated_at": now.isoformat(),
        "scope_note": "Community seed plus MCP transport evidence.",
        "summary": {
            "mcp_server_candidate_count": 2,
            "enabled_mcp_server_source_count": 1,
            "mcp_tool_result_events": 1,
            "repository_metadata_gap_count": 3,
            "daily_review_item_count": 1,
        },
        "daily_review_items": [
            {
                "reason": "mcp_candidate_disabled",
                "source": "candidate",
                "repository": "example/mcp-server",
            }
        ],
    }

    result = generate_report(
        category=category,
        articles=[article],
        output_path=tmp_path / f"{category.category_name}_report.html",
        stats={"sources": 1, "collected": 1, "matched": 1, "window_days": 1},
        quality_report=quality_report,
    )

    html = result.read_text(encoding="utf-8")
    assert 'data-visual-system="radar-unified-v2"' in html
    assert 'data-visual-surface="report"' in html
    assert 'id="mcp-quality"' in html
    assert "MCP Source Quality" in html
    assert "mcp_candidate_disabled" in html
