from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from radar.models import Article
from radar.storage import RadarStorage


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "check_quality.py"
    spec = importlib.util.spec_from_file_location("mcp_check_quality_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generate_quality_artifacts_uses_latest_stored_checkpoint(
    tmp_path: Path,
    capsys,
) -> None:
    project_root = tmp_path
    (project_root / "config" / "categories").mkdir(parents=True)

    (project_root / "config" / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "database_path": "data/radar_data.duckdb",
                "report_dir": "reports",
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (project_root / "config" / "categories" / "test_mcp.yaml").write_text(
        yaml.safe_dump(
            {
                "category_name": "test_mcp",
                "display_name": "Test MCP",
                "sources": [
                    {
                        "id": "directory_seed",
                        "name": "awesome-mcp-directory",
                        "type": "github_readme_section",
                        "url": "https://github.com/example/awesome-mcp#finance",
                        "enabled": True,
                        "section": "Finance",
                        "content_type": "directory",
                        "trust_tier": "T4_community",
                        "config": {
                            "freshness_sla_days": 7,
                            "repository": "example/KIS_MCP_Server",
                            "source_section": "Finance",
                        },
                    }
                ],
                "entities": [],
                "data_quality": {
                    "quality_outputs": {
                        "tracked_event_models": [
                            "mcp_directory_entry",
                            "linked_repository_metadata",
                            "risk_scope_signal",
                        ]
                    }
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    article_time = datetime.now(UTC) - timedelta(days=30)
    db_path = project_root / "data" / "radar_data.duckdb"
    with RadarStorage(db_path) as storage:
        storage.upsert_articles(
            [
                Article(
                    title="KIS_MCP_Server",
                    link="https://github.com/example/KIS_MCP_Server",
                    summary="KIS REST API stock lookup and order MCP server",
                    published=article_time,
                    source="awesome-mcp-directory",
                    category="test_mcp",
                    matched_entities={
                        "MCPDomain": ["finance", "tax"],
                        "Provider": ["kis"],
                        "Capability": ["lookup", "order"],
                        "RiskScope": ["order", "api"],
                        "ProjectHealth": ["mcp", "server"],
                    },
                    collected_at=article_time,
                )
            ]
        )
        storage.conn.execute(
            "UPDATE articles SET collected_at = ? WHERE link = ?",
            [article_time.replace(tzinfo=None), "https://github.com/example/KIS_MCP_Server"],
        )

    module = _load_script_module()
    paths, report, articles = module.generate_quality_artifacts(project_root)

    assert Path(paths["latest"]).exists()
    assert Path(paths["dated"]).exists()
    assert len(articles) == 1
    assert report["summary"]["tracked_sources"] == 1
    assert report["summary"]["mcp_directory_entry_events"] == 1
    assert report["summary"]["mcp_signal_event_count"] == 2

    module.PROJECT_ROOT = project_root
    module.main()
    captured = capsys.readouterr()
    assert "quality_report=" in captured.out
    assert "tracked_sources=1" in captured.out
    assert "mcp_signal_event_count=2" in captured.out
