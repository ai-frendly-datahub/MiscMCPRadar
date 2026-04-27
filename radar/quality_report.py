from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import Article, CategoryConfig, Source


TRACKED_EVENT_MODEL_ORDER = [
    "mcp_directory_entry",
    "mcp_tool_result",
    "linked_repository_metadata",
    "risk_scope_signal",
]
TRACKED_EVENT_MODELS = set(TRACKED_EVENT_MODEL_ORDER)


def build_quality_report(
    *,
    category: CategoryConfig,
    articles: Iterable[Article],
    errors: Iterable[str] | None = None,
    quality_config: Mapping[str, object] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated = _as_utc(generated_at or datetime.now(UTC))
    articles_list = list(articles)
    errors_list = [str(error) for error in (errors or [])]
    quality = _dict(quality_config or {}, "data_quality")
    freshness_sla = _dict(quality, "freshness_sla")
    tracked_event_models = _tracked_event_models(quality)

    events = _build_event_rows(
        articles=articles_list,
        sources=category.sources,
        tracked_event_models=tracked_event_models,
    )
    source_rows = [
        _build_source_row(
            source=source,
            articles=articles_list,
            event_rows=events,
            errors=errors_list,
            freshness_sla=freshness_sla,
            tracked_event_models=tracked_event_models,
            generated_at=generated,
        )
        for source in category.sources
    ]
    daily_review_items = _build_daily_review_items(source_rows, category.sources)

    status_counts = Counter(str(row["status"]) for row in source_rows)
    event_counts = Counter(str(row["event_model"]) for row in events)
    mcp_server_sources = [source for source in category.sources if _is_mcp_server_source(source)]
    summary = {
        "total_sources": len(source_rows),
        "enabled_sources": sum(1 for row in source_rows if row["enabled"]),
        "tracked_sources": sum(1 for row in source_rows if row["tracked"]),
        "fresh_sources": status_counts.get("fresh", 0),
        "stale_sources": status_counts.get("stale", 0),
        "missing_sources": status_counts.get("missing", 0),
        "missing_event_sources": status_counts.get("missing_event", 0),
        "unknown_event_date_sources": status_counts.get("unknown_event_date", 0),
        "not_tracked_sources": status_counts.get("not_tracked", 0),
        "skipped_disabled_sources": status_counts.get("skipped_disabled", 0),
        "collection_error_count": len(errors_list),
        "mcp_signal_event_count": len(events),
        "directory_seed_source_count": sum(
            1 for source in category.sources if _source_event_model(source) == "mcp_directory_entry"
        ),
        "mcp_server_candidate_count": len(mcp_server_sources),
        "enabled_mcp_server_source_count": sum(1 for source in mcp_server_sources if source.enabled),
        "blocked_mcp_server_source_count": sum(1 for source in mcp_server_sources if not source.enabled),
        "real_transport_smoke_tested_source_count": sum(
            1 for source in mcp_server_sources if bool(source.config.get("real_transport_smoke_tested_at"))
        ),
        "tool_allowlist_present_source_count": sum(
            1 for source in mcp_server_sources if bool(source.config.get("tools"))
        ),
        "credential_required_source_count": sum(
            1 for source in mcp_server_sources if bool(source.config.get("env"))
        ),
        "repository_metadata_complete_source_count": sum(
            1 for source in mcp_server_sources if _repository_metadata_complete(source)
        ),
        "repository_metadata_gap_count": sum(
            len(_repository_metadata_gaps(source)) for source in mcp_server_sources
        ),
        "security_activation_gate_count": sum(
            len(_list(source.config.get("activation_gates"))) for source in mcp_server_sources
        ),
        "daily_review_item_count": len(daily_review_items),
    }
    for event_model in TRACKED_EVENT_MODEL_ORDER:
        summary[f"{event_model}_events"] = event_counts.get(event_model, 0)

    return {
        "category": category.category_name,
        "generated_at": generated.isoformat(),
        "scope_note": (
            f"{category.category_name} treats the awesome-mcp-korea section as a T4 community "
            "directory seed. Repository metadata, API credential scope, and security "
            "context remain separate enrichment layers until archived, license, and "
            "advisory fields are collected."
        ),
        "summary": summary,
        "sources": source_rows,
        "events": events,
        "daily_review_items": daily_review_items,
        "source_backlog": (quality_config or {}).get("source_backlog", {}),
        "errors": errors_list,
    }


def write_quality_report(
    report: Mapping[str, object],
    *,
    output_dir: Path,
    category_name: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = _parse_datetime(str(report.get("generated_at") or "")) or datetime.now(UTC)
    date_stamp = _as_utc(generated_at).strftime("%Y%m%d")
    latest_path = output_dir / f"{category_name}_quality.json"
    dated_path = output_dir / f"{category_name}_{date_stamp}_quality.json"
    encoded = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    latest_path.write_text(encoded + "\n", encoding="utf-8")
    dated_path.write_text(encoded + "\n", encoding="utf-8")
    return {"latest": latest_path, "dated": dated_path}


def _build_event_rows(
    *,
    articles: list[Article],
    sources: list[Source],
    tracked_event_models: set[str],
) -> list[dict[str, Any]]:
    source_map = {source.name: source for source in sources}
    rows: list[dict[str, Any]] = []
    for article in articles:
        source = source_map.get(article.source)
        if source is None:
            continue
        for event_model in _article_event_models(article, source, tracked_event_models):
            event_at = _event_datetime(article)
            rows.append(
                {
                    "source": source.name,
                    "event_model": event_model,
                    "title": article.title,
                    "url": article.link,
                    "event_at": event_at.isoformat() if event_at else None,
                    "mcp_domain": _matches(article, "MCPDomain"),
                    "provider": _matches(article, "Provider"),
                    "capability": _matches(article, "Capability"),
                    "risk_scope": _matches(article, "RiskScope"),
                    "project_health": _matches(article, "ProjectHealth"),
                    "required_field_proxy": _required_field_proxy(article, source, event_model),
                }
            )
    return rows


def _build_source_row(
    *,
    source: Source,
    articles: list[Article],
    event_rows: list[dict[str, Any]],
    errors: list[str],
    freshness_sla: Mapping[str, object],
    tracked_event_models: set[str],
    generated_at: datetime,
) -> dict[str, Any]:
    source_articles = [article for article in articles if article.source == source.name]
    source_errors = [error for error in errors if error.startswith(f"{source.name}:")]
    event_model = _source_event_model(source)
    source_event_rows = [
        row
        for row in event_rows
        if row["source"] == source.name and row["event_model"] == event_model
    ]
    latest_event = _latest_event(source_event_rows)
    latest_event_at = (
        _parse_datetime(str(latest_event.get("event_at") or "")) if latest_event else None
    )
    sla_days = _source_sla_days(source, event_model, freshness_sla)
    age_days = _age_days(generated_at, latest_event_at) if latest_event_at else None
    status = _source_status(
        source=source,
        event_model=event_model,
        tracked_event_models=tracked_event_models,
        article_count=len(source_articles),
        event_count=len(source_event_rows),
        latest_event_at=latest_event_at,
        sla_days=sla_days,
        age_days=age_days,
    )

    return {
        "source": source.name,
        "source_type": source.type,
        "enabled": source.enabled,
        "trust_tier": source.trust_tier,
        "content_type": source.content_type,
        "collection_tier": source.collection_tier,
        "producer_role": source.producer_role,
        "info_purpose": source.info_purpose,
        "tracked": event_model in tracked_event_models,
        "event_model": event_model,
        "repository": _source_repository(source),
        "activation_status": str(source.config.get("activation_status", "")),
        "activation_gates": _list(source.config.get("activation_gates")),
        "tools_count": len(_list(source.config.get("tools"))),
        "env_count": len(_list(source.config.get("env"))),
        "repository_metadata_gaps": _repository_metadata_gaps(source),
        "freshness_sla_days": sla_days,
        "status": status,
        "article_count": len(source_articles),
        "event_count": len(source_event_rows),
        "latest_event_at": latest_event_at.isoformat() if latest_event_at else None,
        "age_days": round(age_days, 2) if age_days is not None else None,
        "latest_title": str(latest_event.get("title", "")) if latest_event else "",
        "latest_url": str(latest_event.get("url", "")) if latest_event else "",
        "latest_mcp_domain": latest_event.get("mcp_domain", []) if latest_event else [],
        "latest_provider": latest_event.get("provider", []) if latest_event else [],
        "latest_capability": latest_event.get("capability", []) if latest_event else [],
        "latest_risk_scope": latest_event.get("risk_scope", []) if latest_event else [],
        "latest_required_field_proxy": (
            latest_event.get("required_field_proxy", {}) if latest_event else {}
        ),
        "errors": source_errors,
    }


def _build_daily_review_items(
    source_rows: list[dict[str, Any]],
    sources: list[Source],
) -> list[dict[str, Any]]:
    source_map = {source.name: source for source in sources}
    items: list[dict[str, Any]] = []
    for row in source_rows:
        source_name = str(row.get("source", ""))
        source = source_map.get(source_name)
        status = str(row.get("status", ""))
        event_model = str(row.get("event_model", ""))
        if status in {"missing", "missing_event", "stale", "unknown_event_date"}:
            items.append(
                {
                    "reason": f"source_status_{status}",
                    "source": source_name,
                    "event_model": event_model,
                    "detail": "Tracked MCP evidence source needs collection or freshness follow-up.",
                }
            )
        if source is None or not _is_mcp_server_source(source):
            continue
        repository = _source_repository(source)
        activation_status = str(source.config.get("activation_status", ""))
        if not source.enabled:
            items.append(
                {
                    "reason": "mcp_candidate_disabled",
                    "source": source_name,
                    "repository": repository,
                    "activation_status": activation_status,
                    "activation_gates": _list(source.config.get("activation_gates")),
                }
            )
        elif event_model == "mcp_tool_result" and int(row.get("event_count") or 0) == 0:
            items.append(
                {
                    "reason": "enabled_mcp_source_without_tool_result",
                    "source": source_name,
                    "repository": repository,
                    "activation_status": activation_status,
                }
            )
        metadata_gaps = _repository_metadata_gaps(source)
        if metadata_gaps:
            items.append(
                {
                    "reason": "repository_metadata_gap",
                    "source": source_name,
                    "repository": repository,
                    "missing_fields": metadata_gaps,
                }
            )
    return items[:50]


def _article_event_models(
    article: Article,
    source: Source,
    tracked_event_models: set[str],
) -> list[str]:
    values: set[str] = set()
    source_event_model = _source_event_model(source)
    if source_event_model in tracked_event_models:
        values.add(source_event_model)
    if _matches(article, "RiskScope") and "risk_scope_signal" in tracked_event_models:
        values.add("risk_scope_signal")
    return [event_model for event_model in TRACKED_EVENT_MODEL_ORDER if event_model in values]


def _source_status(
    *,
    source: Source,
    event_model: str,
    tracked_event_models: set[str],
    article_count: int,
    event_count: int,
    latest_event_at: datetime | None,
    sla_days: float | None,
    age_days: float | None,
) -> str:
    if not source.enabled:
        return "skipped_disabled"
    if event_model not in tracked_event_models:
        return "not_tracked"
    if article_count == 0:
        return "missing"
    if event_count == 0:
        return "missing_event"
    if latest_event_at is None or age_days is None:
        return "unknown_event_date"
    if sla_days is not None and age_days > sla_days:
        return "stale"
    return "fresh"


def _tracked_event_models(quality: Mapping[str, object]) -> set[str]:
    outputs = _dict(quality, "quality_outputs")
    raw = outputs.get("tracked_event_models")
    if isinstance(raw, list):
        values = {str(item).strip() for item in raw if str(item).strip()}
        return values & TRACKED_EVENT_MODELS or set(TRACKED_EVENT_MODELS)
    return set(TRACKED_EVENT_MODELS)


def _source_event_model(source: Source) -> str:
    raw = source.config.get("event_model")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    content_type = source.content_type.lower()
    source_type = source.type.lower()
    if content_type in {"mcp_directory", "directory"} or source_type == "github_readme_section":
        return "mcp_directory_entry"
    if content_type in {"mcp_tool_result", "mcp_tool", "mcp_result"} or source_type in {
        "mcp",
        "mcp_http",
        "mcp_sse",
        "mcp_server",
        "mcp_stdio",
        "mcp_streamable_http",
        "mcp_tool",
        "model_context_protocol",
    }:
        return "mcp_tool_result"
    if content_type in {"repository_metadata", "linked_repository_metadata"}:
        return "linked_repository_metadata"
    if content_type in {"risk", "security_risk", "risk_scope_signal"}:
        return "risk_scope_signal"
    return ""


def _is_mcp_server_source(source: Source) -> bool:
    source_type = source.type.lower()
    content_type = source.content_type.lower()
    return content_type in {"mcp_tool_result", "mcp_tool", "mcp_result"} or source_type in {
        "mcp",
        "mcp_http",
        "mcp_sse",
        "mcp_server",
        "mcp_stdio",
        "mcp_streamable_http",
        "mcp_tool",
        "model_context_protocol",
    }


def _source_repository(source: Source) -> str:
    raw = source.config.get("repository")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    link = source.url.strip()
    marker = "github.com/"
    if marker in link.lower():
        return link.split(marker, 1)[1].strip("/ ")
    return ""


def _repository_metadata_complete(source: Source) -> bool:
    return not _repository_metadata_gaps(source)


def _repository_metadata_gaps(source: Source) -> list[str]:
    gaps: list[str] = []
    if not _source_repository(source):
        gaps.append("repository")
    if source.config.get("github_archived") is None:
        gaps.append("github_archived")
    if not source.config.get("github_license"):
        gaps.append("github_license")
    if not source.config.get("github_pushed_at"):
        gaps.append("github_pushed_at")
    return gaps


def _source_sla_days(
    source: Source,
    event_model: str,
    freshness_sla: Mapping[str, object],
) -> float | None:
    raw_source_sla = source.config.get("freshness_sla_days")
    parsed_source_sla = _as_float(raw_source_sla)
    if parsed_source_sla is not None:
        return parsed_source_sla

    candidates = [f"{event_model}_days"]
    if event_model == "mcp_directory_entry":
        candidates.append("directory_days")
    elif event_model == "mcp_tool_result":
        candidates.extend(["mcp_tool_result_days", "tool_result_days", "directory_days"])
    elif event_model == "linked_repository_metadata":
        candidates.append("repository_metadata_days")
    elif event_model == "risk_scope_signal":
        candidates.extend(["risk_scope_days", "directory_days"])

    for key in candidates:
        parsed_days = _as_float(freshness_sla.get(key))
        if parsed_days is not None:
            return parsed_days
    return None


def _latest_event(event_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    dated: list[tuple[datetime, dict[str, Any]]] = []
    undated: list[dict[str, Any]] = []
    for row in event_rows:
        event_at = _parse_datetime(str(row.get("event_at") or ""))
        if event_at is not None:
            dated.append((event_at, row))
        else:
            undated.append(row)
    if dated:
        return max(dated, key=lambda item: item[0])[1]
    return undated[0] if undated else None


def _event_datetime(article: Article) -> datetime | None:
    article_time = article.published or article.collected_at
    return _as_utc(article_time) if article_time else None


def _required_field_proxy(
    article: Article,
    source: Source,
    event_model: str,
) -> dict[str, bool]:
    has_repository = "github.com/" in article.link.lower()
    if event_model == "mcp_directory_entry":
        return {
            "repository": has_repository,
            "source_section": bool(source.section),
            "source_url": bool(source.url),
        }
    if event_model == "mcp_tool_result":
        return {
            "repository": bool(source.config.get("repository")),
            "tool_name": bool(source.config.get("tools")),
            "source_url": bool(source.url),
        }
    if event_model == "linked_repository_metadata":
        return {
            "repository": has_repository,
            "archived": False,
            "license": False,
        }
    if event_model == "risk_scope_signal":
        return {
            "repository": has_repository,
            "risk_scope": bool(_matches(article, "RiskScope")),
            "evidence_url": bool(article.link),
        }
    return {}


def _matches(article: Article, entity_name: str) -> list[str]:
    values = article.matched_entities.get(entity_name, [])
    return [str(value) for value in values]


def _dict(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    raw = value.get(key)
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    return {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_datetime(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(parsed)


def _age_days(generated_at: datetime, event_at: datetime) -> float:
    return max(0.0, (_as_utc(generated_at) - _as_utc(event_at)).total_seconds() / 86400)

