from __future__ import annotations

import json
import os
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

ACTIVATION_GATE_PRIORITY = [
    "command_or_endpoint_unresolved",
    "env_secret_documentation_required",
    "tool_resource_allowlist_required",
    "upstream_runtime_config_patch_required",
    "reliable_stdio_initialize_required",
    "upstream_startup_regression_review_required",
    "bounded_real_transport_preflight_required",
    "bootstrap_performance_review_required",
    "prebuilt_cache_readiness_required",
    "fake_transport_smoke_test_required",
    "real_transport_smoke_test_required",
    "registry_crosscheck_required",
    "risk_scope_review_required",
    "production_monitoring_required",
    "production_enablement_review_required",
]
RUNTIME_REVIEW_GATES = {
    "upstream_runtime_config_patch_required",
    "reliable_stdio_initialize_required",
    "upstream_startup_regression_review_required",
    "bounded_real_transport_preflight_required",
    "bootstrap_performance_review_required",
    "prebuilt_cache_readiness_required",
    "production_monitoring_required",
    "production_enablement_review_required",
}



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
    events.extend(
        _build_repository_metadata_event_rows(
            sources=category.sources,
            tracked_event_models=tracked_event_models,
            generated_at=generated,
            freshness_sla=freshness_sla,
        )
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
    repository_metadata_rows = [
        metadata
        for row in source_rows
        if (metadata := _mapping(row.get("repository_metadata")))
    ]
    repository_metadata_status_counts = Counter(
        str(row.get("status") or "unknown") for row in repository_metadata_rows
    )

    status_counts = Counter(str(row["status"]) for row in source_rows)
    event_counts = Counter(str(row["event_model"]) for row in events)
    mcp_server_sources = [source for source in category.sources if _is_mcp_server_source(source)]

    env_preflight_rows = [_env_preflight_status(source) for source in mcp_server_sources]
    activation_gate_sets = [_activation_gate_set(source) for source in mcp_server_sources]
    activation_gate_counts = Counter(gate for gates in activation_gate_sets for gate in gates)
    activation_command_discovery_status_counts = Counter(
        status
        for source in mcp_server_sources
        if (status := str(source.config.get("command_discovery_status") or ""))
    )
    activation_command_discovery_resolved_count = sum(
        count
        for status, count in activation_command_discovery_status_counts.items()
        if status.startswith("resolved_")
    )
    runtime_review_source_count = sum(
        1 for gates in activation_gate_sets if gates & RUNTIME_REVIEW_GATES
    )
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
        "env_preflight_required_source_count": sum(
            1 for row in env_preflight_rows if row["required_env"]
        ),
        "env_preflight_ready_source_count": sum(
            1 for row in env_preflight_rows if row["status"] == "ready"
        ),
        "env_preflight_missing_source_count": sum(
            1 for row in env_preflight_rows if row["status"] == "missing_required_env"
        ),
        "env_preflight_missing_var_count": sum(
            len(row["missing_env"]) for row in env_preflight_rows
        ),
        "env_preflight_not_required_source_count": sum(
            1 for row in env_preflight_rows if row["status"] == "not_required"
        ),
        "repository_metadata_complete_source_count": sum(
            1 for source in mcp_server_sources if _repository_metadata_complete(source)
        ),
        "repository_metadata_gap_count": sum(
            len(_repository_metadata_gaps(source)) for source in mcp_server_sources
        ),
        "repository_metadata_fresh_source_count": repository_metadata_status_counts.get("fresh", 0),
        "repository_metadata_stale_source_count": repository_metadata_status_counts.get("stale", 0),
        "repository_metadata_incomplete_source_count": repository_metadata_status_counts.get(
            "incomplete", 0
        ),
        "repository_metadata_missing_checked_at_source_count": repository_metadata_status_counts.get(
            "missing_checked_at", 0
        ),
        "repository_metadata_review_required_count": sum(
            count
            for status, count in repository_metadata_status_counts.items()
            if status != "fresh"
        ),
        "repository_docs_present_source_count": sum(
            1
            for row in repository_metadata_rows
            if bool(row.get("github_readme_present") or row.get("github_docs_present"))
        ),
        "repository_docs_missing_source_count": sum(
            1 for row in repository_metadata_rows if row.get("github_docs_present") is False
        ),
        "repository_security_policy_present_source_count": sum(
            1 for row in repository_metadata_rows if row.get("github_security_policy_present") is True
        ),
        "repository_security_advisory_checked_source_count": sum(
            1
            for row in repository_metadata_rows
            if str(row.get("github_security_advisory_access_status") or "").startswith("checked")
        ),
        "repository_security_advisory_total_count": sum(
            _as_int(row.get("github_security_advisory_count"), 0)
            for row in repository_metadata_rows
        ),
        "repository_security_advisory_open_source_count": sum(
            1
            for row in repository_metadata_rows
            if _as_int(row.get("github_security_advisory_open_count"), 0) > 0
        ),
        "repository_security_enrichment_review_required_count": sum(
            1
            for row in repository_metadata_rows
            if row.get("github_docs_present") is False
            or not str(row.get("github_security_advisory_access_status") or "").startswith("checked")
            or _as_int(row.get("github_security_advisory_open_count"), 0) > 0
        ),
        "security_activation_gate_count": sum(
            len(_list(source.config.get("activation_gates"))) for source in mcp_server_sources
        ),

        "activation_gate_source_count": sum(1 for gates in activation_gate_sets if gates),
        "activation_gate_total_count": sum(len(gates) for gates in activation_gate_sets),
        "activation_risk_scope_review_required_source_count": activation_gate_counts.get(
            "risk_scope_review_required", 0
        ),
        "activation_command_unresolved_source_count": activation_gate_counts.get(
            "command_or_endpoint_unresolved", 0
        ),
        "activation_command_discovery_checked_source_count": sum(
            activation_command_discovery_status_counts.values()
        ),
        "activation_command_discovery_resolved_source_count": (
            activation_command_discovery_resolved_count
        ),
        "activation_command_discovery_unresolved_source_count": (
            sum(activation_command_discovery_status_counts.values())
            - activation_command_discovery_resolved_count
        ),
        "activation_command_discovery_multi_server_ambiguous_source_count": (
            activation_command_discovery_status_counts.get("multi_server_ambiguous", 0)
        ),
        "activation_command_discovery_status_counts": dict(
            sorted(activation_command_discovery_status_counts.items())
        ),
        "activation_env_secret_required_source_count": activation_gate_counts.get(
            "env_secret_documentation_required", 0
        ),
        "activation_tool_allowlist_required_source_count": activation_gate_counts.get(
            "tool_resource_allowlist_required", 0
        ),
        "activation_registry_crosscheck_required_source_count": activation_gate_counts.get(
            "registry_crosscheck_required", 0
        ),
        "activation_fake_transport_required_source_count": activation_gate_counts.get(
            "fake_transport_smoke_test_required", 0
        ),
        "activation_real_transport_required_source_count": activation_gate_counts.get(
            "real_transport_smoke_test_required", 0
        ),
        "activation_runtime_review_required_source_count": runtime_review_source_count,
        "activation_ready_for_fake_transport_source_count": sum(
            1
            for source in mcp_server_sources
            if str(source.config.get("activation_status") or "")
            == "candidate_ready_for_fake_transport_test"
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


def _build_repository_metadata_event_rows(
    *,
    sources: list[Source],
    tracked_event_models: set[str],
    generated_at: datetime,
    freshness_sla: Mapping[str, object],
) -> list[dict[str, Any]]:
    if "linked_repository_metadata" not in tracked_event_models:
        return []

    rows: list[dict[str, Any]] = []
    for source in sources:
        if not _is_mcp_server_source(source):
            continue
        metadata = _repository_metadata_status(
            source=source,
            generated_at=generated_at,
            freshness_sla=freshness_sla,
        )
        repository = str(metadata.get("repository") or "")
        event_at = (
            _parse_datetime(str(metadata.get("github_pushed_at") or ""))
            or _parse_datetime(str(metadata.get("checked_at") or ""))
            or generated_at
        )
        rows.append(
            {
                "source": source.name,
                "event_model": "linked_repository_metadata",
                "title": f"{repository or source.name} repository metadata",
                "url": _repository_url(source),
                "event_at": event_at.isoformat(),
                "mcp_domain": [],
                "provider": [],
                "capability": [],
                "risk_scope": [],
                "project_health": [],
                "required_field_proxy": {
                    "repository": bool(repository),
                    "archived": metadata.get("github_archived") is not None,
                    "license": bool(metadata.get("github_license")),
                    "docs": bool(metadata.get("github_docs_present")),
                    "security_advisory_check": str(
                        metadata.get("github_security_advisory_access_status") or ""
                    ).startswith("checked"),
                },
                "repository": repository,
                "repository_metadata_status": metadata.get("status"),
                "repository_metadata_checked_at": metadata.get("checked_at"),
                "github_pushed_at": metadata.get("github_pushed_at"),
                "github_license": metadata.get("github_license"),
                "github_archived": metadata.get("github_archived"),
                "github_readme_present": metadata.get("github_readme_present"),
                "github_docs_present": metadata.get("github_docs_present"),
                "github_docs_paths": metadata.get("github_docs_paths"),
                "github_security_policy_present": metadata.get("github_security_policy_present"),
                "github_security_advisory_access_status": metadata.get(
                    "github_security_advisory_access_status"
                ),
                "github_security_advisory_count": metadata.get("github_security_advisory_count"),
                "github_security_advisory_open_count": metadata.get(
                    "github_security_advisory_open_count"
                ),
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
    repository_metadata = _repository_metadata_status(
        source=source,
        generated_at=generated_at,
        freshness_sla=freshness_sla,
    )
    env_preflight = _env_preflight_status(source)

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
        "activation_gate_count": len(_activation_gate_set(source)),
        "activation_next_gate": _activation_next_gate(source),
        "activation_command_discovery_status": str(
            source.config.get("command_discovery_status", "")
        ),
        "activation_command_discovery_checked_at": str(
            source.config.get("command_discovery_checked_at", "")
        ),
        "tools_count": len(_list(source.config.get("tools"))),
        "env_count": len(_env_required_names(source)),
        "env_preflight_status": env_preflight["status"],
        "env_required_names": env_preflight["required_env"],
        "env_missing_names": env_preflight["missing_env"],
        "repository_metadata_gaps": _list(repository_metadata.get("missing_fields"))
        or _repository_metadata_gaps(source),
        "repository_metadata": repository_metadata,
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
        env_preflight = _env_preflight_status(source)
        if env_preflight["status"] == "missing_required_env":
            items.append(
                {
                    "reason": "mcp_env_preflight_missing",
                    "source": source_name,
                    "repository": repository,
                    "missing_env": env_preflight["missing_env"],
                    "activation_status": activation_status,
                }
            )
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
        repository_metadata = _mapping(row.get("repository_metadata"))
        metadata_status = str(repository_metadata.get("status") or "")
        if metadata_status == "stale":
            items.append(
                {
                    "reason": "repository_metadata_stale",
                    "source": source_name,
                    "repository": repository,
                    "checked_at": repository_metadata.get("checked_at", ""),
                    "age_days": repository_metadata.get("age_days"),
                    "freshness_sla_days": repository_metadata.get("freshness_sla_days"),
                }
            )
        elif metadata_status == "missing_checked_at":
            items.append(
                {
                    "reason": "repository_metadata_missing_checked_at",
                    "source": source_name,
                    "repository": repository,
                    "detail": "Repository metadata exists but has no metadata_checked_at timestamp.",
                }
            )
        metadata_gaps = _list(repository_metadata.get("missing_fields")) or _repository_metadata_gaps(source)
        if metadata_gaps:
            items.append(
                {
                    "reason": "repository_metadata_gap",
                    "source": source_name,
                    "repository": repository,
                    "missing_fields": metadata_gaps,
                    "repository_metadata_status": metadata_status,
                }
            )
        if repository_metadata.get("github_docs_present") is False:
            items.append(
                {
                    "reason": "repository_docs_gap",
                    "source": source_name,
                    "repository": repository,
                    "detail": "Repository README/docs were not found during docs/advisory audit.",
                }
            )
        advisory_status = str(
            repository_metadata.get("github_security_advisory_access_status") or ""
        )
        if advisory_status and not advisory_status.startswith("checked"):
            items.append(
                {
                    "reason": "repository_security_advisory_unchecked",
                    "source": source_name,
                    "repository": repository,
                    "access_status": advisory_status,
                    "detail": "Repository security advisory endpoint was not fully checked.",
                }
            )
        advisory_open_count = _as_int(
            repository_metadata.get("github_security_advisory_open_count"), 0
        )
        if advisory_open_count > 0:
            items.append(
                {
                    "reason": "repository_security_advisory_open",
                    "source": source_name,
                    "repository": repository,
                    "open_advisory_count": advisory_open_count,
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



def _env_required_names(source: Source) -> list[str]:
    raw = source.config.get("env")
    if isinstance(raw, Mapping):
        return [str(name).strip() for name in raw if str(name).strip()]
    return [str(name).strip() for name in _list(raw) if str(name).strip()]


def _env_resolved_values(source: Source) -> dict[str, str]:
    raw = source.config.get("env")
    if isinstance(raw, list):
        return {str(name).strip(): os.environ.get(str(name).strip(), "") for name in raw if str(name).strip()}
    if not isinstance(raw, Mapping):
        return {}

    values: dict[str, str] = {}
    for key, raw_value in raw.items():
        env_name = str(key).strip()
        if not env_name:
            continue
        text_value = "" if raw_value is None else str(raw_value)
        if text_value.startswith("${") and text_value.endswith("}"):
            values[env_name] = os.environ.get(text_value[2:-1], "")
        else:
            values[env_name] = text_value
    return values


def _env_missing_names(source: Source) -> list[str]:
    values = _env_resolved_values(source)
    return [name for name in _env_required_names(source) if not values.get(name, "").strip()]


def _env_preflight_status(source: Source) -> dict[str, Any]:
    required = _env_required_names(source)
    missing = _env_missing_names(source)
    if not required:
        status = "not_required"
    elif missing:
        status = "missing_required_env"
    else:
        status = "ready"
    return {"status": status, "required_env": required, "missing_env": missing}


def _activation_gate_set(source: Source) -> set[str]:
    return set(_list(source.config.get("activation_gates")))


def _activation_next_gate(source: Source) -> str:
    gates = _activation_gate_set(source)
    for gate in ACTIVATION_GATE_PRIORITY:
        if gate in gates:
            return gate
    return sorted(gates)[0] if gates else ""


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


def _repository_metadata_status(
    *,
    source: Source,
    generated_at: datetime,
    freshness_sla: Mapping[str, object],
) -> dict[str, Any]:
    if not _is_mcp_server_source(source):
        return {}

    missing_fields = _repository_metadata_gaps(source)
    checked_at = _parse_datetime(str(source.config.get("metadata_checked_at") or ""))
    sla_days = _source_sla_days(source, "linked_repository_metadata", freshness_sla)
    age_days = _age_days(generated_at, checked_at) if checked_at else None

    if missing_fields:
        status = "incomplete"
    elif checked_at is None:
        status = "missing_checked_at"
    elif sla_days is not None and age_days is not None and age_days > sla_days:
        status = "stale"
    else:
        status = "fresh"

    return {
        "status": status,
        "repository": _source_repository(source),
        "checked_at": checked_at.isoformat() if checked_at else "",
        "age_days": round(age_days, 2) if age_days is not None else None,
        "freshness_sla_days": sla_days,
        "missing_fields": missing_fields,
        "github_pushed_at": str(source.config.get("github_pushed_at") or ""),
        "github_license": str(source.config.get("github_license") or ""),
        "github_archived": source.config.get("github_archived"),
        "github_disabled": source.config.get("github_disabled"),
        "docs_advisory_checked_at": str(source.config.get("docs_advisory_checked_at") or ""),
        "github_readme_present": source.config.get("github_readme_present"),
        "github_readme_path": str(source.config.get("github_readme_path") or ""),
        "github_docs_present": source.config.get("github_docs_present"),
        "github_docs_paths": _list(source.config.get("github_docs_paths")),
        "github_security_policy_present": source.config.get("github_security_policy_present"),
        "github_security_policy_paths": _list(source.config.get("github_security_policy_paths")),
        "github_security_advisory_access_status": str(
            source.config.get("github_security_advisory_access_status") or ""
        ),
        "github_security_advisory_count": _as_int(
            source.config.get("github_security_advisory_count"), 0
        ),
        "github_security_advisory_open_count": _as_int(
            source.config.get("github_security_advisory_open_count"), 0
        ),
        "github_security_advisory_published_count": _as_int(
            source.config.get("github_security_advisory_published_count"), 0
        ),
        "github_security_advisory_state_counts": dict(
            _mapping(source.config.get("github_security_advisory_state_counts"))
        ),
        "github_security_advisory_ids": _list(source.config.get("github_security_advisory_ids")),
    }


def _repository_url(source: Source) -> str:
    if source.url:
        return source.url
    repository = _source_repository(source)
    if repository:
        return f"https://github.com/{repository}"
    return ""


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


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    return {}


def _list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


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

