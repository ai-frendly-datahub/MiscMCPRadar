from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_runtime_contract_dir(search_from: Path | None = None) -> Path | None:
    candidates: list[Path] = []

    runtime_dir_env = os.getenv("RADAR_ONTOLOGY_RUNTIME_DIR", "").strip()
    if runtime_dir_env:
        candidates.append(Path(runtime_dir_env).expanduser())

    ontology_dir_env = os.getenv("RADAR_ONTOLOGY_DIR", "").strip()
    if ontology_dir_env:
        candidates.append(Path(ontology_dir_env).expanduser() / "runtime_contracts")

    search_roots = [Path.cwd()]
    if search_from is not None:
        search_roots.append(search_from)
    search_roots.append(Path(__file__).resolve())

    seen: set[Path] = set()
    for root in search_roots:
        base = root if root.is_dir() else root.parent
        for parent in (base, *base.parents):
            candidates.append(parent / "radar-ontology" / "runtime_contracts")
            candidates.append(parent / "runtime_contracts")

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            return resolved
    return None


def load_runtime_contract(
    repo_name: str,
    *,
    runtime_contract_dir: Path | None = None,
    search_from: Path | None = None,
) -> dict[str, Any] | None:
    contract_dir = runtime_contract_dir or resolve_runtime_contract_dir(search_from)
    if contract_dir is None:
        return None

    contract_path = contract_dir / f"{repo_name}.json"
    if not contract_path.exists():
        return None

    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_summary_ontology_metadata(
    repo_name: str,
    *,
    category_name: str | None = None,
    runtime_contract_dir: Path | None = None,
    search_from: Path | None = None,
) -> dict[str, Any] | None:
    contract = load_runtime_contract(
        repo_name,
        runtime_contract_dir=runtime_contract_dir,
        search_from=search_from,
    )
    if contract is None:
        return None

    event_model_mappings = _string_mapping(contract.get("event_model_mappings"))
    source_role_mappings = _string_mapping(contract.get("source_role_mappings"))
    entity_type_hints = _string_list(contract.get("entity_type_hints"))
    evidence_policy_ids = _string_list(contract.get("evidence_policy_ids"))

    metadata = {
        "repo": repo_name,
        "category": str(contract.get("category") or category_name or "").strip(),
        "ontology_version": str(contract.get("ontology_version") or "").strip(),
        "event_model_ids": sorted(set(event_model_mappings.values())),
        "event_model_mappings": event_model_mappings,
        "entity_type_hints": entity_type_hints,
        "source_role_ids": sorted(set(source_role_mappings.values())),
        "source_role_mappings": source_role_mappings,
        "evidence_policy_ids": evidence_policy_ids,
    }
    return {key: value for key, value in metadata.items() if _has_value(value)}


def build_article_ontology_metadata(
    repo_name: str,
    *,
    source_name: str,
    source_event_model: str | None = None,
    category_name: str | None = None,
    runtime_contract_dir: Path | None = None,
    search_from: Path | None = None,
) -> dict[str, Any] | None:
    contract = load_runtime_contract(
        repo_name,
        runtime_contract_dir=runtime_contract_dir,
        search_from=search_from,
    )
    if contract is None:
        return None

    source_name_normalized = source_name.strip()
    source_event_model_normalized = str(source_event_model or "").strip()
    event_model_mappings = _string_mapping(contract.get("event_model_mappings"))
    source_role_mappings = _string_mapping(contract.get("source_role_mappings"))
    evidence_policy_ids = _string_list(contract.get("evidence_policy_ids"))

    metadata = {
        "repo": repo_name,
        "category": str(contract.get("category") or category_name or "").strip(),
        "ontology_version": str(contract.get("ontology_version") or "").strip(),
        "source_role_id": source_role_mappings.get(source_name_normalized, "").strip(),
        "source_event_model": source_event_model_normalized,
        "event_model_id": event_model_mappings.get(source_event_model_normalized, "").strip(),
        "evidence_policy_ids": evidence_policy_ids,
    }
    return {key: value for key, value in metadata.items() if _has_value(value)}


def _shadow_default_published_at(article: object) -> str | None:
    value = getattr(article, "published", None)
    if value is None:
        return None
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        try:
            return iso()
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    return text or None


def _shadow_default_tags(article: object) -> list[str] | None:
    matched = getattr(article, "matched_entities", None)
    if not isinstance(matched, Mapping):
        return None
    keys = sorted({str(k).strip() for k in matched.keys() if str(k).strip()})
    return keys or None


_SHADOW_FIELD_EXTRACTORS = {
    "source_name": "source",
    "headline": "title",
    "source_url": "link",
    "summary": "summary",
}


def _shadow_build_event_model_payload(
    article: object,
    *,
    repo_name: str,
    event_model_key: str,
    overrides: Mapping[str, object] | None,
    runtime_contract_dir: Path | None,
    search_from: Path | None,
) -> dict[str, object] | None:
    contract = load_runtime_contract(
        repo_name,
        runtime_contract_dir=runtime_contract_dir,
        search_from=search_from,
    )
    if contract is None:
        return None
    field_specs = contract.get("event_model_field_specs")
    if not isinstance(field_specs, Mapping):
        return None
    spec = field_specs.get(event_model_key.strip())
    if not isinstance(spec, Mapping):
        return None
    declared: list[str] = []
    seen: set[str] = set()
    for fname in [*(spec.get("required_fields") or []), *(spec.get("optional_fields") or [])]:
        if not isinstance(fname, str) or not fname.strip() or fname in seen:
            continue
        seen.add(fname)
        declared.append(fname)
    overrides_map: Mapping[str, object] = overrides or {}
    payload: dict[str, object] = {}
    for fname in declared:
        if fname in overrides_map:
            value = overrides_map[fname]
        elif fname == "published_at":
            value = _shadow_default_published_at(article)
        elif fname == "tags":
            value = _shadow_default_tags(article)
        elif fname in _SHADOW_FIELD_EXTRACTORS:
            attr = _SHADOW_FIELD_EXTRACTORS[fname]
            value = getattr(article, attr, None)
            if isinstance(value, str):
                value = value.strip() or None
        else:
            value = None
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        if isinstance(value, (list, tuple)) and not value:
            continue
        payload[fname] = list(value) if isinstance(value, tuple) else value
    return payload




def _shadow_extract_source_payload_overrides(source: object) -> Mapping[str, object] | None:
    if source is None:
        return None
    config = getattr(source, "config", None)
    if not isinstance(config, Mapping):
        return None
    raw = config.get("event_model_payload_overrides")
    if not isinstance(raw, Mapping):
        return None
    cleaned: dict[str, object] = {}
    for key, value in raw.items():
        normalized_key = str(key).strip()
        if not normalized_key:
            continue
        cleaned[normalized_key] = value
    return cleaned or None


def annotate_articles_with_ontology(
    articles: list[Any],
    *,
    repo_name: str,
    sources_by_name: Mapping[str, object],
    category_name: str | None = None,
    runtime_contract_dir: Path | None = None,
    search_from: Path | None = None,
    attach_event_model_payload: bool = False,
    payload_overrides_by_source: Mapping[str, Mapping[str, object]] | None = None,
    strict_enums: bool = False,  # noqa: ARG001 — shadow does not enforce enums yet
) -> list[Any]:
    explicit_overrides = payload_overrides_by_source is not None
    overrides_table: Mapping[str, Mapping[str, object]] = (
        payload_overrides_by_source if payload_overrides_by_source else {}
    )
    for article in articles:
        source_name = str(getattr(article, "source", "") or "").strip()
        source = sources_by_name.get(source_name)
        source_event_model = _extract_source_event_model(source)
        metadata = build_article_ontology_metadata(
            repo_name,
            source_name=source_name,
            source_event_model=source_event_model,
            category_name=category_name,
            runtime_contract_dir=runtime_contract_dir,
            search_from=search_from,
        )
        if metadata is None:
            continue
        if (
            attach_event_model_payload
            and source_event_model
            and metadata.get("event_model_id")
        ):
            if explicit_overrides:
                source_overrides = overrides_table.get(source_name)
            else:
                source_overrides = _shadow_extract_source_payload_overrides(source)
            payload = _shadow_build_event_model_payload(
                article,
                repo_name=repo_name,
                event_model_key=source_event_model,
                overrides=source_overrides,
                runtime_contract_dir=runtime_contract_dir,
                search_from=search_from,
            )
            if payload:
                metadata["event_model_payload"] = payload
        setattr(article, "ontology", metadata)
    return articles


def backfill_duckdb_ontology(
    db_path: Path,
    *,
    repo_name: str,
    sources_by_name: Mapping[str, object],
    category_name: str | None = None,
    runtime_contract_dir: Path | None = None,
    search_from: Path | None = None,
    table_name: str = "articles",
) -> dict[str, int]:
    """Re-emit ontology_json for existing rows using current source config + runtime contract.

    Returns a dict with `scanned`, `updated`, and `with_event_model_id` counts.
    Safe to run repeatedly: only rewrites rows whose ontology_json would change.
    """
    import duckdb  # imported lazily so radar-core consumers without duckdb still work

    quoted_table = '"' + table_name.replace('"', '""') + '"'
    counts = {"scanned": 0, "updated": 0, "with_event_model_id": 0}
    with duckdb.connect(str(db_path)) as conn:
        existing_columns = {
            str(row[0])
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ?",
                [table_name],
            ).fetchall()
        }
        if "ontology_json" not in existing_columns or "link" not in existing_columns:
            return counts

        rows = conn.execute(
            f"SELECT link, source, ontology_json FROM {quoted_table}"
        ).fetchall()
        updates: list[tuple[str, str]] = []
        for link, source_name, existing_ontology_json in rows:
            counts["scanned"] += 1
            normalized_source = str(source_name or "").strip()
            source = sources_by_name.get(normalized_source)
            source_event_model = _extract_source_event_model(source)
            metadata = build_article_ontology_metadata(
                repo_name,
                source_name=normalized_source,
                source_event_model=source_event_model,
                category_name=category_name,
                runtime_contract_dir=runtime_contract_dir,
                search_from=search_from,
            )
            if metadata is None:
                continue
            new_payload = json.dumps(metadata, ensure_ascii=False)
            if existing_ontology_json == new_payload:
                if metadata.get("event_model_id"):
                    counts["with_event_model_id"] += 1
                continue
            updates.append((new_payload, str(link)))
            if metadata.get("event_model_id"):
                counts["with_event_model_id"] += 1

        if updates:
            conn.executemany(
                f"UPDATE {quoted_table} SET ontology_json = ? WHERE link = ?",
                updates,
            )
            counts["updated"] = len(updates)
    return counts


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _string_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        normalized_key = str(key).strip()
        normalized_value = str(item).strip()
        if normalized_key and normalized_value:
            normalized[normalized_key] = normalized_value
    return normalized


def _extract_source_event_model(source: object) -> str | None:
    if source is None:
        return None
    config = getattr(source, "config", None)
    if not isinstance(config, Mapping):
        return None
    raw_value = str(config.get("event_model") or "").strip()
    return raw_value or None


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True
