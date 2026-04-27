from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from .models import Article, CategoryConfig


_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,
    )


def _copy_static_assets(report_dir: Path) -> None:
    src = _TEMPLATE_DIR / "static"
    dst = report_dir / "static"
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        _ = shutil.copytree(str(src), str(dst))


def generate_report(
    *,
    category: CategoryConfig,
    articles: Iterable[Article],
    output_path: Path,
    stats: dict[str, int],
    errors: list[str] | None = None,
    quality_report: Mapping[str, Any] | None = None,
) -> Path:
    """Render a simple HTML report for the collected articles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    articles_list = list(articles)
    entity_counts = _count_entities(articles_list)

    articles_json: list[dict[str, object]] = []
    for article in articles_list:
        article_data: dict[str, object] = {
            "title": article.title,
            "link": article.link,
            "source": article.source,
            "published": article.published.isoformat() if article.published else None,
            "published_at": article.published.isoformat() if article.published else None,
            "summary": article.summary,
            "matched_entities": article.matched_entities or {},
            "collected_at": article.collected_at.isoformat()
            if hasattr(article, "collected_at") and article.collected_at
            else None,
        }
        articles_json.append(article_data)

    now = datetime.now(tz=UTC)

    env = _get_jinja_env()
    template = env.get_template("report.html")
    rendered = template.render(
        category=category,
        articles=articles_list,
        articles_json=articles_json,
        generated_at=now,
        stats=stats,
        entity_counts=entity_counts,
        errors=errors or [],
        quality_report=quality_report or {},
    )

    _ = output_path.write_text(rendered, encoding="utf-8")

    date_stamp = now.strftime("%Y%m%d")
    dated_name = f"{category.category_name}_{date_stamp}.html"
    dated_path = output_path.parent / dated_name
    _ = dated_path.write_text(rendered, encoding="utf-8")

    _generate_summary_json(
        category_name=category.category_name,
        articles=articles_json,
        stats=stats,
        output_dir=output_path.parent,
        ontology_metadata=_load_ontology_metadata(
            "MiscMCPRadar",
            category_name=category.category_name,
        ),
    )

    _copy_static_assets(output_path.parent)

    return output_path


def _count_entities(articles: Iterable[Article]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for article in articles:
        for entity_name, keywords in (article.matched_entities or {}).items():
            counter[entity_name] += len(keywords)
    return counter


def generate_index_html(report_dir: Path) -> Path:
    """Generate an index.html that lists all available report files."""
    report_dir.mkdir(parents=True, exist_ok=True)

    _date_pattern = re.compile(r"^(.+)_(\d{8})$")

    dated_entries: list[tuple[str, str, dict[str, str]]] = []
    latest_entries: list[dict[str, str]] = []
    for html_file in report_dir.glob("*.html"):
        if html_file.name == "index.html":
            continue

        stem = html_file.stem
        m = _date_pattern.match(stem)
        if m:
            base = m.group(1)
            raw_date = m.group(2)
            display_name = base.replace("_report", "").replace("_", " ").title()
            date_label = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            dated_entries.append(
                (
                    raw_date,
                    base,
                    {
                        "filename": html_file.name,
                        "display_name": display_name,
                        "date": date_label,
                    },
                )
            )
            continue

        latest_entries.append(
            {
                "filename": html_file.name,
                "display_name": stem.replace("_report", "").replace("_", " ").title(),
                "date": "",
            }
        )

    dated_entries.sort(key=lambda t: (t[0], t[1]), reverse=True)
    latest_entries.sort(key=lambda r: r["filename"])

    reports = [t[2] for t in dated_entries] + latest_entries

    env = _get_jinja_env()
    template = env.get_template("index.html")
    rendered = template.render(
        reports=reports,
        generated_at=datetime.now(UTC),
    )

    index_path = report_dir / "index.html"
    _ = index_path.write_text(rendered, encoding="utf-8")
    return index_path


def _generate_summary_json(
    *,
    category_name: str,
    articles: list[dict[str, object]],
    stats: dict[str, int],
    output_dir: Path,
    ontology_metadata: Mapping[str, object] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    source_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    matched_count = 0

    for article in articles:
        source = article.get("source")
        if isinstance(source, str) and source:
            source_counts[source] += 1

        matched_entities = article.get("matched_entities")
        if not isinstance(matched_entities, Mapping):
            continue
        if matched_entities:
            matched_count += 1
        for entity_name, keywords in matched_entities.items():
            if not isinstance(entity_name, str) or not entity_name:
                continue
            if isinstance(keywords, list):
                entity_counts[entity_name] += len(keywords)
            else:
                entity_counts[entity_name] += 1

    now = datetime.now(tz=UTC)
    date_stamp = now.strftime("%Y%m%d")
    summary: dict[str, object] = {
        "date": now.date().isoformat(),
        "category": category_name,
        "article_count": int(stats.get("article_count", len(articles))),
        "source_count": int(stats.get("source_count", len(source_counts))),
        "matched_count": int(stats.get("matched_count", matched_count)),
        "top_entities": [
            {"name": name, "count": count}
            for name, count in entity_counts.most_common(20)
        ],
        "sources": dict(source_counts),
        "generated_at": now.isoformat(),
    }
    if ontology_metadata:
        summary["ontology"] = dict(ontology_metadata)

    output_path = output_dir / f"{category_name}_{date_stamp}_summary.json"
    _ = output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def _load_ontology_metadata(
    repo_name: str,
    *,
    category_name: str,
) -> Mapping[str, object] | None:
    contract_dir = _resolve_runtime_contract_dir()
    if contract_dir is None:
        return None

    contract_path = contract_dir / f"{repo_name}.json"
    if not contract_path.exists():
        return None

    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None

    event_model_mappings = _string_mapping(payload.get("event_model_mappings"))
    source_role_mappings = _string_mapping(payload.get("source_role_mappings"))
    entity_type_hints = _string_list(payload.get("entity_type_hints"))
    evidence_policy_ids = _string_list(payload.get("evidence_policy_ids"))
    metadata: dict[str, object] = {
        "repo": repo_name,
        "category": str(payload.get("category") or category_name).strip(),
        "ontology_version": str(payload.get("ontology_version") or "").strip(),
        "event_model_ids": sorted(set(event_model_mappings.values())),
        "event_model_mappings": event_model_mappings,
        "entity_type_hints": entity_type_hints,
        "source_role_ids": sorted(set(source_role_mappings.values())),
        "source_role_mappings": source_role_mappings,
        "evidence_policy_ids": evidence_policy_ids,
    }
    return {key: value for key, value in metadata.items() if _has_value(value)}


def _resolve_runtime_contract_dir() -> Path | None:
    candidates: list[Path] = []

    runtime_dir_env = os.getenv("RADAR_ONTOLOGY_RUNTIME_DIR", "").strip()
    if runtime_dir_env:
        candidates.append(Path(runtime_dir_env).expanduser())

    ontology_dir_env = os.getenv("RADAR_ONTOLOGY_DIR", "").strip()
    if ontology_dir_env:
        candidates.append(Path(ontology_dir_env).expanduser() / "runtime_contracts")

    for root in (Path.cwd(), Path(__file__).resolve()):
        base = root if root.is_dir() else root.parent
        for parent in (base, *base.parents):
            candidates.append(parent / "radar-ontology" / "runtime_contracts")
            candidates.append(parent / "runtime_contracts")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir():
            return resolved
    return None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


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


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True
