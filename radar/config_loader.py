from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import yaml

from .models import (
    CategoryConfig,
    EmailSettings,
    EntityDefinition,
    NotificationConfig,
    RadarSettings,
    Source,
    TelegramSettings,
)


def _resolve_path(path_value: str, *, project_root: Path) -> Path:
    """Resolve a path from config, treating relative paths as project-root relative."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _read_yaml_dict(path: Path) -> dict[str, object]:
    raw = cast(object, yaml.safe_load(path.read_text(encoding="utf-8")))
    if isinstance(raw, dict):
        raw_dict = cast(dict[object, object], raw)
        return {str(k): v for k, v in raw_dict.items()}
    return {}


def _string_value(raw: dict[str, object], key: str, default: str) -> str:
    value = raw.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return default


def _bool_value(raw: dict[str, object], key: str, default: bool) -> bool:
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return default


def _float_value(raw: dict[str, object], key: str, default: float) -> float:
    value = raw.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _string_list_value(raw: dict[str, object], key: str) -> list[str]:
    value = raw.get(key)
    values: list[object]
    if isinstance(value, list):
        values = cast(list[object], value)
    elif isinstance(value, tuple | set):
        values = list(cast(tuple[object, ...] | set[object], value))
    elif isinstance(value, str) and value.strip():
        values = [value]
    else:
        values = []
    return [str(item).strip() for item in values if str(item).strip()]


def _dict_value(raw: dict[str, object], key: str) -> dict[str, object]:
    value = raw.get(key)
    if isinstance(value, dict):
        return {str(k): v for k, v in cast(dict[object, object], value).items()}
    return {}


def _string_dict_value(raw: dict[str, object], key: str) -> dict[str, str]:
    return {k: str(v) for k, v in _dict_value(raw, key).items()}


def _dict_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []

    items: list[dict[str, object]] = []
    for item in cast(list[object], value):
        if isinstance(item, dict):
            item_dict = cast(dict[object, object], item)
            items.append({str(k): v for k, v in item_dict.items()})
    return items


def load_settings(config_path: Path | None = None) -> RadarSettings:
    """Load global radar settings such as database and report directories."""
    project_root = Path(__file__).resolve().parent.parent
    config_file = config_path or project_root / "config" / "config.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    raw = _read_yaml_dict(config_file)
    db_path = _resolve_path(
        _string_value(raw, "database_path", "data/radar_data.duckdb"), project_root=project_root
    )
    report_dir = _resolve_path(
        _string_value(raw, "report_dir", "reports"), project_root=project_root
    )
    raw_data_dir = _resolve_path(
        _string_value(raw, "raw_data_dir", "data/raw"), project_root=project_root
    )
    search_db_path = _resolve_path(
        _string_value(raw, "search_db_path", "data/search_index.db"), project_root=project_root
    )
    return RadarSettings(
        database_path=db_path,
        report_dir=report_dir,
        raw_data_dir=raw_data_dir,
        search_db_path=search_db_path,
    )


def load_category_config(category_name: str, categories_dir: Path | None = None) -> CategoryConfig:
    """Load a category YAML and parse it into a CategoryConfig object."""
    project_root = Path(__file__).resolve().parent.parent
    base_dir = categories_dir or project_root / "config" / "categories"
    config_file = Path(base_dir) / f"{category_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Category config not found: {config_file}")

    raw = _read_yaml_dict(config_file)
    sources = [_parse_source(entry) for entry in _dict_items(raw.get("sources"))]
    entities = [_parse_entity(entry) for entry in _dict_items(raw.get("entities"))]

    display_name = (
        _string_value(raw, "display_name", "")
        or _string_value(raw, "category_name", "")
        or category_name
    )

    return CategoryConfig(
        category_name=_string_value(raw, "category_name", category_name),
        display_name=display_name,
        sources=sources,
        entities=entities,
    )


def _parse_source(entry: dict[str, object]) -> Source:
    if not entry:
        raise ValueError("Empty source entry in category config")
    known_keys = {
        "name",
        "type",
        "url",
        "headers",
        "rate_limit",
        "section",
        "id",
        "enabled",
        "language",
        "country",
        "region",
        "trust_tier",
        "weight",
        "content_type",
        "collection_tier",
        "producer_role",
        "info_purpose",
        "notes",
        "config",
    }
    config = _dict_value(entry, "config")
    for key, value in entry.items():
        if key not in known_keys:
            config[key] = value
    return Source(
        name=_string_value(entry, "name", "Unnamed Source"),
        type=_string_value(entry, "type", "rss"),
        url=_string_value(entry, "url", ""),
        headers=_string_dict_value(entry, "headers"),
        rate_limit=_float_value(entry, "rate_limit", 1.0),
        section=_string_value(entry, "section", ""),
        id=_string_value(entry, "id", ""),
        enabled=_bool_value(entry, "enabled", True),
        language=_string_value(entry, "language", ""),
        country=_string_value(entry, "country", ""),
        region=_string_value(entry, "region", ""),
        trust_tier=_string_value(entry, "trust_tier", "T3_professional"),
        weight=_float_value(entry, "weight", 1.0),
        content_type=_string_value(entry, "content_type", "news"),
        collection_tier=_string_value(entry, "collection_tier", "C1_rss"),
        producer_role=_string_value(entry, "producer_role", ""),
        info_purpose=_string_list_value(entry, "info_purpose"),
        notes=_string_value(entry, "notes", ""),
        config=config,
    )


def _parse_entity(entry: dict[str, object]) -> EntityDefinition:
    if not entry:
        raise ValueError("Empty entity entry in category config")
    name = _string_value(entry, "name", "entity")
    display_name = _string_value(entry, "display_name", name)
    keywords_raw = entry.get("keywords")
    keywords: list[object]
    if isinstance(keywords_raw, list):
        keywords = []
        for keyword in cast(list[object], keywords_raw):
            keywords.append(keyword)
    elif isinstance(keywords_raw, tuple | set):
        keywords = []
        for keyword in cast(tuple[object, ...] | set[object], keywords_raw):
            keywords.append(keyword)
    else:
        keywords = []
    keyword_list = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    return EntityDefinition(name=name, display_name=display_name, keywords=keyword_list)


def _resolve_env_refs(value: object) -> object:
    """Resolve ${VAR} environment variable references in strings."""
    if isinstance(value, str):
        result = value
        import re

        for match in re.finditer(r"\$\{([^}]+)\}", value):
            var_name = match.group(1)
            env_value = os.environ.get(var_name, "")
            result = result.replace(match.group(0), env_value)
        return result
    elif isinstance(value, dict):
        return {k: _resolve_env_refs(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_refs(item) for item in value]
    return value


def load_category_quality_config(
    category_name: str, categories_dir: Path | None = None
) -> dict[str, object]:
    """Load quality metadata from a category YAML."""
    project_root = Path(__file__).resolve().parent.parent
    base_dir = categories_dir or project_root / "config" / "categories"
    config_file = Path(base_dir) / f"{category_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Category config not found: {config_file}")

    raw = _read_yaml_dict(config_file)
    return {
        "data_quality": _dict_value(raw, "data_quality"),
        "source_backlog": _dict_value(raw, "source_backlog"),
    }


def load_notification_config(
    config_path: Path | None = None,
) -> NotificationConfig:
    """Load notification configuration from notifications.yaml.

    Args:
        config_path: Path to notifications.yaml. If None, uses project_root/config/notifications.yaml

    Returns:
        NotificationConfig with resolved environment variables

    Raises:
        FileNotFoundError: If notifications.yaml does not exist
    """
    project_root = Path(__file__).resolve().parent.parent
    config_file = config_path or project_root / "config" / "notifications.yaml"

    if not config_file.exists():
        return NotificationConfig(enabled=False, channels=[])

    raw = _read_yaml_dict(config_file)
    notifications_raw = raw.get("notifications", {})
    if not isinstance(notifications_raw, dict):
        return NotificationConfig(enabled=False, channels=[])

    notifications_dict = cast(dict[str, object], notifications_raw)
    enabled = bool(notifications_dict.get("enabled", False))
    channels_raw = notifications_dict.get("channels", [])
    channels = [str(c) for c in cast(list[object], channels_raw) if isinstance(c, str)]

    email_settings = None
    email_raw = notifications_dict.get("email")
    if isinstance(email_raw, dict):
        email_dict = cast(dict[str, object], _resolve_env_refs(email_raw))
        try:
            smtp_port_raw = email_dict.get("smtp_port", 587)
            smtp_port = int(smtp_port_raw) if isinstance(smtp_port_raw, (int, str)) else 587
            email_settings = EmailSettings(
                smtp_host=_string_value(email_dict, "smtp_host", ""),
                smtp_port=smtp_port,
                username=_string_value(email_dict, "username", ""),
                password=_string_value(email_dict, "password", ""),
                from_address=_string_value(email_dict, "from_address", ""),
                to_addresses=[
                    str(addr)
                    for addr in cast(list[object], email_dict.get("to_addresses", []))
                    if isinstance(addr, str)
                ],
            )
        except (ValueError, KeyError):
            email_settings = None

    webhook_url = None
    webhook_raw = notifications_dict.get("webhook_url")
    if isinstance(webhook_raw, str):
        resolved = _resolve_env_refs(webhook_raw)
        webhook_url = str(resolved) if resolved else None

    telegram_settings = None
    telegram_raw = notifications_dict.get("telegram")
    if isinstance(telegram_raw, dict):
        telegram_dict = cast(dict[str, object], _resolve_env_refs(telegram_raw))
        try:
            telegram_settings = TelegramSettings(
                bot_token=_string_value(telegram_dict, "bot_token", ""),
                chat_id=_string_value(telegram_dict, "chat_id", ""),
            )
        except (ValueError, KeyError):
            telegram_settings = None

    rules_raw = notifications_dict.get("rules", {})
    rules = (
        cast(dict[str, object], _resolve_env_refs(rules_raw)) if isinstance(rules_raw, dict) else {}
    )

    return NotificationConfig(
        enabled=enabled,
        channels=channels,
        email=email_settings,
        webhook_url=webhook_url,
        telegram=telegram_settings,
        rules=rules,
    )


def filter_sources(
    sources: list[Source],
    *,
    max_sources: int | None = None,
    exclude_sources: tuple[str, ...] | list[str] = (),
) -> list[Source]:
    """Filter sources by exclusion id/name and a hard max count.

    Mirrors ``radar_core.config_loader.filter_sources`` so MCP-style repos with
    a vendored ``radar`` package can call it locally.
    """
    excluded_tokens: set[str] = set()
    if exclude_sources:
        for token in exclude_sources:
            if not isinstance(token, str):
                continue
            normalized = token.strip().lower()
            if normalized:
                excluded_tokens.add(normalized)

    filtered: list[Source]
    if excluded_tokens:
        filtered = []
        for source in sources:
            source_id = (getattr(source, "id", "") or "").strip().lower()
            source_name = (getattr(source, "name", "") or "").strip().lower()
            match_token = source_id if source_id else source_name
            if match_token and match_token in excluded_tokens:
                continue
            if source_name and source_name in excluded_tokens:
                continue
            filtered.append(source)
    else:
        filtered = list(sources)

    if isinstance(max_sources, int) and max_sources > 0:
        filtered = filtered[:max_sources]

    return filtered
