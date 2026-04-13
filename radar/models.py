from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Article:
    """기사 데이터 모델"""

    title: str
    link: str
    summary: str = ""
    published: datetime | None = None
    source: str = ""
    category: str = ""
    matched_entities: dict[str, list[str]] = field(default_factory=dict)
    collected_at: datetime | None = None


@dataclass
class Source:
    """소스 정의"""

    name: str
    type: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    rate_limit: float = 1.0  # 초당 요청 수
    section: str = ""


@dataclass
class EntityDefinition:
    """엔티티 정의"""

    name: str
    display_name: str
    keywords: list[str]


@dataclass
class CategoryConfig:
    """카테고리 설정"""

    category_name: str
    display_name: str
    sources: list[Source] = field(default_factory=list)
    entities: list[EntityDefinition] = field(default_factory=list)


@dataclass
class TelegramSettings:
    """텔레그램 설정"""

    bot_token: str = ""
    chat_id: str = ""


@dataclass
class EmailSettings:
    """이메일 설정"""

    smtp_server: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_email: str = ""
    from_address: str = ""
    to_email: str = ""
    to_addresses: list[str] = field(default_factory=list)


@dataclass
class NotificationConfig:
    """알림 설정"""

    enabled: bool = True
    channels: list[str] = field(default_factory=list)
    email: EmailSettings | None = None
    webhook_url: str | None = None
    telegram: TelegramSettings | None = None
    rules: dict[str, object] = field(default_factory=dict)


@dataclass
class RadarSettings:
    """Radar 설정"""

    name: str = "Radar"
    version: str = "0.1.0"
    data_dir: str = "data"
    retention_days: int = 90
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    extra: dict[str, Any] = field(default_factory=dict)
    database_path: Path = Path("data/radar_data.duckdb")
    report_dir: Path = Path("reports")
    raw_data_dir: Path = Path("data/raw")
    search_db_path: Path = Path("data/search_index.db")


__all__ = [
    "Article",
    "CategoryConfig",
    "EmailSettings",
    "EntityDefinition",
    "NotificationConfig",
    "RadarSettings",
    "Source",
    "TelegramSettings",
]
