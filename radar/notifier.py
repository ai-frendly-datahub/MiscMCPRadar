# Canonical Notifier implementation for AI-Friendly DataHub
# Synced to: HomeRadar, BenefitRadar, WineRadar, TrendRadar, PriceRadar
# DO NOT MODIFY core classes (Notifier, NotificationPayload, EmailNotifier, WebhookNotifier, CompositeNotifier)
# Domain-specific detection functions (detect_*_notifications) belong in each domain repo

from __future__ import annotations

import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from typing import Protocol

import requests
import structlog


logger = structlog.get_logger(__name__)


@dataclass
class NotificationPayload:
    """Payload for notification delivery."""

    category_name: str
    sources_count: int
    collected_count: int
    matched_count: int
    errors_count: int
    timestamp: datetime
    report_url: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert payload to dictionary for JSON serialization."""
        return {
            "category_name": self.category_name,
            "sources_count": self.sources_count,
            "collected_count": self.collected_count,
            "matched_count": self.matched_count,
            "errors_count": self.errors_count,
            "timestamp": self.timestamp.isoformat(),
            "report_url": self.report_url,
        }


class Notifier(Protocol):
    """Protocol for notification delivery."""

    def send(self, payload: NotificationPayload) -> bool:
        """Send notification. Return True if successful, False otherwise."""
        ...


class EmailNotifier:
    """Send notifications via email using SMTP."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        from_addr: str,
        to_addrs: list[str],
    ) -> None:
        """Initialize email notifier.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            from_addr: Sender email address
            to_addrs: List of recipient email addresses
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def send(self, payload: NotificationPayload) -> bool:
        """Send email notification.

        Args:
            payload: Notification payload

        Returns:
            True if successful, False otherwise
        """
        try:
            subject = f"Radar Pipeline Complete: {payload.category_name}"
            body = self._build_email_body(payload)

            msg = MIMEText(body, "plain")
            msg["Subject"] = subject
            msg["From"] = self.from_addr
            msg["To"] = ", ".join(self.to_addrs)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info("email_notification_sent", category=payload.category_name)
            return True
        except Exception as e:
            logger.error(
                "email_notification_failed",
                category=payload.category_name,
                error=str(e),
            )
            return False

    def _build_email_body(self, payload: NotificationPayload) -> str:
        """Build email body from payload."""
        lines = [
            "Radar Pipeline Completion Report",
            "================================",
            "",
            f"Category: {payload.category_name}",
            f"Timestamp: {payload.timestamp.isoformat()}",
            "",
            "Statistics:",
            f"  Sources: {payload.sources_count}",
            f"  Collected: {payload.collected_count}",
            f"  Matched: {payload.matched_count}",
            f"  Errors: {payload.errors_count}",
        ]
        if payload.report_url:
            lines.append("")
            lines.append(f"Report: {payload.report_url}")
        return "\n".join(lines)


class WebhookNotifier:
    """Send notifications via HTTP webhook."""

    def __init__(
        self,
        url: str,
        method: str = "POST",
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize webhook notifier.

        Args:
            url: Webhook URL
            method: HTTP method (POST or GET)
            headers: Optional HTTP headers
        """
        self.url = url
        self.method = method.upper()
        self.headers = headers or {}

    def send(self, payload: NotificationPayload) -> bool:
        """Send webhook notification.

        Args:
            payload: Notification payload

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.method == "POST":
                response = requests.post(
                    self.url,
                    json=payload.to_dict(),
                    headers=self.headers,
                    timeout=10,
                )
            elif self.method == "GET":
                response = requests.get(
                    self.url,
                    headers=self.headers,
                    timeout=10,
                )
            else:
                logger.error(
                    "webhook_invalid_method",
                    method=self.method,
                    url=self.url,
                )
                return False

            if response.status_code >= 400:
                logger.error(
                    "webhook_notification_failed",
                    url=self.url,
                    status_code=response.status_code,
                )
                return False

            logger.info("webhook_notification_sent", url=self.url)
            return True
        except Exception as e:
            logger.error(
                "webhook_notification_failed",
                url=self.url,
                error=str(e),
            )
            return False


class CompositeNotifier:
    """Send notifications to multiple notifiers."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        """Initialize composite notifier.

        Args:
            notifiers: List of notifiers to send to
        """
        self.notifiers = notifiers

    def send(self, payload: NotificationPayload) -> bool:
        """Send notification to all notifiers.

        Args:
            payload: Notification payload

        Returns:
            True if all notifiers succeeded, False if any failed
        """
        if not self.notifiers:
            return True

        results = []
        for notifier in self.notifiers:
            try:
                result = notifier.send(payload)
                results.append(result)
            except Exception:
                results.append(False)
        return all(results) if results else True
