from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import requests

from .exceptions import NetworkError, SourceError
from .models import Article, Source


MCP_SOURCE_TYPES = {
    "mcp",
    "mcp_http",
    "mcp_sse",
    "mcp_server",
    "mcp_stdio",
    "mcp_streamable_http",
    "mcp_tool",
    "model_context_protocol",
}


@dataclass(frozen=True)
class MCPToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPSourceConfig:
    transport: str
    command: str = ""
    args: tuple[str, ...] = ()
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    tools: tuple[MCPToolCall, ...] = ()
    resources: tuple[str, ...] = ()
    timeout_seconds: int = 15
    max_items: int = 30


def collect_mcp_server_source(
    source: Source,
    *,
    category: str,
    limit: int,
    timeout: int,
) -> list[Article]:
    """Collect articles from an explicitly configured MCP server source.

    This adapter is intentionally allowlist-driven: it only invokes tool names or
    resource URIs present in source config. Disabled sources are skipped by the
    caller and should remain disabled until activation gates pass.
    """
    config = parse_mcp_source_config(source, timeout=timeout, limit=limit)
    payloads = collect_mcp_payloads(source, config)
    return normalize_mcp_payloads(payloads, source=source, category=category, limit=config.max_items)


def parse_mcp_source_config(source: Source, *, timeout: int, limit: int) -> MCPSourceConfig:
    raw = dict(source.config)
    transport = _string(raw, "transport") or _transport_from_type(source.type)
    if not transport and source.url:
        transport = "streamable_http"
    if not transport:
        transport = "stdio"
    transport = transport.lower().replace("-", "_")

    tools = tuple(_parse_tools(raw))
    resources = tuple(_string_list(raw.get("resources") or raw.get("resource")))
    timeout_seconds = _int(raw.get("timeout_seconds"), timeout)
    max_items = _int(raw.get("max_items"), limit)
    url = _string(raw, "url") or _string(raw, "server_url") or _string(raw, "mcp_url") or source.url

    return MCPSourceConfig(
        transport=transport,
        command=_string(raw, "command"),
        args=tuple(_string_list(raw.get("args"))),
        url=url,
        headers={**source.headers, **_string_dict(raw.get("headers"))},
        env=_resolve_env(raw.get("env")),
        tools=tools,
        resources=resources,
        timeout_seconds=max(1, timeout_seconds),
        max_items=max(1, max_items),
    )


def collect_mcp_payloads(source: Source, config: MCPSourceConfig) -> list[Any]:
    if not config.tools and not config.resources:
        raise SourceError(source.name, "mcp_server source requires allowed tools or resources")

    if config.transport == "stdio":
        return _collect_stdio_payloads(source, config)
    if config.transport in {"streamable_http", "http", "mcp_http"}:
        return _collect_streamable_http_payloads(source, config)
    if config.transport in {"sse", "mcp_sse"}:
        raise SourceError(source.name, "SSE MCP transport is not enabled by the stdlib adapter")
    raise SourceError(source.name, f"Unsupported MCP transport '{config.transport}'")


def normalize_mcp_payloads(
    payloads: list[Any],
    *,
    source: Source,
    category: str,
    limit: int,
) -> list[Article]:
    articles: list[Article] = []
    for payload in payloads:
        for item in _iter_payload_items(payload):
            article = _payload_item_to_article(item, source=source, category=category)
            if article is not None:
                articles.append(article)
                if len(articles) >= limit:
                    return articles
    return articles


def _collect_stdio_payloads(source: Source, config: MCPSourceConfig) -> list[Any]:
    if not config.command:
        raise SourceError(source.name, "stdio mcp_server source requires config.command")
    try:
        return asyncio.run(_run_stdio_session(source, config))
    except TimeoutError as exc:
        raise NetworkError(f"MCP stdio timeout for {source.name}: {exc}") from exc


async def _run_stdio_session(source: Source, config: MCPSourceConfig) -> list[Any]:
    env = {**os.environ, **config.env}
    command = _resolve_command(config.command)
    try:
        process = await asyncio.create_subprocess_exec(
            command,
            *config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except OSError as exc:
        raise SourceError(source.name, f"Failed to start MCP server command: {exc}") from exc

    request_id = 1
    payloads: list[Any] = []
    try:
        await _stdio_send(
            process,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "MCPRadar", "version": "0.1.0"},
                },
            },
        )
        await _stdio_read_result(process, request_id, timeout=config.timeout_seconds)
        await _stdio_send(
            process,
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        )

        for tool in config.tools:
            request_id += 1
            await _stdio_send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": tool.name, "arguments": tool.arguments},
                },
            )
            payloads.append(await _stdio_read_result(process, request_id, timeout=config.timeout_seconds))

        for uri in config.resources:
            request_id += 1
            await _stdio_send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "resources/read",
                    "params": {"uri": uri},
                },
            )
            payloads.append(await _stdio_read_result(process, request_id, timeout=config.timeout_seconds))
    finally:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            process.kill()
    return payloads


def _resolve_command(command: str) -> str:
    """Resolve commands through PATHEXT on Windows and PATH elsewhere."""
    resolved = shutil.which(command)
    return resolved or command


async def _stdio_send(process: asyncio.subprocess.Process, payload: dict[str, Any]) -> None:
    if process.stdin is None:
        raise RuntimeError("MCP stdio process has no stdin")
    process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    await process.stdin.drain()


async def _stdio_read_result(
    process: asyncio.subprocess.Process,
    request_id: int,
    *,
    timeout: int,
) -> Any:
    if process.stdout is None:
        raise RuntimeError("MCP stdio process has no stdout")
    while True:
        raw = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
        if not raw:
            stderr = ""
            if process.stderr is not None:
                stderr = (await process.stderr.read()).decode("utf-8", errors="replace")
            raise RuntimeError(f"MCP stdio process ended before response {request_id}: {stderr}")
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict) or message.get("id") != request_id:
            continue
        return _jsonrpc_result(message)


def _collect_streamable_http_payloads(source: Source, config: MCPSourceConfig) -> list[Any]:
    if not config.url:
        raise SourceError(source.name, "streamable_http mcp_server source requires config.url")

    headers = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        **config.headers,
    }
    session = requests.Session()
    payloads: list[Any] = []
    request_id = 1
    try:
        init_response = _post_jsonrpc(
            session,
            config.url,
            headers=headers,
            timeout=config.timeout_seconds,
            payload={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "MCPRadar", "version": "0.1.0"},
                },
            },
        )
        session_id = init_response.headers.get("Mcp-Session-Id")
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        _ = _jsonrpc_result(_response_json(init_response))
        _post_jsonrpc(
            session,
            config.url,
            headers=headers,
            timeout=config.timeout_seconds,
            payload={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
        )

        for tool in config.tools:
            request_id += 1
            response = _post_jsonrpc(
                session,
                config.url,
                headers=headers,
                timeout=config.timeout_seconds,
                payload={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": tool.name, "arguments": tool.arguments},
                },
            )
            payloads.append(_jsonrpc_result(_response_json(response)))

        for uri in config.resources:
            request_id += 1
            response = _post_jsonrpc(
                session,
                config.url,
                headers=headers,
                timeout=config.timeout_seconds,
                payload={
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "resources/read",
                    "params": {"uri": uri},
                },
            )
            payloads.append(_jsonrpc_result(_response_json(response)))
    except requests.exceptions.RequestException as exc:
        raise NetworkError(f"MCP HTTP error for {source.name}: {exc}") from exc
    finally:
        session.close()
    return payloads


def _post_jsonrpc(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    payload: dict[str, Any],
) -> requests.Response:
    response = session.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def _response_json(response: requests.Response) -> dict[str, Any]:
    content_type = response.headers.get("Content-Type", "")
    if "text/event-stream" in content_type:
        for line in response.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line.removeprefix("data:").strip())
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("MCP JSON-RPC response must be an object")
    return data


def _jsonrpc_result(message: Mapping[str, Any]) -> Any:
    error = message.get("error")
    if error:
        raise RuntimeError(f"MCP JSON-RPC error: {error}")
    return message.get("result")


def _iter_payload_items(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        items: list[Any] = []
        for value in payload:
            items.extend(_iter_payload_items(value))
        return items
    if isinstance(payload, str):
        parsed = _try_parse_json(payload)
        return _iter_payload_items(parsed) if parsed is not None else [payload]
    if isinstance(payload, dict):
        if "result" in payload:
            return _iter_payload_items(payload["result"])
        if "content" in payload:
            return _iter_content_blocks(payload["content"])
        if "contents" in payload:
            return _iter_content_blocks(payload["contents"])
        return [payload]
    return [payload]


def _iter_content_blocks(blocks: Any) -> list[Any]:
    if not isinstance(blocks, list):
        return _iter_payload_items(blocks)
    items: list[Any] = []
    for block in blocks:
        if isinstance(block, dict):
            if "json" in block:
                items.extend(_iter_payload_items(block["json"]))
                continue
            if "text" in block:
                items.extend(_iter_payload_items(block["text"]))
                continue
        items.extend(_iter_payload_items(block))
    return items


def _payload_item_to_article(item: Any, *, source: Source, category: str) -> Article | None:
    now = datetime.now(UTC)
    if isinstance(item, str):
        text = item.strip()
        if not text:
            return None
        return Article(
            title=_first_line(text, default=source.name),
            link=_fallback_link(source),
            summary=text,
            published=now,
            source=source.name,
            category=category,
        )

    if not isinstance(item, Mapping):
        return Article(
            title=source.name,
            link=_fallback_link(source),
            summary=json.dumps(item, ensure_ascii=False),
            published=now,
            source=source.name,
            category=category,
        )

    title = _first_nonempty(item, "title", "name", "repository", "id") or source.name
    link = (
        _first_nonempty(item, "url", "link", "uri", "repository_url", "homepage")
        or _repository_link(_first_nonempty(item, "repository"))
        or _fallback_link(source)
    )
    summary = _first_nonempty(item, "summary", "description", "text", "content")
    if not summary:
        summary = json.dumps(dict(item), ensure_ascii=False, sort_keys=True)
    return Article(
        title=title,
        link=link,
        summary=summary,
        published=now,
        source=source.name,
        category=category,
    )


def _parse_tools(raw: dict[str, Any]) -> list[MCPToolCall]:
    raw_tools = raw.get("tools") or raw.get("tool")
    values = raw_tools if isinstance(raw_tools, list) else [raw_tools]
    tools: list[MCPToolCall] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            tools.append(MCPToolCall(name=value.strip()))
        elif isinstance(value, dict):
            name = str(value.get("name") or value.get("tool") or "").strip()
            arguments = value.get("arguments") or value.get("args") or value.get("input") or {}
            if name and isinstance(arguments, dict):
                tools.append(MCPToolCall(name=name, arguments=dict(arguments)))
    return tools


def _resolve_env(value: Any) -> dict[str, str]:
    if isinstance(value, list):
        return {str(name): os.environ.get(str(name), "") for name in value if str(name).strip()}
    if not isinstance(value, dict):
        return {}
    env: dict[str, str] = {}
    for key, raw_value in value.items():
        env_name = str(key).strip()
        if not env_name:
            continue
        text_value = str(raw_value)
        if text_value.startswith("${") and text_value.endswith("}"):
            env[env_name] = os.environ.get(text_value[2:-1], "")
        else:
            env[env_name] = text_value
    return env


def _transport_from_type(source_type: str) -> str:
    normalized = source_type.lower()
    if normalized == "mcp_stdio":
        return "stdio"
    if normalized in {"mcp_http", "mcp_streamable_http"}:
        return "streamable_http"
    if normalized == "mcp_sse":
        return "sse"
    return ""


def _string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _try_parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _first_nonempty(item: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_line(text: str, *, default: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:160]
    return default


def _repository_link(value: str) -> str:
    if value and "/" in value and not value.startswith(("http://", "https://")):
        return f"https://github.com/{value.strip()}"
    return ""


def _fallback_link(source: Source) -> str:
    if source.url:
        return source.url
    if source.id:
        return f"mcp://{source.id}"
    return f"mcp://{source.name.replace(' ', '_')}"
