"""Minimal MCP client over stdio.

This file intentionally avoids the official MCP SDK so the protocol shape stays
visible: initialize -> tools/list -> tools/call. It is enough for local demo
servers that speak newline-delimited JSON-RPC over stdin/stdout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

JsonObject = dict[str, Any]


class MCPError(RuntimeError):
    """Raised when the MCP server returns a JSON-RPC error or invalid response."""


@dataclass
class MCPTool:
    """Tool metadata discovered from `tools/list`."""

    name: str
    description: str = ""
    input_schema: JsonObject = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: JsonObject) -> "MCPTool":
        return cls(
            name=str(payload.get("name", "")),
            description=str(payload.get("description", "")),
            input_schema=dict(payload.get("inputSchema") or {}),
        )

    def to_openai_tool(self) -> JsonObject:
        """Convert MCP tool metadata into an OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


@dataclass
class MCPClientInfo:
    """Identity sent to the server during `initialize`."""

    name: str = "multi-agent-mcp-client"
    version: str = "0.1.0"


class MCPStdioClient:
    """A small synchronous MCP client for local stdio servers.

    The client owns one server subprocess. `stdout` is reserved for protocol
    messages, so MCP servers should write logs to `stderr`.
    """

    def __init__(
        self,
        command: list[str],
        *,
        cwd: str | Path | None = None,
        protocol_version: str = "2024-11-05",
        client_info: MCPClientInfo | None = None,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        self.command = command
        self.cwd = str(cwd) if cwd is not None else None
        self.protocol_version = protocol_version
        self.client_info = client_info or MCPClientInfo()
        self._request_id = 0
        self._process: subprocess.Popen[str] | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Start the MCP server process if it is not already running."""
        if self.is_running:
            return
        self._process = subprocess.Popen(
            self.command,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )

    def close(self) -> None:
        """Stop the server process owned by this client."""
        process = self._process
        if process is None:
            return
        self._process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def __enter__(self) -> "MCPStdioClient":
        self.start()
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def initialize(self) -> JsonObject:
        """Open an MCP session and send the `initialized` notification."""
        result = self.request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_info.name,
                    "version": self.client_info.version,
                },
            },
        )
        self.notify("notifications/initialized", {})
        return result

    def list_tools(self) -> list[MCPTool]:
        """Discover tools exposed by the connected MCP server."""
        result = self.request("tools/list", {})
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            raise MCPError("Invalid tools/list result: expected a tools array")
        return [MCPTool.from_json(tool) for tool in tools if isinstance(tool, dict)]

    def call_tool(self, name: str, arguments: JsonObject | None = None) -> JsonObject:
        """Call a discovered MCP tool and return the raw MCP result object."""
        return self.request(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )

    def call_tool_text(self, name: str, arguments: JsonObject | None = None) -> str:
        """Call a tool and flatten text content blocks into one observation string."""
        result = self.call_tool(name, arguments)
        if result.get("isError"):
            raise MCPError(_extract_text_content(result) or f"Tool {name!r} failed")
        return _extract_text_content(result)

    def request(self, method: str, params: JsonObject | None = None) -> JsonObject:
        """Send a JSON-RPC request and wait for the matching response."""
        self.start()
        request_id = self._next_request_id()
        self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )

        while True:
            response = self._read_message()
            if response.get("id") != request_id:
                # Servers may emit notifications while a request is in flight.
                continue
            if "error" in response:
                raise MCPError(f"MCP {method} failed: {response['error']}")
            result = response.get("result", {})
            if not isinstance(result, dict):
                raise MCPError(f"MCP {method} returned a non-object result")
            return result

    def notify(self, method: str, params: JsonObject | None = None) -> None:
        """Send a JSON-RPC notification without waiting for a response."""
        self.start()
        self._write_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {},
            }
        )

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _stdio(self) -> tuple[TextIO, TextIO]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise MCPError("MCP server process is not running")
        if process.poll() is not None:
            raise MCPError(f"MCP server exited with code {process.returncode}")
        return process.stdin, process.stdout

    def _write_message(self, message: JsonObject) -> None:
        stdin, _ = self._stdio()
        stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        stdin.flush()

    def _read_message(self) -> JsonObject:
        _, stdout = self._stdio()
        while True:
            line = stdout.readline()
            if line == "":
                process = self._process
                code = process.returncode if process is not None else None
                raise MCPError(f"MCP server closed stdout (exit code: {code})")
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MCPError(f"Invalid JSON-RPC message from MCP server: {line}") from exc
            if not isinstance(payload, dict):
                raise MCPError("Invalid JSON-RPC message: expected object")
            return payload


def _extract_text_content(result: JsonObject) -> str:
    content = result.get("content", [])
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)
