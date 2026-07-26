from __future__ import annotations

import base64
import binascii
import os
import re
import shlex
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, model_validator


app = FastAPI(
    title="Agent Pre-Tool-Call Guardrail",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

AGENT_HOME = Path("/home/agent")
AGENT_WORKDIR = Path("/home/agent/workspace")
PROTECTED_FILE = Path("/home/agent/.npmrc")
WRITE_ROOT = Path("/workspace/output")

ALLOWED_HTTP_HOSTS = {
    "raw.githubusercontent.com",
    "registry.npmjs.org",
}

EXTRA_BLOCKED_READS = {
    Path("/etc/shadow"),
}

REDIRECT_RE = re.compile(
    r"(?:^|[\s;|&])(?:\d*>>?|\d*>\|)\s*(?P<path>"
    r"'[^']+'|\"[^\"]+\"|[^\s;&|]+)"
)

BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/]{16,}={0,2})"
    r"(?![A-Za-z0-9+/=_-])"
)

SHELL_C_RE = re.compile(
    r"(?:^|[\s;&|])(?:/bin/)?(?:ba|z|da)?sh\s+-c\s+"
    r"(?P<quote>['\"])(?P<body>.*?)(?P=quote)",
    re.IGNORECASE | re.DOTALL,
)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["bash", "write_file", "http_request"]
    command: str | None = None
    path: str | None = None
    content: str | None = None
    method: Literal["GET", "POST"] | None = None
    url: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "ToolCall":
        if self.tool == "bash":
            if self.command is None:
                raise ValueError("bash requires command")
            if any(
                value is not None
                for value in (self.path, self.content, self.method, self.url)
            ):
                raise ValueError("Invalid bash tool-call shape")

        elif self.tool == "write_file":
            if self.path is None or self.content is None:
                raise ValueError("write_file requires path and content")
            if any(
                value is not None
                for value in (self.command, self.method, self.url)
            ):
                raise ValueError("Invalid write_file tool-call shape")

        elif self.tool == "http_request":
            if self.method is None or self.url is None:
                raise ValueError("http_request requires method and url")
            if any(
                value is not None
                for value in (self.command, self.path, self.content)
            ):
                raise ValueError("Invalid http_request tool-call shape")

        return self


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["allow", "block"]
    reason: str


def expand_agent_path(raw: str) -> str:
    """
    Deterministically expand path forms relevant to the policy.

    This does not execute a shell.
    """
    value = raw.strip().strip("'\"")

    value = value.replace("${HOME}", str(AGENT_HOME))
    value = value.replace("$HOME", str(AGENT_HOME))

    if value == "~":
        value = str(AGENT_HOME)
    elif value.startswith("~/"):
        value = str(AGENT_HOME / value[2:])

    value = value.replace(r"\/", "/")
    value = value.replace(r"\.", ".")

    return value


def normalise_path(raw: str, cwd: Path = AGENT_WORKDIR) -> Path:
    """
    Lexically normalise a POSIX path without requiring it to exist.

    Relative bash paths are resolved from /home/agent/workspace.
    """
    expanded = expand_agent_path(raw)

    if not expanded.startswith("/"):
        expanded = str(cwd / expanded)

    return Path(os.path.normpath(expanded))


def is_within(path: Path, root: Path) -> bool:
    """
    Return True only when path is root itself or a descendant of root.
    """
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def tokenise(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        # Fallback still exposes path-like tokens for inspection.
        cleaned = re.sub(r"[;|&()]", " ", command)
        return cleaned.split()


def decoded_base64_payloads(command: str) -> list[str]:
    decoded: list[str] = []

    for match in BASE64_RE.finditer(command):
        token = match.group(1)

        try:
            raw = base64.b64decode(token, validate=True)
            text = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue

        if text and all(
            character.isprintable() or character in "\r\n\t"
            for character in text
        ):
            decoded.append(text)

    return decoded


def recursively_unwrap(command: str, max_depth: int = 4) -> list[str]:
    """
    Inspect the original command, nested shell -c bodies, and printable
    base64-decoded payloads.
    """
    queue: list[tuple[str, int]] = [(command, 0)]
    seen: set[str] = set()
    result: list[str] = []

    while queue:
        text, depth = queue.pop(0)

        if text in seen:
            continue

        seen.add(text)
        result.append(text)

        if depth >= max_depth:
            continue

        for match in SHELL_C_RE.finditer(text):
            queue.append((match.group("body"), depth + 1))

        for payload in decoded_base64_payloads(text):
            queue.append((payload, depth + 1))

    return result


def candidate_path_tokens(command: str) -> list[str]:
    candidates: list[str] = []

    for token in tokenise(command):
        cleaned = token.strip("(){}[],").strip("'\"")

        if (
            "/" in cleaned
            or cleaned.startswith("~")
            or "$HOME" in cleaned
            or "${HOME}" in cleaned
            or cleaned == ".npmrc"
            or cleaned.startswith("../")
        ):
            candidates.append(cleaned)

    for match in REDIRECT_RE.finditer(command):
        candidates.append(match.group("path").strip("'\""))

    return candidates


def references_blocked_read(command: str) -> Path | None:
    for layer in recursively_unwrap(command):
        expanded_layer = expand_agent_path(layer)

        for blocked in {PROTECTED_FILE, *EXTRA_BLOCKED_READS}:
            if str(blocked) in expanded_layer:
                return blocked

        for raw_path in candidate_path_tokens(layer):
            try:
                path = normalise_path(raw_path)
            except (TypeError, ValueError, OSError):
                continue

            if path == PROTECTED_FILE:
                return PROTECTED_FILE

            if path in EXTRA_BLOCKED_READS:
                return path

    return None


def split_shell_segments(tokens: list[str]) -> list[list[str]]:
    """
    Split a tokenized command into simple segments around shell operators.
    """
    segments: list[list[str]] = [[]]

    for token in tokens:
        if token in {";", "&&", "||", "|"}:
            if segments[-1]:
                segments.append([])
        else:
            segments[-1].append(token)

    return [segment for segment in segments if segment]


def non_option_operands(segment: list[str]) -> list[str]:
    """
    Return likely positional operands.

    This is intentionally conservative and deterministic.
    """
    return [
        token.strip("'\"")
        for token in segment[1:]
        if token and not token.startswith("-")
    ]


def bash_write_destinations(command: str) -> list[Path]:
    """
    Extract common bash write destinations, including traversal attempts.

    Supported forms include:
      > file
      >> file
      2> file
      tee file
      touch file
      mkdir dir
      cp source destination
      mv source destination
      install source destination
      printf ... > file
      echo ... > file

    Every destination is normalised before policy comparison.
    """
    destinations: list[Path] = []

    for layer in recursively_unwrap(command):
        # Detect all shell redirection destinations.
        for match in REDIRECT_RE.finditer(layer):
            raw_destination = match.group("path").strip("'\"")
            destinations.append(normalise_path(raw_destination))

        tokens = tokenise(layer)

        if not tokens:
            continue

        for segment in split_shell_segments(tokens):
            executable = Path(segment[0]).name
            operands = non_option_operands(segment)

            if not operands:
                continue

            if executable == "tee":
                # All non-option tee operands are output files.
                for operand in operands:
                    destinations.append(normalise_path(operand))

            elif executable in {"touch", "mkdir", "truncate"}:
                # Each operand is a write/create destination.
                for operand in operands:
                    destinations.append(normalise_path(operand))

            elif executable in {"cp", "mv", "install"} and len(operands) >= 2:
                # The final positional operand is the destination.
                destinations.append(normalise_path(operands[-1]))

            elif executable == "dd":
                # Detect dd of=/path syntax.
                for token in segment[1:]:
                    if token.startswith("of="):
                        raw_destination = token[3:].strip("'\"")
                        destinations.append(normalise_path(raw_destination))

    return destinations


def decide_bash(command: str) -> Decision:
    blocked_path = references_blocked_read(command)

    if blocked_path == PROTECTED_FILE:
        return Decision(
            decision="block",
            reason="Reading /home/agent/.npmrc is forbidden because it contains secrets.",
        )

    if blocked_path in EXTRA_BLOCKED_READS:
        return Decision(
            decision="block",
            reason=f"Reading {blocked_path} is never permitted by this agent's policy.",
        )

    for destination in bash_write_destinations(command):
        if not is_within(destination, WRITE_ROOT):
            return Decision(
                decision="block",
                reason="Bash writes are allowed only inside /workspace/output/.",
            )

    return Decision(
        decision="allow",
        reason="The bash command does not violate the protected-read or write-boundary rules.",
    )


def decide_write_file(path_text: str) -> Decision:
    path = normalise_path(path_text)

    if not is_within(path, WRITE_ROOT):
        return Decision(
            decision="block",
            reason="Files may be written only inside /workspace/output/.",
        )

    return Decision(
        decision="allow",
        reason="The normalised destination is inside /workspace/output/.",
    )


def decide_http(url: str) -> Decision:
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
    except ValueError:
        parsed = None
        hostname = None

    if (
        parsed is None
        or parsed.scheme.lower() not in {"http", "https"}
        or not hostname
    ):
        return Decision(
            decision="block",
            reason="The request URL must be a valid absolute HTTP or HTTPS URL.",
        )

    host = hostname.rstrip(".").lower()

    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return Decision(
            decision="block",
            reason="The request hostname is invalid.",
        )

    if host not in ALLOWED_HTTP_HOSTS:
        return Decision(
            decision="block",
            reason="Outbound HTTP is allowed only to the exact approved hostnames.",
        )

    return Decision(
        decision="allow",
        reason=f"Hostname {host} exactly matches the outbound allowlist.",
    )


@app.get("/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/guardrail", response_model=Decision)
def guardrail(call: ToolCall) -> Decision:
    if call.tool == "bash":
        return decide_bash(call.command or "")

    if call.tool == "write_file":
        return decide_write_file(call.path or "")

    return decide_http(call.url or "")
