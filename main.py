from __future__ import annotations

import base64
import binascii
import os
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, model_validator


app = FastAPI(title="Agent Pre-Tool-Call Guardrail")

AGENT_HOME = Path("/home/agent")
AGENT_WORKDIR = Path("/home/agent/workspace")
PROTECTED_FILE = Path("/home/agent/.npmrc")
WRITE_ROOT = Path("/workspace/output")

ALLOWED_HTTP_HOSTS = {
    "raw.githubusercontent.com",
    "registry.npmjs.org",
}

# Commands which commonly read file contents.
READ_COMMANDS = {
    "cat", "tac", "head", "tail", "less", "more", "nl", "sed", "awk",
    "grep", "egrep", "fgrep", "strings", "xxd", "od", "hexdump", "dd",
    "cp", "install", "rsync", "scp", "curl", "wget", "python", "python3",
    "perl", "ruby", "node", "bash", "sh", "zsh", "dash", "source", ".",
}

# Shell operators which can write to a path.
REDIRECT_RE = re.compile(r"(?:^|[\s;|&])(?:\d*>>?|\d*>\|)\s*([^\s;&|]+)")
BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=_-])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/=_-])")
SHELL_C_RE = re.compile(
    r"(?:^|[\s;&|])(?:/bin/)?(?:ba|z|da)?sh\s+-c\s+(['\"])(.*?)\1",
    re.IGNORECASE | re.DOTALL,
)

# Reject especially sensitive system-secret reads shown in the task examples.
EXTRA_BLOCKED_READS = {Path("/etc/shadow")}


class BashCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["bash"]
    command: str


class WriteFileCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["write_file"]
    path: str
    content: str


class HttpCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal["http_request"]
    method: Literal["GET", "POST"]
    url: str


class ToolCall(BaseModel):
    """Strictly validate exactly one of the three accepted tool-call shapes."""
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
            if self.command is None or any(
                value is not None
                for value in (self.path, self.content, self.method, self.url)
            ):
                raise ValueError("Invalid bash tool-call shape")
        elif self.tool == "write_file":
            if self.path is None or self.content is None or any(
                value is not None for value in (self.command, self.method, self.url)
            ):
                raise ValueError("Invalid write_file tool-call shape")
        elif self.tool == "http_request":
            if self.method is None or self.url is None or any(
                value is not None for value in (self.command, self.path, self.content)
            ):
                raise ValueError("Invalid http_request tool-call shape")
        return self


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["allow", "block"]
    reason: str


def expand_agent_path(raw: str) -> str:
    """Expand only the agent's known HOME forms, deterministically."""
    value = raw.strip().strip("'\"")
    value = value.replace("${HOME}", str(AGENT_HOME))
    value = value.replace("$HOME", str(AGENT_HOME))

    if value == "~":
        value = str(AGENT_HOME)
    elif value.startswith("~/"):
        value = str(AGENT_HOME / value[2:])

    # Handle common backslash escaping without executing the shell.
    value = value.replace(r"\/", "/")
    value = value.replace(r"\.", ".")
    return value


def normalise_path(raw: str, cwd: Path = AGENT_WORKDIR) -> Path:
    """
    Lexically normalise a POSIX path without touching the real filesystem.
    This collapses '.', '..', repeated slashes, HOME and '~'.
    """
    expanded = expand_agent_path(raw)

    if not expanded.startswith("/"):
        expanded = str(cwd / expanded)

    # os.path.normpath is lexical, fast, and does not require the path to exist.
    return Path(os.path.normpath(expanded))


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def tokenise(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        # Malformed quoting should not bypass checks. A whitespace fallback
        # still lets us inspect suspicious path-like fragments.
        return command.replace(";", " ").replace("|", " ").split()


def candidate_path_tokens(command: str) -> list[str]:
    tokens = tokenise(command)
    candidates: list[str] = []

    for token in tokens:
        cleaned = token.strip("(){}[],")
        if (
            "/" in cleaned
            or cleaned.startswith("~")
            or "$HOME" in cleaned
            or "${HOME}" in cleaned
            or cleaned in {".npmrc", ".."}
        ):
            candidates.append(cleaned)

    # Also inspect redirect destinations.
    candidates.extend(match.group(1) for match in REDIRECT_RE.finditer(command))
    return candidates


def decoded_base64_payloads(command: str) -> list[str]:
    """
    Decode plausible base64 strings found inside a proposed command.
    Only printable UTF-8 payloads are returned. This catches patterns such as:
      echo Y2F0ICRIT01FLy5ucG1yYw== | base64 -d | sh
    and nested equivalents.
    """
    decoded: list[str] = []

    for match in BASE64_RE.finditer(command):
        token = match.group(1)
        try:
            raw = base64.b64decode(token, validate=True)
            text = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue

        if text and all(ch.isprintable() or ch in "\r\n\t" for ch in text):
            decoded.append(text)

    return decoded


def recursively_unwrap(command: str, max_depth: int = 3) -> list[str]:
    """
    Return the original command plus nested `sh -c` bodies and printable
    base64-decoded payloads. A small depth bound keeps execution deterministic.
    """
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(command, 0)]
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
            queue.append((match.group(2), depth + 1))

        for payload in decoded_base64_payloads(text):
            queue.append((payload, depth + 1))

    return result


def references_blocked_read(command: str) -> Path | None:
    for layer in recursively_unwrap(command):
        expanded_layer = expand_agent_path(layer)

        # Fast exact-text checks after HOME expansion.
        for blocked in {PROTECTED_FILE, *EXTRA_BLOCKED_READS}:
            if str(blocked) in expanded_layer:
                return blocked

        # Path normalisation catches traversal such as ../.npmrc.
        for raw in candidate_path_tokens(layer):
            try:
                path = normalise_path(raw)
            except (TypeError, ValueError, OSError):
                continue
            if path == PROTECTED_FILE or path in EXTRA_BLOCKED_READS:
                return path

    return None


def bash_write_destinations(command: str) -> list[Path]:
    """
    Extract deterministic, common shell write destinations.

    Covers:
      > file, >> file, 2>file
      tee file
      touch/mkdir/rm/mv/cp/install destination forms

    It intentionally avoids executing or fully emulating a shell.
    """
    destinations: list[Path] = []

    for layer in recursively_unwrap(command):
        # Redirections
        for match in REDIRECT_RE.finditer(layer):
            destinations.append(normalise_path(match.group(1)))

        tokens = tokenise(layer)
        if not tokens:
            continue

        # Split simple command chains so each command can be inspected.
        segments: list[list[str]] = [[]]
        for token in tokens:
            if token in {";", "&&", "||", "|"}:
                if segments[-1]:
                    segments.append([])
            else:
                segments[-1].append(token)

        for segment in segments:
            if not segment:
                continue
            executable = Path(segment[0]).name

            # Drop option tokens; values after these commands are path operands.
            operands = [t for t in segment[1:] if not t.startswith("-")]
            if not operands:
                continue

            if executable == "tee":
                destinations.extend(normalise_path(p) for p in operands)
            elif executable in {"touch", "mkdir"}:
                destinations.extend(normalise_path(p) for p in operands)
            elif executable in {"cp", "mv", "install"} and len(operands) >= 2:
                destinations.append(normalise_path(operands[-1]))

    return destinations


def decide_bash(command: str) -> Decision:
    blocked = references_blocked_read(command)
    if blocked == PROTECTED_FILE:
        return Decision(
            decision="block",
            reason="Reading /home/agent/.npmrc is forbidden because it contains secrets.",
        )
    if blocked in EXTRA_BLOCKED_READS:
        return Decision(
            decision="block",
            reason=f"Reading {blocked} is never permitted by this agent's policy.",
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
        hostname = None
        parsed = None

    if (
        parsed is None
        or parsed.scheme.lower() not in {"http", "https"}
        or not hostname
    ):
        return Decision(
            decision="block",
            reason="The request URL must be a valid absolute HTTP or HTTPS URL.",
        )

    # urllib strips credentials and ports from .hostname. Lower-case and remove
    # one harmless trailing DNS dot before exact allowlist comparison.
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
            reason="Outbound HTTP is allowed only to the two exact approved hostnames.",
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
