from __future__ import annotations

import json
import re
from typing import Any, Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(
    title="Agent Run Budget and Loop Guard",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_number: int
    tool: str
    args: dict[str, Any]
    tokens_used: int = Field(ge=0)

class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    budget_tokens: int = Field(ge=0)
    steps: list[Step]

class CheckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["continue", "halt"]
    reason: str

def normalize_string(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()

def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonicalize(v) for k, v in sorted(value.items()) if k != "client_ts"}
    if isinstance(value, list):
        return [canonicalize(v) for v in value]
    if isinstance(value, str):
        return normalize_string(value)
    return value

def call_signature(step: Step):
    return (
        step.tool,
        json.dumps(
            canonicalize(step.args),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
    )

def has_three_identical_trailing_calls(signatures):
    return len(signatures) >= 3 and signatures[-1] == signatures[-2] == signatures[-3]

def trailing_two_step_cycle_length(signatures):
    if len(signatures) < 6:
        return 0

    A = signatures[-2]
    B = signatures[-1]

    if A == B:
        return 0

    length = 2
    expected = B
    i = len(signatures) - 3

    while i >= 0:
        if signatures[i] != expected:
            break
        length += 1
        expected = A if expected == B else B
        i -= 1

    return length

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/check", response_model=CheckResponse)
def check_run(request: CheckRequest):
    total = sum(s.tokens_used for s in request.steps)

    if total >= request.budget_tokens:
        return CheckResponse(
            decision="halt",
            reason=f"Cumulative tokens_used ({total}) has reached the budget ({request.budget_tokens}).",
        )

    sigs = [call_signature(s) for s in request.steps]

    if has_three_identical_trailing_calls(sigs):
        return CheckResponse(
            decision="halt",
            reason="The same tool was called three or more times in a row with functionally identical arguments.",
        )

    cycle = trailing_two_step_cycle_length(sigs)
    if cycle >= 6:
        return CheckResponse(
            decision="halt",
            reason=f"The trailing {cycle} steps form a repeated two-step cycle.",
        )

    return CheckResponse(
        decision="continue",
        reason="Budget remains and no prohibited loop was detected.",
    )
