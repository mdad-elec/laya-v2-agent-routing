"""Backends: how a cell (`provider/model@effort`) is asked. Each returns (text, usage).

GatewayBackend speaks the OpenAI chat-completions wire to any gateway that routes a NAMED model to
its provider -- the dss-platform gateway is how the CLI seats (Claude Code, Codex) are reached, and
an OpenAI-compatible endpoint (vLLM, SGLang, NIM) speaks the same wire directly. The credential is
read from an environment variable by NAME and sent as the whole Authorization header; it never
appears in an argument, a log line or an error.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from harness.runner import SeatLimit

_LIMIT_WORDS = ("usage limit", "session limit", "rate limit", "rate_limited", "quota", "spent its allowance")


class GatewayBackend:
    def __init__(self, base_url: str, token_env: str | None, catalogue: dict, timeout_s: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.token_env = token_env
        self.timeout_s = timeout_s
        self.efforts = {m["model"]: m["efforts"] for m in catalogue["models"]}

    def _body(self, cell: str, messages: list) -> dict:
        model_ref, _, effort = cell.partition("@")
        declared = self.efforts.get(model_ref)
        if declared is None:
            raise ValueError(f"{model_ref} is not in the catalogue")
        if effort not in declared:
            raise ValueError(f"{model_ref} does not declare effort {effort!r} (declares: {', '.join(declared)})")
        body = {"model": model_ref.split("/", 1)[1], "messages": messages, "stream": False}
        # "none" as the ONLY entry means the model declares no efforts: send no field. Codex declares
        # "none" as a real effort id among others: send it.
        if declared != ["none"]:
            body["reasoning_effort"] = effort
        return body

    def ask(self, cell: str, messages: list) -> tuple[str, dict]:
        body = self._body(cell, messages)
        headers = {"content-type": "application/json"}
        if self.token_env:
            token = os.environ.get(self.token_env, "").strip()
            if not token:
                raise RuntimeError(f"credential {self.token_env} is not set")
            headers["Authorization"] = token
        request = urllib.request.Request(f"{self.base_url}/api/v1/chat/completions" if "/v1" not in self.base_url else f"{self.base_url}/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers, method="POST")
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:600]
            if e.code == 429 or any(w in detail.lower() for w in _LIMIT_WORDS):
                raise SeatLimit(f"HTTP {e.code}: {detail[:200]}") from None
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"transport: {e.reason}") from None
        usage = payload.get("usage") or {}
        text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return text, {"latency_ms": int((time.monotonic() - started) * 1000),
                      "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
