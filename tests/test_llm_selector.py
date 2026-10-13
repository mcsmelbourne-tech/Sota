"""LLM selector transport: request shape, tolerant JSON parsing, graceful failure (no network)."""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from sota.config import LLMSettings, RiskLimits
from sota.features import MarketFeatures
from sota.selectors.base import PortfolioView
from sota.selectors.llm import SYSTEM_PROMPT, LLMSelector, _parse_json


def _features(u: str = "SPY") -> MarketFeatures:
    nan = float("nan")
    return MarketFeatures(
        u, date(2025, 1, 2), 500.0, 0.001, 0.12, 0.15, 0.0, 0.03, -0.01, 0.02, 0.01, nan, 0.0
    )


def _view() -> PortfolioView:
    return PortfolioView(date(2025, 1, 2), 50_000.0, 50_000.0, [])


def _selector(handler) -> LLMSelector:
    settings = LLMSettings(
        base_url="https://api.x.ai/v1", api_key="k", model="grok-4.7", reasoning_effort="low"
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LLMSelector(settings, RiskLimits(), client=client)


def test_parse_json_tolerates_fences_and_prose() -> None:
    assert _parse_json('{"a": 1}') == {"a": 1}
    assert _parse_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert _parse_json('Sure:\n{"chosen": [0], "rationale": "x"}\nHope it helps') == {
        "chosen": [0],
        "rationale": "x",
    }
    with pytest.raises(ValueError):
        _parse_json("no json here")


def test_propose_sends_openai_request_and_parses_candidates() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        reply = {
            "candidates": [
                {"asset": "asset_1", "family": "ic", "tenor": "8-30", "rationale": "rich iv"},
                {"asset": "asset_9", "family": "ic"},  # unknown asset -> dropped
            ]
        }
        return httpx.Response(
            200,
            json={
                "model": "grok-4.7",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "choices": [{"message": {"content": f"```json\n{json.dumps(reply)}\n```"}}],
            },
        )

    proposals = _selector(handler).propose({"SPY": _features()}, _view())

    assert [(p.underlying, p.family, p.tenor.value) for p in proposals] == [("SPY", "ic", "8-30")]
    req = seen[0]
    assert str(req.url) == "https://api.x.ai/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer k"
    body = json.loads(req.content)
    assert body["model"] == "grok-4.7"
    assert body["reasoning_effort"] == "low"
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "SPY" not in body["messages"][1]["content"]  # anonymized


def test_http_error_degrades_to_no_proposals() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    assert _selector(handler).propose({"SPY": _features()}, _view()) == []
