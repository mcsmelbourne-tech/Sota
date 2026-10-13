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


def _selector(handler, limits: RiskLimits | None = None, cache_dir=None) -> LLMSelector:
    settings = LLMSettings(
        base_url="https://api.x.ai/v1",
        api_key="k",
        model="grok-4.7",
        reasoning_effort="low",
        cache_dir=cache_dir,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return LLMSelector(settings, limits or RiskLimits(), client=client, sleep=lambda _: None)


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

    sel = _selector(handler)
    assert sel.propose({"SPY": _features()}, _view()) == []
    assert sel.usage.failures == 1 and sel.usage.retries == 0, "401 is not retried"


def _ok(reply: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "grok-4.7",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
            "choices": [{"message": {"content": json.dumps(reply)}}],
        },
    )


def test_transient_errors_are_retried_with_backoff() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(429, headers={"retry-after": "3"}, json={"error": "slow"})
        if len(attempts) == 2:
            return httpx.Response(503, text="upstream")
        return _ok({"candidates": [{"asset": "asset_1", "family": "ic"}]})

    sel = _selector(handler)
    assert len(sel.propose({"SPY": _features()}, _view())) == 1
    assert len(attempts) == 3 and sel.usage.retries == 2 and sel.usage.failures == 0
    assert sel.usage.prompt_tokens == 100 and sel.usage.reasoning_tokens == 7
    assert sel.usage.models == {"grok-4.7"}


def test_replies_are_cached_by_request_hash(tmp_path) -> None:
    hits: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(1)
        return _ok({"candidates": [{"asset": "asset_1", "family": "cv", "side": "short"}]})

    sel = _selector(handler, cache_dir=tmp_path)
    first = sel.propose({"SPY": _features()}, _view())
    again = _selector(handler, cache_dir=tmp_path).propose({"SPY": _features()}, _view())
    assert len(hits) == 1, "an identical request is served from disk"
    assert [(p.family, p.side) for p in first] == [(p.family, p.side) for p in again]
    assert list(tmp_path.rglob("*.json")) and not list(tmp_path.rglob("*.tmp"))

    # Same numbers under another ticker hash identically: the ticker never reaches the model.
    _selector(handler, cache_dir=tmp_path).propose({"NVDA": _features("NVDA")}, _view())
    assert len(hits) == 1

    nan = float("nan")
    other = MarketFeatures(
        "SPY", date(2025, 1, 2), 500.0, 0.02, 0.30, 0.25, 0.0, -0.05, 0.01, 0.01, 0.0, nan, 0.0
    )
    _selector(handler, cache_dir=tmp_path).propose({"SPY": other}, _view())
    assert len(hits) == 2, "different features -> different hash -> a real call"


def test_system_prompt_reflects_the_risk_profile() -> None:
    from sota.selectors.llm import build_system_prompt

    retail = build_system_prompt(RiskLimits(hedge=False, families=["cv", "ic", "bf"]))
    assert "do NOT delta-hedge" in retail
    assert "- cv (" in retail and "- ic (" in retail and "- bf (" in retail
    assert "- ls (" not in retail and "- ol (" not in retail

    paper = build_system_prompt(RiskLimits())
    assert "delta-hedge volatility families with shares" in paper
    assert all(f"- {code} (" in paper for code in ("ol", "ls", "lg", "ib"))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][0]["content"] == retail
        assert body["temperature"] == 0.0
        return _ok({"candidates": []})

    sel = _selector(handler, RiskLimits(hedge=False, families=["cv", "ic", "bf"]))
    assert sel.propose({"SPY": _features()}, _view()) == []
