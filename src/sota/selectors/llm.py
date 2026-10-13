"""LLM analyst selector: anonymized market state -> JSON strategy candidates -> price-aware final picks.

Mirrors the paper's frontier-teacher stage (§3.1, §5.1): ticker, price level and calendar date are
masked; news is excluded by default. Works with any OpenAI-compatible chat-completions endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from sota.config import LLMSettings, RiskLimits
from sota.features import MarketFeatures
from sota.models import Side, Tenor
from sota.selectors.base import (
    BaseSelector,
    PortfolioView,
    PricedCandidate,
    Proposal,
    SelectorUnavailable,
)
from sota.strategies.families import FAMILIES, describe_families

log = logging.getLogger(__name__)

_CHAT_ERRORS = (httpx.HTTPError, ValidationError, ValueError, KeyError)
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_PERMANENT_STATUS = {401, 402, 403, 404}  # credentials, billing/quota, unknown model


def _error_text(r: httpx.Response) -> str:
    try:
        body = r.json()
        msg = body.get("error") if isinstance(body, dict) else None
        if isinstance(msg, dict):
            msg = msg.get("message")
        if msg:
            return str(msg)[:300]
    except ValueError:
        pass
    return r.text[:300]


class CandidateModel(BaseModel):
    asset: str
    family: str
    side: Literal["long", "short"] = "long"
    tenor: Literal["0-7", "8-30", "31-90", "91-180"] = "8-30"
    coords: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""

    @field_validator("family")
    @classmethod
    def _known_family(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in FAMILIES:
            raise ValueError(f"unknown family {v!r}")
        return v


class StageOneResponse(BaseModel):
    candidates: list[CandidateModel] = Field(default_factory=list)


class StageTwoResponse(BaseModel):
    chosen: list[int] = Field(default_factory=list)
    rationale: str = ""


def _round(x: float, nd: int = 4) -> float | None:
    return None if x is None or math.isnan(x) else round(x, nd)


def _parse_json(text: str) -> dict[str, Any]:
    """Outermost JSON object in a reply; tolerates code fences or prose around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in model reply")
    return json.loads(text[start : end + 1])


@dataclass
class Usage:
    """Per-session ledger so a backtest can report what the selector cost."""

    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    seconds: float = 0.0
    models: set[str] = field(default_factory=set)

    def summary(self) -> str:
        return (
            f"{self.calls} calls ({self.cache_hits} cached, {self.retries} retries, "
            f"{self.failures} failed), {self.prompt_tokens:,} in / {self.completion_tokens:,} out "
            f"(+{self.reasoning_tokens:,} reasoning) tokens, {self.seconds / 60:.1f} min, "
            f"model {', '.join(sorted(self.models)) or '?'}"
        )


def build_system_prompt(limits: RiskLimits | None = None) -> str:
    """System prompt describing exactly the strategy space the resolvers will accept."""
    codes = limits.families if limits is not None and limits.families else None
    hedged = limits.hedge if limits is not None else True
    hedge_line = (
        "and delta-hedge volatility families with shares"
        if hedged
        else "but do NOT delta-hedge: long straddles/strangles become directional bets as spot moves"
    )
    return f"""You are an options strategist selecting option structures for a risk-managed portfolio.
You see anonymized per-asset market states built from the option-implied return distribution.
Prices, tickers and dates are masked on purpose; reason only from the numbers given.

Features per asset: ret (session log return), rv (21d realized vol), iv (30d ATM implied vol),
dv (change in iv), wedge (iv - rv), ts (90d - 30d ATM iv; negative = backwardation),
sk (25d put iv - 25d call iv), bf (25d butterfly curvature), fi (option flow imbalance),
doi (change in open interest), iv_rv (iv / rv), sk_iv, bf_iv. A null means unavailable.

Strategy families you may use (code, category, meaning, default coordinates [admitted range]):
{describe_families(codes)}

Tenor buckets (calendar days to expiry): "0-7", "8-30", "31-90", "91-180".
Side: for directional families "long" = bullish, "short" = bearish; for ls/lg/ib/ic it is ignored;
for bf "long" uses calls, "short" uses puts.

Deterministic resolvers will choose exact strikes, size every position to equal scenario risk,
{hedge_line}. Positions carry a 60%-of-premium stop and are closed one day before expiry.
Every trade pays the bid-ask spread, so trade only when the edge clearly exceeds it; proposing
nothing for an asset is a valid and common answer. Respond with JSON only."""


SYSTEM_PROMPT = build_system_prompt()


class LLMSelector(BaseSelector):
    name = "llm"

    def __init__(
        self,
        settings: LLMSettings,
        limits: RiskLimits,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.limits = limits
        self.client = client or httpx.Client(timeout=settings.timeout_s)
        self.system_prompt = build_system_prompt(limits)
        self.usage = Usage()
        self._sleep = sleep
        self._cache_dir = Path(settings.cache_dir) if settings.cache_dir else None
        self.disabled: str | None = None  # reason, once a permanent error has been seen

    # ------------------------------------------------------------------ transport
    def _request_body(self, user: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user},
            ],
        }
        if self.settings.reasoning_effort:
            body["reasoning_effort"] = self.settings.reasoning_effort
        return body

    def _cache_path(self, body: dict[str, Any]) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        return self._cache_dir / digest[:2] / f"{digest}.json"

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key is not None:
            headers["Authorization"] = f"Bearer {self.settings.api_key.get_secret_value()}"
        url = f"{self.settings.base_url.rstrip('/')}/chat/completions"
        delay = 2.0
        for attempt in range(self.settings.max_retries + 1):
            try:
                r = self.client.post(url, json=body, headers=headers)
                if r.status_code in _PERMANENT_STATUS:
                    self.disabled = f"HTTP {r.status_code} from {url}: {_error_text(r)}"
                    raise SelectorUnavailable(self.disabled)
                if r.status_code in _RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"{r.status_code}: {r.text[:200]}", request=r.request, response=r
                    )
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == self.settings.max_retries or (
                    status is not None and status not in _RETRY_STATUS
                ):
                    raise
                self.usage.retries += 1
                retry_after = getattr(exc, "response", None)
                wait = delay
                if retry_after is not None and retry_after.headers.get("retry-after", "").isdigit():
                    wait = max(wait, float(retry_after.headers["retry-after"]))
                log.warning("llm request failed (%s); retry %d in %.0fs", exc, attempt + 1, wait)
                self._sleep(wait)
                delay *= 2
        raise RuntimeError("unreachable")

    def _chat(self, user: str) -> dict[str, Any]:
        if self.disabled:
            raise SelectorUnavailable(self.disabled)
        body = self._request_body(user)
        path = self._cache_path(body)
        self.usage.calls += 1
        if path is not None and path.exists():
            self.usage.cache_hits += 1
            data = json.loads(path.read_text())
        else:
            t0 = time.monotonic()
            try:
                data = self._post(body)
            except Exception:
                self.usage.failures += 1
                raise
            finally:
                self.usage.seconds += time.monotonic() - t0
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(data))
                tmp.replace(path)
        if model := data.get("model"):
            self.usage.models.add(str(model))
        if usage := data.get("usage"):
            details = usage.get("completion_tokens_details") or {}
            self.usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.usage.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.usage.reasoning_tokens += int(details.get("reasoning_tokens") or 0)
            log.info(
                "llm usage model=%s in=%s out=%s reasoning=%s",
                data.get("model"),
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
                details.get("reasoning_tokens"),
            )
        return _parse_json(data["choices"][0]["message"]["content"])

    # ------------------------------------------------------------------ anonymization
    @staticmethod
    def anonymize(
        features: dict[str, MarketFeatures], portfolio: PortfolioView
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Rows keyed by `asset_N` with no ticker, price level or date; returns (rows, asset->ticker)."""
        mapping = {f"asset_{i + 1}": u for i, u in enumerate(sorted(features))}
        rows: list[dict[str, Any]] = []
        for asset, u in mapping.items():
            f = features[u]
            d = {k: _round(v) for k, v in f.as_dict().items()}
            iv = f.iv if not math.isnan(f.iv) and f.iv > 0 else float("nan")
            d.update(iv_rv=_round(f.iv_rv_ratio), sk_iv=_round(f.sk / iv), bf_iv=_round(f.bf / iv))
            d["open_positions"] = [
                {
                    "family": p.package.family,
                    "side": p.package.side.value,
                    "tenor": p.package.tenor.value,
                    "dte": (p.package.expiry - portfolio.asof).days,
                    "pnl_pct_of_premium": _round(portfolio.pnl_pct(p) or 0.0, 3),
                }
                for p in portfolio.open_positions(u)
            ]
            rows.append({"asset": asset, **d})
        return rows, mapping

    # ------------------------------------------------------------------ stage 1
    def propose(
        self, features: dict[str, MarketFeatures], portfolio: PortfolioView
    ) -> list[Proposal]:
        if not features:
            return []
        rows, mapping = self.anonymize(features, portfolio)
        q = self.limits.candidates_per_underlying
        user = json.dumps(
            {
                "task": f"Propose up to {q} candidate strategies per asset (fewer or none is fine).",
                "schema": {
                    "candidates": [
                        {
                            "asset": "asset_1",
                            "family": "ic",
                            "side": "long",
                            "tenor": "8-30",
                            "coords": {"delta": 0.30, "width": 0.10},
                            "rationale": "short reason",
                        }
                    ]
                },
                "assets": rows,
            }
        )
        try:
            parsed = StageOneResponse.model_validate(self._chat(user))
        except _CHAT_ERRORS as exc:
            log.warning("LLM stage-1 failed: %s", exc)
            return []

        out: list[Proposal] = []
        per_asset: dict[str, int] = {}
        for c in parsed.candidates:
            u = mapping.get(c.asset)
            if u is None or u not in features or per_asset.get(u, 0) >= q:
                continue
            per_asset[u] = per_asset.get(u, 0) + 1
            out.append(
                Proposal(u, c.family, Side(c.side), Tenor(c.tenor), dict(c.coords), c.rationale)
            )
        return out

    # ------------------------------------------------------------------ stage 2
    def decide(
        self, candidates: list[PricedCandidate], portfolio: PortfolioView, max_lines: int
    ) -> list[PricedCandidate]:
        if not candidates:
            return []
        asset_of = {
            u: f"asset_{i + 1}"
            for i, u in enumerate(sorted({c.package.underlying for c in candidates}))
        }
        priced = []
        for i, c in enumerate(candidates):
            pkg = c.package
            priced.append(
                {
                    "index": i,
                    "asset": asset_of[pkg.underlying],
                    "family": pkg.family,
                    "side": pkg.side.value,
                    "tenor": pkg.tenor.value,
                    "dte": (pkg.expiry - portfolio.asof).days,
                    "net_premium_pct_of_spot": _round(
                        pkg.net_premium / c.package.legs[0].contract.strike, 4
                    ),
                    "max_loss_over_premium": _round(pkg.max_loss / abs(pkg.net_premium), 2)
                    if pkg.max_loss and pkg.net_premium
                    else None,
                    "half_spread_pct_of_premium": _round(pkg.half_spread / abs(pkg.net_premium), 3)
                    if pkg.net_premium
                    else None,
                    "delta": _round(pkg.delta, 3),
                    "vega_over_premium": _round(pkg.vega / abs(pkg.net_premium), 2)
                    if pkg.net_premium
                    else None,
                    "theta_over_premium": _round(pkg.theta / abs(pkg.net_premium), 4)
                    if pkg.net_premium
                    else None,
                    "units": c.units,
                    "scenario_risk_pct_nav": _round(c.risk_per_unit * c.units / portfolio.nav, 4),
                    "rationale": c.proposal.rationale,
                }
            )
        user = json.dumps(
            {
                "task": f"Pick at most {max_lines} candidates to open now (can be none). Avoid wide spreads relative to premium.",
                "schema": {"chosen": [0, 3], "rationale": "short reason"},
                "candidates": priced,
            }
        )
        try:
            parsed = StageTwoResponse.model_validate(self._chat(user))
        except _CHAT_ERRORS as exc:
            log.warning("LLM stage-2 failed, falling back to first %d: %s", max_lines, exc)
            return candidates[:max_lines]
        chosen = [candidates[i] for i in dict.fromkeys(parsed.chosen) if 0 <= i < len(candidates)]
        return chosen[:max_lines]
