"""One-shot smoke test of the LLM selector's stage-1 call against the configured endpoint."""

import logging
import time
from datetime import date

from sota.config import Settings
from sota.features import MarketFeatures
from sota.selectors.base import PortfolioView
from sota.selectors.llm import LLMSelector

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

s = Settings()
print(
    f"endpoint={s.llm.base_url} model={s.llm.model} effort={s.llm.reasoning_effort} "
    f"key={'set' if s.llm.api_key else 'MISSING'} timeout={s.llm.timeout_s}s"
)

nan = float("nan")
asof = date(2026, 10, 12)
feats = {
    # SPY: iv rich vs rv, backwardated, steep put skew -> a classic short-vol/defined-risk setup
    "SPY": MarketFeatures(
        "SPY", asof, 600.0, -0.012, 0.14, 0.21, 0.03, 0.07, -0.02, 0.045, 0.012, nan, 0.01
    ),
    # NVDA: iv ~ rv, flat term structure -> should get few or no candidates
    "NVDA": MarketFeatures(
        "NVDA", asof, 180.0, 0.02, 0.45, 0.42, -0.01, -0.03, 0.01, 0.02, 0.01, nan, -0.005
    ),
}
view = PortfolioView(asof, 50_000.0, 50_000.0, [])

sel = LLMSelector(s.llm, s.limits)
t0 = time.time()
props = sel.propose(feats, view)
print(f"stage-1: {len(props)} proposals in {time.time() - t0:.1f}s")
for p in props:
    print(
        f"  {p.underlying:5s} {p.family:3s} {p.side.value:5s} {p.tenor.value:6s} {p.coords} | {p.rationale[:100]}"
    )
